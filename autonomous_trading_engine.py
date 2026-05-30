#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
  AUTONOMOUS TRADING SIGNAL ENGINE v1.0
  Analyzes Crypto + Forex pairs across multiple timeframes
  Generates BUY/SELL/NO-TRADE signals with full risk parameters
═══════════════════════════════════════════════════════════════════════════════
"""

import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

CRYPTO_PAIRS = [
    'BTCUSDT', 'ETHUSDT', 'XRPUSDT', 'SOLUSDT', 'BNBUSDT',
    'ADAUSDT', 'DOGEUSDT', 'DOTUSDT', 'AVAXUSDT', 'LINKUSDT',
    'LTCUSDT', 'UNIUSDT', 'ATOMUSDT', 'ETCUSDT', 'APTUSDT',
    'ARBUSDT', 'OPUSDT', 'NEARUSDT', 'FILUSDT', 'MATICUSDT'
]

FOREX_PAIRS = [
    'EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'AUDUSD=X',
    'USDCAD=X', 'USDCHF=X', 'NZDUSD=X'
]

TIMEFRAMES = {
    '1H': {'interval': '1h', 'candles': 100, 'expiry': '2-3 Hours'},
    '4H': {'interval': '4h', 'candles': 100, 'expiry': '8-12 Hours'},
    '1D': {'interval': '1d', 'candles': 60, 'expiry': '1-3 Days'}
}

# Risk parameters
MAX_RISK_PER_TRADE = 0.02  # 2% account risk
MIN_RR_RATIO = 1.5
OVERBOUGHT_RSI = 70
OVERSOLD_RSI = 30

# ═══════════════════════════════════════════════════════════════════════════════
# DATA FETCHING
# ═══════════════════════════════════════════════════════════════════════════════

class DataFetcher:
    """Fetches live market data from Binance (crypto) and Yahoo Finance (forex)"""

    @staticmethod
    def fetch_crypto_prices(symbols):
        """Fetch 24h ticker data from Binance"""
        url = "https://api.binance.com/api/v3/ticker/24hr"
        params = {'symbols': json.dumps([s for s in symbols])}
        try:
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            df = pd.DataFrame(data)
            df['symbol'] = df['symbol']
            df['price'] = df['lastPrice'].astype(float)
            df['open'] = df['openPrice'].astype(float)
            df['high'] = df['highPrice'].astype(float)
            df['low'] = df['lowPrice'].astype(float)
            df['change'] = df['priceChangePercent'].astype(float)
            df['volume'] = df['volume'].astype(float)
            df['quoteVolume'] = df['quoteVolume'].astype(float)
            return df[['symbol', 'price', 'open', 'high', 'low', 'change', 'volume', 'quoteVolume']]
        except Exception as e:
            print(f"[ERROR] Crypto fetch failed: {e}")
            return pd.DataFrame()

    @staticmethod
    def fetch_crypto_klines(symbol, interval='1h', limit=100):
        """Fetch OHLCV kline data from Binance"""
        url = "https://api.binance.com/api/v3/klines"
        params = {'symbol': symbol, 'interval': interval, 'limit': limit}
        try:
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            return df
        except Exception as e:
            print(f"[ERROR] Kline fetch failed for {symbol}: {e}")
            return pd.DataFrame()

    @staticmethod
    def fetch_forex_data(ticker, period='1mo', interval='1h'):
        """Fetch forex data using Yahoo Finance API"""
        # Using Yahoo Finance API via rapidapi or direct
        # For this engine, we'll use a simplified approach
        # In production, use yfinance library or dedicated forex API
        try:
            import yfinance as yf
            data = yf.download(ticker, period=period, interval=interval, progress=False)
            if not data.empty:
                data = data.reset_index()
                data.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
                return data
            return pd.DataFrame()
        except ImportError:
            print("[WARNING] yfinance not installed. Install with: pip install yfinance")
            return pd.DataFrame()
        except Exception as e:
            print(f"[ERROR] Forex fetch failed for {ticker}: {e}")
            return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════════════════
# TECHNICAL ANALYSIS ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class TechnicalAnalyzer:
    """Calculates all technical indicators"""

    @staticmethod
    def rsi(prices, period=14):
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def ema(prices, period):
        return prices.ewm(span=period, adjust=False).mean()

    @staticmethod
    def macd(prices, fast=12, slow=26, signal=9):
        ema_fast = TechnicalAnalyzer.ema(prices, fast)
        ema_slow = TechnicalAnalyzer.ema(prices, slow)
        macd_line = ema_fast - ema_slow
        signal_line = TechnicalAnalyzer.ema(macd_line, signal)
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    @staticmethod
    def bollinger_bands(prices, period=20, std_dev=2):
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, sma, lower

    @staticmethod
    def atr(df, period=14):
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        return true_range.rolling(period).mean()

    @staticmethod
    def fibonacci_levels(high, low):
        diff = high - low
        return {
            '0.0': high,
            '0.236': high - 0.236 * diff,
            '0.382': high - 0.382 * diff,
            '0.5': high - 0.5 * diff,
            '0.618': high - 0.618 * diff,
            '0.786': high - 0.786 * diff,
            '1.0': low
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SIGNAL GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

class SignalGenerator:
    """Generates trading signals with full risk parameters"""

    def __init__(self):
        self.analyzer = TechnicalAnalyzer()

    def analyze_crypto(self, ticker_data, kline_data, timeframe_config):
        """Analyze a crypto pair and generate signal"""
        if kline_data.empty or len(kline_data) < 50:
            return None

        close = kline_data['close']
        high = kline_data['high']
        low = kline_data['low']

        # Calculate indicators
        rsi = self.analyzer.rsi(close).iloc[-1]
        ema50 = self.analyzer.ema(close, 50).iloc[-1]
        ema200 = self.analyzer.ema(close, 200).iloc[-1] if len(close) >= 200 else ema50
        macd_line, signal_line, hist = self.analyzer.macd(close)
        macd_val = macd_line.iloc[-1]
        signal_val = signal_line.iloc[-1]
        hist_val = hist.iloc[-1]

        bb_upper, bb_sma, bb_lower = self.analyzer.bollinger_bands(close)
        atr = self.analyzer.atr(kline_data).iloc[-1]

        current_price = close.iloc[-1]
        daily_high = high.max()
        daily_low = low.min()

        # Fibonacci levels
        fibs = self.analyzer.fibonacci_levels(daily_high, daily_low)

        # Determine trend
        trend = 'Bullish' if current_price > ema50 > ema200 else \
                'Bearish' if current_price < ema50 < ema200 else 'Mixed'

        # Generate signal
        signal = self._generate_signal(
            current_price, rsi, macd_val, signal_val, hist_val,
            ema50, ema200, trend, daily_high, daily_low
        )

        # Calculate risk parameters
        risk_params = self._calculate_crypto_risk(
            signal, current_price, daily_high, daily_low, atr, fibs
        )

        return {
            'pair': ticker_data['symbol'],
            'price': round(current_price, 4),
            'timeframe': timeframe_config,
            'timestamp': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC'),
            'rsi': round(rsi, 2) if not pd.isna(rsi) else 50,
            'ema50': round(ema50, 4),
            'ema200': round(ema200, 4),
            'macd': round(macd_val, 6),
            'signal_line': round(signal_val, 6),
            'histogram': round(hist_val, 6),
            'trend': trend,
            'direction': signal['direction'],
            'confidence': signal['confidence'],
            'entry': risk_params['entry'],
            'tp1': risk_params['tp1'],
            'tp2': risk_params['tp2'],
            'sl': risk_params['sl'],
            'rr_ratio': risk_params['rr_ratio'],
            'position_size': risk_params['position_size'],
            'expiry': timeframe_config['expiry'],
            'reason': signal['reason'],
            'daily_high': round(daily_high, 4),
            'daily_low': round(daily_low, 4),
            'atr': round(atr, 4) if not pd.isna(atr) else 0
        }

    def analyze_forex(self, df, pair_name, timeframe_config):
        """Analyze a forex pair and generate signal"""
        if df.empty or len(df) < 50:
            return None

        close = df['Close']
        high = df['High']
        low = df['Low']

        rsi = self.analyzer.rsi(close).iloc[-1]
        ema50 = self.analyzer.ema(close, 50).iloc[-1]
        ema200 = self.analyzer.ema(close, 200).iloc[-1] if len(close) >= 200 else ema50
        macd_line, signal_line, hist = self.analyzer.macd(close)
        macd_val = macd_line.iloc[-1]
        signal_val = signal_line.iloc[-1]
        hist_val = hist.iloc[-1]

        bb_upper, bb_sma, bb_lower = self.analyzer.bollinger_bands(close)
        atr = self.analyzer.atr(df).iloc[-1]

        current_price = close.iloc[-1]
        swing_high = high.tail(50).max()
        swing_low = low.tail(50).min()

        trend = 'Bullish' if current_price > ema50 > ema200 else \
                'Bearish' if current_price < ema50 < ema200 else 'Mixed'

        signal = self._generate_signal(
            current_price, rsi, macd_val, signal_val, hist_val,
            ema50, ema200, trend, swing_high, swing_low
        )

        risk_params = self._calculate_forex_risk(
            signal, current_price, swing_high, swing_low, atr
        )

        return {
            'pair': pair_name,
            'price': round(current_price, 5),
            'timeframe': timeframe_config,
            'timestamp': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC'),
            'rsi': round(rsi, 2) if not pd.isna(rsi) else 50,
            'ema50': round(ema50, 5),
            'ema200': round(ema200, 5),
            'macd': round(macd_val, 5),
            'signal_line': round(signal_val, 5),
            'histogram': round(hist_val, 5),
            'trend': trend,
            'direction': signal['direction'],
            'confidence': signal['confidence'],
            'entry': risk_params['entry'],
            'tp1': risk_params['tp1'],
            'tp2': risk_params['tp2'],
            'sl': risk_params['sl'],
            'rr_ratio': risk_params['rr_ratio'],
            'position_size': risk_params['position_size'],
            'expiry': timeframe_config['expiry'],
            'reason': signal['reason'],
            'swing_high': round(swing_high, 5),
            'swing_low': round(swing_low, 5),
            'atr': round(atr, 5) if not pd.isna(atr) else 0
        }

    def _generate_signal(self, price, rsi, macd, signal, hist, ema50, ema200, trend, high, low):
        """Core signal generation logic"""

        # Overbought/Oversold conditions
        if rsi > OVERBOUGHT_RSI and macd < signal and hist < 0:
            return {
                'direction': 'SELL',
                'confidence': 'HIGH' if rsi > 75 else 'MEDIUM',
                'reason': f'RSI overbought ({rsi:.1f}) + MACD bearish crossover'
            }

        if rsi < OVERSOLD_RSI and macd > signal and hist > 0:
            return {
                'direction': 'BUY',
                'confidence': 'HIGH' if rsi < 25 else 'MEDIUM',
                'reason': f'RSI oversold ({rsi:.1f}) + MACD bullish crossover'
            }

        # Trend-following conditions
        if trend == 'Bullish' and macd > signal and hist > 0 and price > ema50:
            return {
                'direction': 'BUY',
                'confidence': 'MEDIUM',
                'reason': f'Bullish trend + MACD bullish + price above EMA50'
            }

        if trend == 'Bearish' and macd < signal and hist < 0 and price < ema50:
            return {
                'direction': 'SELL',
                'confidence': 'MEDIUM',
                'reason': f'Bearish trend + MACD bearish + price below EMA50'
            }

        # Mixed conditions
        if macd > signal and hist > 0 and price > ema50:
            return {
                'direction': 'BUY',
                'confidence': 'LOW',
                'reason': 'MACD bullish + price above EMA50'
            }

        if macd < signal and hist < 0 and price < ema50:
            return {
                'direction': 'SELL',
                'confidence': 'LOW',
                'reason': 'MACD bearish + price below EMA50'
            }

        return {
            'direction': 'NO TRADE',
            'confidence': 'N/A',
            'reason': 'No clear confluence of indicators'
        }

    def _calculate_crypto_risk(self, signal, price, high, low, atr, fibs):
        """Calculate risk parameters for crypto"""
        if signal['direction'] == 'NO TRADE':
            return {'entry': None, 'tp1': None, 'tp2': None, 'sl': None, 'rr_ratio': None, 'position_size': 0}

        daily_range = high - low

        if signal['direction'] == 'BUY':
            entry = fibs['0.618']  # Buy at 61.8% retracement
            tp1 = high * 0.995  # Near daily high
            tp2 = high + (daily_range * 0.5)  # Extended target
            sl = low - (atr * 1.5)  # Below daily low + ATR buffer
        else:  # SELL
            entry = fibs['0.382']  # Sell at 38.2% retracement
            tp1 = low * 1.005  # Near daily low
            tp2 = low - (daily_range * 0.5)  # Extended target
            sl = high + (atr * 1.5)  # Above daily high + ATR buffer

        risk = abs(entry - sl)
        reward1 = abs(tp1 - entry)
        reward2 = abs(tp2 - entry)

        rr_ratio = round(reward1 / risk, 2) if risk > 0 else 0

        # Position sizing: risk 2% of account
        position_size = round(MAX_RISK_PER_TRADE / (risk / entry), 4) if risk > 0 else 0

        return {
            'entry': round(entry, 4),
            'tp1': round(tp1, 4),
            'tp2': round(tp2, 4),
            'sl': round(sl, 4),
            'rr_ratio': rr_ratio,
            'position_size': position_size
        }

    def _calculate_forex_risk(self, signal, price, swing_high, swing_low, atr):
        """Calculate risk parameters for forex"""
        if signal['direction'] == 'NO TRADE':
            return {'entry': None, 'tp1': None, 'tp2': None, 'sl': None, 'rr_ratio': None, 'position_size': 0}

        range_pips = swing_high - swing_low

        if signal['direction'] == 'BUY':
            entry = price - (atr * 0.5)  # Pullback entry
            tp1 = swing_high
            tp2 = swing_high + (range_pips * 0.5)
            sl = swing_low - (atr * 1.5)
        else:  # SELL
            entry = price + (atr * 0.5)  # Pullback entry
            tp1 = swing_low
            tp2 = swing_low - (range_pips * 0.5)
            sl = swing_high + (atr * 1.5)

        risk = abs(entry - sl)
        reward1 = abs(tp1 - entry)

        rr_ratio = round(reward1 / risk, 2) if risk > 0 else 0
        position_size = round(MAX_RISK_PER_TRADE / (risk / entry), 4) if risk > 0 else 0

        return {
            'entry': round(entry, 5),
            'tp1': round(tp1, 5),
            'tp2': round(tp2, 5),
            'sl': round(sl, 5),
            'rr_ratio': rr_ratio,
            'position_size': position_size
        }


# ═══════════════════════════════════════════════════════════════════════════════
# REPORT GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

class ReportGenerator:
    """Generates formatted trading reports"""

    @staticmethod
    def generate_signal_card(signal):
        """Generate a formatted signal card"""
        if not signal or signal['direction'] == 'NO TRADE':
            return None

        card = f"""
{'═' * 70}
  🤖 AUTONOMOUS SIGNAL: {signal['pair']}
{'═' * 70}
  Timeframe: {signal['timeframe']} | Data Live As Of: {signal['timestamp']}

  📊 TECHNICAL SNAPSHOT
    Current Price:     {signal['price']}
    RSI(14):           {signal['rsi']}
    EMA50:             {signal['ema50']}
    EMA200:            {signal['ema200']}
    MACD:              {signal['macd']} | Signal: {signal['signal_line']}
    Histogram:         {signal['histogram']}
    Trend:             {signal['trend']}
    ATR:               {signal['atr']}

  📈 EXECUTION PARAMETERS
    Direction:         {signal['direction']} ({signal['confidence']} CONFIDENCE)
    Entry Price:       {signal['entry']}
    Take Profit 1:     {signal['tp1']} (Conservative)
    Take Profit 2:     {signal['tp2']} (Extended)
    Stop Loss:         {signal['sl']}
    Risk:Reward:       1:{signal['rr_ratio']}
    Position Size:     {signal['position_size']}x (2% risk)

  ⏱️ BINARY OPTIONS
    Direction:         {'HIGHER (Call)' if signal['direction'] == 'BUY' else 'LOWER (Put)'}
    Expiry:            {signal['expiry']}

  🔍 LOGIC
    {signal['reason']}
{'═' * 70}
"""
        return card

    @staticmethod
    def generate_summary_table(signals):
        """Generate summary table of all signals"""
        valid_signals = [s for s in signals if s and s['direction'] != 'NO TRADE']

        if not valid_signals:
            return "\n[NO TRADE SIGNALS GENERATED]\n"

        table = "\n" + "═" * 100 + "\n"
        table += f"{'PAIR':<12} {'DIR':<6} {'ENTRY':<12} {'TP1':<12} {'TP2':<12} {'SL':<12} {'R:R':<8} {'CONF':<8}\n"
        table += "─" * 100 + "\n"

        for s in valid_signals:
            table += f"{s['pair']:<12} {s['direction']:<6} {str(s['entry']):<12} {str(s['tp1']):<12} {str(s['tp2']):<12} {str(s['sl']):<12} {str(s['rr_ratio']):<8} {s['confidence']:<8}\n"

        table += "═" * 100 + "\n"
        return table

    @staticmethod
    def generate_full_report(crypto_signals, forex_signals, timeframe):
        """Generate complete trading report"""
        report = f"""
{'╔' + '═' * 98 + '╗'}
{'║' + ' ' * 30 + 'AUTONOMOUS TRADING SIGNAL REPORT' + ' ' * 36 + '║'}
{'║' + ' ' * 35 + f'Timeframe: {timeframe}' + ' ' * 42 + '║'}
{'║' + ' ' * 32 + f'Generated: {datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")}' + ' ' * 35 + '║'}
{'╚' + '═' * 98 + '╝'}

{'─' * 100}
  🪙 CRYPTO SIGNALS
{'─' * 100}
"""

        for signal in crypto_signals:
            card = ReportGenerator.generate_signal_card(signal)
            if card:
                report += card

        report += f"""
{'─' * 100}
  💱 FOREX SIGNALS
{'─' * 100}
"""

        for signal in forex_signals:
            card = ReportGenerator.generate_signal_card(signal)
            if card:
                report += card

        all_signals = crypto_signals + forex_signals
        report += ReportGenerator.generate_summary_table(all_signals)

        report += f"""
{'─' * 100}
  ⚠️  RISK MANAGEMENT RULES
{'─' * 100}
  • Maximum risk per trade: 2% of account balance
  • Minimum R:R ratio: 1:{MIN_RR_RATIO}
  • No trading during high-impact news (CPI, NFP, Rate Decisions)
  • All SLs placed beyond structural invalidation points
  • Position sizing calculated automatically based on risk

  ⚠️  DISCLAIMER: This is algorithmic analysis for educational purposes.
  Past performance does not guarantee future results. Trade at your own risk.
{'═' * 100}
"""
        return report


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class TradingSignalEngine:
    """Main autonomous trading signal engine"""

    def __init__(self):
        self.fetcher = DataFetcher()
        self.generator = SignalGenerator()
        self.reporter = ReportGenerator()

    def run_crypto_scan(self, timeframe_key='1H'):
        """Run full crypto scan"""
        print(f"[INFO] Starting crypto scan for timeframe: {timeframe_key}")

        tf_config = TIMEFRAMES[timeframe_key]

        # Fetch all crypto prices
        ticker_data = self.fetcher.fetch_crypto_prices(CRYPTO_PAIRS)

        signals = []
        for _, row in ticker_data.iterrows():
            symbol = row['symbol']
            print(f"[INFO] Analyzing {symbol}...")

            # Fetch klines for technical analysis
            klines = self.fetcher.fetch_crypto_klines(
                symbol, 
                interval=tf_config['interval'], 
                limit=tf_config['candles']
            )

            if not klines.empty:
                signal = self.generator.analyze_crypto(row, klines, tf_config)
                if signal:
                    signals.append(signal)

        return signals

    def run_forex_scan(self, timeframe_key='1H'):
        """Run full forex scan"""
        print(f"[INFO] Starting forex scan for timeframe: {timeframe_key}")

        tf_config = TIMEFRAMES[timeframe_key]
        signals = []

        for pair in FOREX_PAIRS:
            print(f"[INFO] Analyzing {pair}...")

            df = self.fetcher.fetch_forex_data(
                pair, 
                period='1mo', 
                interval=tf_config['interval']
            )

            if not df.empty:
                signal = self.generator.analyze_forex(df, pair, tf_config)
                if signal:
                    signals.append(signal)

        return signals

    def run_full_scan(self, timeframe='1H'):
        """Run complete scan across all asset classes"""
        print("\n" + "═" * 70)
        print("  AUTONOMOUS TRADING SIGNAL ENGINE v1.0")
        print("  Starting full market scan...")
        print("═" * 70 + "\n")

        crypto_signals = self.run_crypto_scan(timeframe)
        forex_signals = self.run_forex_scan(timeframe)

        report = self.reporter.generate_full_report(crypto_signals, forex_signals, timeframe)

        # Save report
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        filename = f'trading_signals_{timeframe}_{timestamp}.txt'
        with open(filename, 'w') as f:
            f.write(report)

        print(report)
        print(f"\n[INFO] Report saved to: {filename}")

        return report


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    engine = TradingSignalEngine()

    # Run scan for all timeframes
    for tf in ['1H', '4H', '1D']:
        engine.run_full_scan(tf)
        print("\n" + "═" * 70 + "\n")
