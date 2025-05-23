# =============================================================================
# DHAN EQUITY TRADING BOT - EMA/RSI/HEIKIN-ASHI STRATEGY
# =============================================================================
import os
import json
import time
import requests
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from datetime import datetime, timedelta
import csv
import logging
import sys
import io
import random
import functools
from config import STRATEGY, RISK, MARKET, TELEGRAM, WATCHLIST_FILE

# =============================================================================
# INITIALIZATION
# =============================================================================
# Initialize global variables
cached_prices = {}  # Cache for LTP values

# Load environment variables
load_dotenv()
ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")
CLIENT_ID = os.getenv("CLIENT_ID")

# Initialize Dhan API client
from dhanhq import dhanhq
dhan = dhanhq(CLIENT_ID, ACCESS_TOKEN)

# Enable proper UTF-8 output for Windows terminals
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding='utf-8')

# =============================================================================
# LOGGING SETUP
# =============================================================================
class FlushFileHandler(logging.FileHandler):
    """Custom file handler that flushes after each log entry"""
    def emit(self, record):
        super().emit(record)
        self.flush()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    handlers=[
        FlushFileHandler('dhan_bot.log'),
        logging.StreamHandler()
    ]
)

# =============================================================================
# API ERROR MONITORING
# =============================================================================
class APIMonitor:
    """Tracks API errors and implements circuit breakers"""
    def __init__(self):
        self.error_counts = {}
        self.max_errors = 10
        self.error_window = 60 * 5  # 5 minutes window
        self.last_reset = time.time()
    
    def record_error(self, endpoint):
        current_time = time.time()
        
        # Reset counters if outside window
        if current_time - self.last_reset > self.error_window:
            self.error_counts = {}
            self.last_reset = current_time
        
        # Record error
        self.error_counts[endpoint] = self.error_counts.get(endpoint, 0) + 1
        
        # Return if we should pause this endpoint
        return self.error_counts.get(endpoint, 0) >= self.max_errors
    
    def should_pause_endpoint(self, endpoint):
        return self.error_counts.get(endpoint, 0) >= self.max_errors

# Initialize the API monitor
api_monitor = APIMonitor()

# =============================================================================
# INDICATOR CACHING FOR PERFORMANCE
# =============================================================================
class IndicatorCache:
    """Caches indicator calculations to improve performance"""
    def __init__(self):
        self.cache = {}
        self.timeframes = {}
    #-------------------------EMA Calculation------------------------
    def calculate_ema(self, symbol, prices, period, timestamp=None):
        """Calculate EMA with caching for efficiency"""
        current_time = timestamp or time.time()
        cache_key = f"{symbol}_ema_{period}"
        
        # If we have a cached value and it's recent enough
        if cache_key in self.cache and current_time - self.timeframes.get(cache_key, 0) < 60:
            # Just calculate the latest value based on the previous one
            if len(prices) > 0:
                last_price = prices.iloc[-1]
                last_ema = self.cache[cache_key]
                smoothing = 2 / (period + 1)
                new_ema = last_price * smoothing + last_ema * (1 - smoothing)
                self.cache[cache_key] = new_ema
                self.timeframes[cache_key] = current_time
                return new_ema
        
        # Otherwise calculate from scratch
        if len(prices) < period:
            return np.nan
        
        series = pd.Series(prices)
        ema = series.ewm(span=period, adjust=False).mean().iloc[-1]
        
        # Cache the result
        self.cache[cache_key] = ema
        self.timeframes[cache_key] = current_time
        return ema
    
    #-------------------------RSI Calculation-------------------------
    def calculate_rsi(self, symbol, prices, period=14, timestamp=None):
        """Calculate RSI with caching for efficiency"""
        current_time = timestamp or time.time()
        cache_key = f"{symbol}_rsi_{period}"
        
        # For RSI we need to recalculate more often as it's more sensitive
        # to recent price changes
        if cache_key in self.cache and current_time - self.timeframes.get(cache_key, 0) < 30:
            # For an approximate update, we'd need more complex logic
            # For now, just recalculate if the cache is outdated
            pass
        
        # Calculate from scratch
        series = pd.Series(prices)
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs)).iloc[-1]
        
        # Cache the result
        self.cache[cache_key] = rsi
        self.timeframes[cache_key] = current_time
        return rsi

# Initialize the indicator cache
indicator_cache = IndicatorCache()

# =============================================================================
# API REQUEST HANDLERS
# =============================================================================
def rate_limit_handler(func):
    """Decorator to handle rate limits with exponential backoff"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        max_retries = 5
        base_delay = 1  # Initial delay in seconds
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    # Check if it's a rate limit error (429)
                    if hasattr(e, 'response') and e.response is not None and e.response.status_code == 429:
                        # Exponential backoff: 1s, 2s, 4s, 8s, 16s
                        delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                        logging.warning(f"Rate limit hit. Retrying in {delay:.2f} seconds. Attempt {attempt+1}/{max_retries}")
                        time.sleep(delay)
                    else:
                        # For other exceptions, retry with shorter delay
                        delay = base_delay + random.uniform(0, 1)
                        logging.warning(f"Request failed. Retrying in {delay:.2f} seconds. Attempt {attempt+1}/{max_retries}")
                        time.sleep(delay)
                else:
                    raise
    return wrapper

#--------------------------Fetch Last Trading Price--------------------------
@rate_limit_handler
def fetch_ltp(symbols_tokens):
    """Fetch last traded price with improved error handling"""
    global cached_prices  # Move this to the beginning of the function
    
    if api_monitor.should_pause_endpoint("ltp"):
        logging.warning("Too many LTP API errors, using cached prices for now")
        return cached_prices  # Return last successful prices
        
    tokens = [int(item['token']) for item in symbols_tokens]
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "access-token": ACCESS_TOKEN,
        "client-id": CLIENT_ID  # Note: This is "client-id", not "dhan-client-id"
    }
    url = "https://api.dhan.co/v2/marketfeed/ltp"
    payload = {"NSE_EQ": tokens}
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        
        if response.status_code == 200:
            ltp_data = response.json()
            if "data" in ltp_data and "NSE_EQ" in ltp_data["data"]:
                result = {}
                for item in symbols_tokens:
                    symbol = item['symbol']
                    token_str = str(item['token'])
                    price_data = ltp_data["data"]["NSE_EQ"].get(token_str)
                    
                    if price_data:
                        last_price_raw = price_data.get("last_price")
                        if last_price_raw is not None:
                            try:
                                ltp_value = float(last_price_raw)
                                result[symbol] = ltp_value
                            except (ValueError, TypeError) as e_conv:
                                logging.warning(f"[{symbol}] LTP conversion error: {e_conv}")
                        
                # Cache successful results
                if result:
                    cached_prices = result.copy()
                return result
            
            # Rest of the function remains the same
            else:
                api_monitor.record_error("ltp")
                logging.warning("LTP response missing expected fields")
                return cached_prices
                
        elif response.status_code == 429:  # Rate limit
            api_monitor.record_error("ltp")
            logging.warning("LTP rate limit hit")
            raise requests.exceptions.RequestException("Rate limit exceeded")
            
        else:
            api_monitor.record_error("ltp")
            logging.error(f"LTP Error: {response.status_code} - {response.text}")
            return cached_prices
            
    except Exception as e:
        api_monitor.record_error("ltp")
        logging.error(f"Exception in LTP request: {e}")
        return cached_prices

#--------------------------Fetch Historical Data--------------------------
def fetch_historical_data(symbol, security_id, interval):
    """Fetch historical OHLC data from Dhan API"""
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "access-token": ACCESS_TOKEN,
        "dhan-client-id": CLIENT_ID  # Note: This is "dhan-client-id", not "client-id"
    }
    days = 30
    from_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    to_date = datetime.now().strftime("%Y-%m-%d")
    payload = {
        "securityId": security_id,
        "exchangeSegment": "NSE_EQ",
        "instrument": "EQUITY",
        "fromDate": from_date,
        "toDate": to_date,
        "interval": interval
    }
    try:
        response = requests.post(
            "https://api.dhan.co/v2/charts/historical",
            headers=headers,
            json=payload
        )
        if response.status_code == 200:
            data = response.json()
            if not all(k in data for k in ['open', 'high', 'low', 'close']):
                return None
            df = pd.DataFrame({
                'open': data['open'],
                'high': data['high'],
                'low': data['low'],
                'close': data['close'],
                'volume': data.get('volume', [0]*len(data['close']))
            })
            return df
        else:
            logging.error(f"{symbol}: Error {response.status_code} - {response.text}")
            return None
    except Exception as e:
        logging.error(f"Exception for {symbol}: {e}")
        return None

#--------------------------Get Dhan Account Balance--------------------------
def get_dhan_balance():
    """Get available balance from Dhan account"""
    try:
        fund_limits = dhan.get_fund_limits()
        # Dhan returns: {'status': ..., 'remarks': ..., 'data': {...}}
        data = fund_limits.get('data', {})
        return float(data.get('availabelBalance', 0))  # Note: API misspells 'availabelBalance'
    except Exception as e:
        logging.error(f"Error fetching Dhan balance: {e}")
        return 0

#--------------------------Batch API Requests--------------------------
def batch_api_requests(symbols_list, batch_size=5):
    """Process symbols in batches to avoid rate limits"""
    all_results = {}
    
    # Split the symbols into batches
    batches = [symbols_list[i:i + batch_size] for i in range(0, len(symbols_list), batch_size)]
    
    for batch_num, batch in enumerate(batches):
        logging.info(f"Processing batch {batch_num+1}/{len(batches)} with {len(batch)} symbols")
        
        # Process this batch
        batch_prices = fetch_ltp(batch)
        all_results.update(batch_prices)
        
        # Sleep between batches to avoid rate limits
        if batch_num < len(batches) - 1:
            time.sleep(0.5)
    
    return all_results

# =============================================================================
# DATA MANAGEMENT FUNCTIONS
# =============================================================================
#--------------------------Load Watchlist--------------------------
def load_watchlist(file_path=WATCHLIST_FILE):
    """Load trading symbols from CSV watchlist file"""
    watchlist = []
    try:
        if not os.path.exists(file_path):
            logging.error(f"Watchlist file not found: {file_path}")
            return []
        with open(file_path, "r") as csvfile:
            csvfile = (line for line in csvfile if not line.startswith('//'))
            reader = csv.DictReader(csvfile)
            for row in reader:
                if 'symbol' in row and 'token' in row:
                    watchlist.append({
                        'symbol': row['symbol'],
                        'token': row['token'],
                        'label': row.get('label', row['symbol'])
                    })
        logging.info(f"Loaded {len(watchlist)} symbols from watchlist")
        return watchlist
    except Exception as e:
        logging.error(f"Error loading watchlist: {e}")
        return []

#--------------------------Load Live Trades--------------------------
def load_live_trades():
    """Load active trades from JSON file"""
    try:
        with open("live_trades.json", "r") as f:
            return json.load(f)
    except:
        return []

#--------------------------Save Live Trades--------------------------
def save_live_trades(trades):
    """Save active trades to JSON file"""
    with open("live_trades.json", "w") as f:
        json.dump(trades, f, indent=2)

#--------------------------Log Trade History--------------------------
def log_trade(trade, log_type="entry"):
    """Log trade details to trade history file"""
    log_file = "trade_log.json"
    try:
        if os.path.exists(log_file):
            with open(log_file, "r") as f:
                logs = json.load(f)
        else:
            logs = []
        logs.append(trade)
        if len(logs) > 1000:
            logs = logs[-500:]  # Keep only last 500 trades
        with open(log_file, "w") as f:
            json.dump(logs, f, indent=2)
        logging.info(f"Trade log updated: {trade['symbol']} {trade.get('event', log_type)}")
    except Exception as e:
        logging.error(f"Trade log error: {e}")

#--------------------------Initialize Symbol Data--------------------------
def initialize_symbol_data(watchlist, interval):
    """Fetch historical data once at startup for indicator warm-up"""
    symbol_data = {}
    logging.info("Initializing historical data for all symbols...")
    
    for item in watchlist:
        symbol = item['symbol']
        token = item['token']
        logging.info(f"Fetching historical data for {symbol}...")
        
        df = fetch_historical_data(symbol, token, interval)
        if df is not None and len(df) >= STRATEGY['ema_slow_period']:
            symbol_data[symbol] = df
            logging.info(f"[{symbol}] Loaded {len(df)} historical candles")
        else:
            logging.warning(f"[{symbol}] Not enough historical data, skipping.")
        
        # Add small delay to avoid rate limits
        time.sleep(0.5)
    
    logging.info(f"Historical data initialized for {len(symbol_data)} symbols")
    return symbol_data

#--------------------------Update Live Candle--------------------------
def update_live_candle(df, ltp, prev_close=None):
    """Create a new live candle and append it to the DataFrame"""
    if prev_close is None:
        prev_close = df['close'].iloc[-1] if len(df) > 0 else ltp
    
    # Create new live candle
    new_candle = pd.DataFrame({
        'open': [prev_close],  # Use previous close as new open
        'high': [max(prev_close, ltp)],
        'low': [min(prev_close, ltp)],
        'close': [ltp],
        'volume': [0]  # Volume not available for live data
    })
    
    # Append new candle and keep only last 100 candles
    df = pd.concat([df, new_candle], ignore_index=True)
    if len(df) > 100:
        df = df.iloc[-100:].reset_index(drop=True)
    
    return df

# =============================================================================
# TECHNICAL ANALYSIS FUNCTIONS
# =============================================================================
#--------------------------EMA--------------------------
def calculate_ema(prices, period):
    """Calculate Exponential Moving Average"""
    if len(prices) < period:
        return pd.Series([np.nan]*len(prices))
    series = pd.Series(prices)
    ema = series.ewm(span=period, adjust=False).mean()
    return ema

#--------------------------RSI--------------------------
def calculate_rsi(prices, period=14):
    """Calculate Relative Strength Index"""
    series = pd.Series(prices)
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

#--------------------------Heikin-Ashi--------------------------
def heikin_ashi(df):
    """Calculate Heikin-Ashi candles"""
    ha_df = df.copy()
    ha_df['HA_Close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    ha_open = [(df['open'][0] + df['close'][0]) / 2]
    for i in range(1, len(df)):
        ha_open.append((ha_open[i-1] + ha_df['HA_Close'][i-1]) / 2)
    ha_df['HA_Open'] = ha_open
    ha_df['HA_High'] = ha_df[['HA_Open', 'HA_Close', 'high']].max(axis=1)
    ha_df['HA_Low'] = ha_df[['HA_Open', 'HA_Close', 'low']].min(axis=1)
    return ha_df

#--------------------------Analyze Entry/Exit Signals--------------------------
def analyze_entry_exit(df, ha_df, symbol, last_trade):
    """Analyze price data for entry/exit signals"""
    close = df['close']
    ema_fast = calculate_ema(close, STRATEGY['ema_fast_period'])
    ema_slow = calculate_ema(close, STRATEGY['ema_slow_period'])
    rsi = calculate_rsi(close, STRATEGY['rsi_period'])
    ha_close = ha_df['HA_Close']

    signal = None
    # Entry signals
    if len(ema_fast) > 1 and len(ema_slow) > 1:
        # Long entry: Fast EMA crosses above Slow EMA + RSI not overbought
        if (ema_fast.iloc[-2] < ema_slow.iloc[-2] and ema_fast.iloc[-1] > ema_slow.iloc[-1]
            and rsi.iloc[-1] < STRATEGY['rsi_overbought']):
            signal = "long"
        # Short entry: Fast EMA crosses below Slow EMA + RSI not oversold
        elif (ema_fast.iloc[-2] > ema_slow.iloc[-2] and ema_fast.iloc[-1] < ema_slow.iloc[-1]
              and rsi.iloc[-1] > STRATEGY['rsi_oversold']):
            signal = "short"
    return signal, ema_fast.iloc[-1], ema_slow.iloc[-1], rsi.iloc[-1], ha_close.iloc[-1]

#--------------------------Log Indicator Values--------------------------
def log_indicator(symbol, ema_fast, ema_slow, rsi, ha_close, prev_ema_fast, prev_ema_slow, prev_rsi, prev_ha_close):
    """Log indicator values for analysis"""
    log_entry = {
        "time": datetime.now().isoformat(),
        "symbol": symbol,
        "ema_fast": ema_fast,
        "ema_slow": ema_slow,
        "rsi": rsi,
        "ha_close": ha_close,
        "prev_ema_fast": prev_ema_fast,
        "prev_ema_slow": prev_ema_slow,
        "prev_rsi": prev_rsi,
        "prev_ha_close": prev_ha_close
    }
    logging.info(
        f"[{symbol}] EMA Fast: {ema_fast:.2f} (prev {prev_ema_fast:.2f}), "
        f"EMA Slow: {ema_slow:.2f} (prev {prev_ema_slow:.2f}), "
        f"RSI: {rsi:.2f} (prev {prev_rsi:.2f}), "
        f"HA Close: {ha_close:.2f} (prev {prev_ha_close:.2f})"
    )
    with open("indicator_log.jsonl", "a") as f:
        f.write(json.dumps(log_entry) + "\n")

# =============================================================================
# MESSAGING & NOTIFICATION FUNCTIONS
# =============================================================================

#--------------------------Send Telegram Alert--------------------------
def send_telegram_alert(message):
    """Send alert message to Telegram channel"""
    if not TELEGRAM['enabled']:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM['token']}/sendMessage"
    payload = {"chat_id": TELEGRAM['chat_id'], "text": message, "parse_mode": "Markdown"}
    try:
        requests.post(url, json=payload)
    except Exception as e:
        logging.error(f"Telegram alert failed: {e}")

#--------------------------Format Log Messages--------------------------
def format_startup_log(balance=0):
    """Format startup log message"""
    return (
        "[Dhan Bot] Equity Trading Bot Started\n"
        "-------------------------------------\n"
        f"Strategy: EMA/RSI/HA\n"
        f"Trading Timeframe: {STRATEGY['trading_interval']}m\n"
        f"Leverage: {RISK['margin_multiplier']}x\n"
        f"Risk per Trade: {round(100*RISK['max_position_value']/100000, 1)}%\n"
        f"Initial SL: {RISK['initial_stop_loss']*100:.1f}%\n"
        f"Take-Profit: {RISK['take_profit_pct']*100:.1f}%\n"
        f"Trailing: Stepped ({RISK['trail_step']*100:.1f}%)\n"
        f"Max Positions: {RISK['max_positions']}\n"
        f"Available Funds: {balance:.2f} INR\n"
        "-------------------------------------"
    )

#--------------------------Format Status Log--------------------------
def format_status_log(live_trades, uptime_sec, balance=0):
    """Format status log message"""
    uptime = f"{int(uptime_sec//3600):02d}h {int((uptime_sec%3600)//60):02d}m"
    msg = (
        "[Dhan Bot] Bot Status\n"
        "-------------------------------------\n"
        f"Strategy: EMA/RSI/HA\n"
        f"Entry TF: {STRATEGY['trading_interval']}m\n"
        f"Uptime: {uptime}\n"
        f"Balance: {balance:.2f} INR\n"
        f"Active Trades:\n"
    )
    if live_trades:
        for t in live_trades:
            msg += f"  - {t['symbol']} | Side: {t['side']} | Entry: {t['entry_price']} | Current: {t.get('current_price', '-') } | SL: {t.get('trailing_sl', t['initial_sl'])}\n"
    else:
        msg += "  - None\n"
    msg += "-------------------------------------"
    return msg

#--------------------------Format Entry/Exit Log--------------------------
def format_entry_log(entry):
    """Format trade entry log message"""
    return (
        "[Dhan Bot] NEW TRADE ENTRY\n"
        "-------------------------------------\n"
        f"Symbol: {entry['symbol']}\n"
        f"Side: {entry['side'].capitalize()}\n"
        f"Entry Price: {entry['entry_price']}\n"
        f"Quantity: {entry['qty']}\n"
        f"Position Size: {entry['qty']*entry['entry_price']:.2f} INR\n"
        f"Leverage: {RISK['margin_multiplier']}x\n"
        f"Stop-Loss: {entry['initial_sl']}\n"
        f"Take-Profit: {entry['target']}\n"
        f"Entry Timeframe: {STRATEGY['trading_interval']}m\n"
        f"Time: {entry['entry_time']}\n"
        "-------------------------------------"
    )

#--------------------------Format Exit Log--------------------------
def format_exit_log(trade):
    """Format trade exit log message"""
    duration = (
        datetime.fromisoformat(trade['exit_time']) - datetime.fromisoformat(trade['entry_time'])
    )
    return (
        f"[Dhan Bot] {'Stop-Loss Hit' if trade['exit_reason']=='stop_loss' else 'Target Hit'}\n"
        "-------------------------------------\n"
        f"Symbol: {trade['symbol']}\n"
        f"Side: {trade['side'].capitalize()}\n"
        f"Entry Price: {trade['entry_price']}\n"
        f"Exit Price: {trade['exit_price']}\n"
        f"Profit/Loss: {trade['pnl']:.2f} INR ({(trade['pnl']/trade['entry_price']*100):.2f}%)\n"
        f"Trade Duration: {duration}\n"
        "-------------------------------------"
    )

#--------------------------Format Telegram Alerts--------------------------
def format_startup_alert(balance=0):
    """Format Telegram startup alert"""
    return (
        "📈 [Dhan Bot] 🚀 •Equity Trading Bot Started•\n"
        "━━━━━━━━━━━━━━\n"
        f"• Strategy: 'EMA/RSI/HA'\n"
        f"• Trading Timeframe: '{STRATEGY['trading_interval']}m'\n"
        f"• Leverage: '{RISK['margin_multiplier']}x'\n"
        f"• Risk per Trade: '{round(100*RISK['max_position_value']/100000, 1)}%'\n"
        f"• Initial SL: '{RISK['initial_stop_loss']*100:.1f}%'\n"
        f"• Take-Profit: '{RISK['take_profit_pct']*100:.1f}%'\n"
        f"• Trailing: 'Stepped ({RISK['trail_step']*100:.1f}%)'\n"
        f"• Max Positions: '{RISK['max_positions']}'\n"
        f"• Available Funds: '{balance:.2f} INR'\n"
        "━━━━━━━━━━━━━━"
    )

#--------------------------Format Status Alert--------------------------
def format_status_alert(live_trades, uptime_sec, balance=0):
    """Format Telegram status alert"""
    uptime = f"{int(uptime_sec//3600):02d}h {int((uptime_sec%3600)//60):02d}m"
    msg = (
        "📈 [Dhan Bot] 📊 •Bot Status•\n"
        "━━━━━━━━━━━━━━\n"
        f"• Strategy: 'EMA/RSI/HA'\n"
        f"• Entry TF: '{STRATEGY['trading_interval']}m'\n"
        f"• Uptime: '{uptime}'\n"
        f"• Balance: '{balance:.2f} INR'\n"
        f"• Active Trades:\n"
    )
    if live_trades:
        for t in live_trades:
            msg += f"  • {t['symbol']} | Side: '{t['side']}' | Entry: '{t['entry_price']}' | Current: '{t.get('current_price', '-')}' | SL: '{t.get('trailing_sl', t['initial_sl'])}'\n"
    else:
        msg += "  • None\n"
    msg += "━━━━━━━━━━━━━━"
    return msg

#--------------------------Format Entry Alert--------------------------
def format_entry_alert(entry):
    """Format Telegram entry alert"""
    return (
        "📈 [Dhan Bot] 🚀 •NEW TRADE ENTRY• 🚀\n"
        "━━━━━━━━━━━━━━\n"
        f"• Symbol: '{entry['symbol']}'\n"
        f"• Side: '{entry['side'].capitalize()}'\n"
        f"• Entry Price: '{entry['entry_price']}'\n"
        f"• Quantity: '{entry['qty']}'\n"
        f"• Position Size: '{entry['qty']*entry['entry_price']:.2f} INR'\n"
        f"• Leverage: '{RISK['margin_multiplier']}x'\n"
        f"• Stop-Loss: '{entry['initial_sl']}'\n"
        f"• Take-Profit: '{entry['target']}'\n"
        f"• Entry Timeframe: '{STRATEGY['trading_interval']}m'\n"
        f"• Time: '{entry['entry_time']}'\n"
        "━━━━━━━━━━━━━━"
    )

#--------------------------Format Exit Alert--------------------------
def format_exit_alert(trade):
    """Format Telegram exit alert"""
    duration = (
        datetime.fromisoformat(trade['exit_time']) - datetime.fromisoformat(trade['entry_time'])
    )
    return (
        f"📈 [Dhan Bot] ⚠️ •{'Stop-Loss Hit' if trade['exit_reason']=='stop_loss' else 'Target Hit'}• ⚠️\n"
        "━━━━━━━━━━━━━━\n"
        f"• 🏷 Symbol: '{trade['symbol']}'\n"
        f"• 📊 Side: '{trade['side'].capitalize()}'\n"
        f"• ⬆️ Entry Price: '{trade['entry_price']}'\n"
        f"• ⬇️ Exit Price: '{trade['exit_price']}'\n"
        f"• 💰 Profit/Loss: '{trade['pnl']:.2f} INR ({(trade['pnl']/trade['entry_price']*100):.2f}%)'\n"
        f"• 🕒 Trade Duration: '{duration}'\n"
        "━━━━━━━━━━━━━━"
    )

# =============================================================================
# MAIN BOT LOGIC
# =============================================================================
def main():
    """Main bot execution loop"""
    # Initialize bot
    start_time = time.time()
    live_trades = load_live_trades()
    funds = get_dhan_balance()
    global cached_prices
    cached_prices = {}
    
    # Set separate intervals for different functions
    log_interval_sec = 60          # Log every 60 seconds
    alert_interval_sec = 15 * 60   # Telegram alerts every 15 minutes
    
    # Log startup information
    logging.info(format_startup_log(balance=funds))
    send_telegram_alert(format_startup_alert(balance=funds))
    
    # Load watchlist and verify
    watchlist = load_watchlist()
    if not watchlist:
        logging.error("No symbols found in watchlist. Exiting.")
        return
    
    # Initialize trading parameters
    logging.info(f"Fetched Dhan balance: {funds:.2f} INR")
    max_positions = RISK['max_positions']
    margin_multiplier = RISK['margin_multiplier']
    trade_size = funds / max_positions
    
    # Initialize timing variables
    last_log_time = time.time()
    last_alert_time = time.time()
    
    # Load historical data for all symbols
    symbol_data = initialize_symbol_data(watchlist, STRATEGY['trading_interval'])
    
    if not symbol_data:
        logging.error("No symbols have sufficient historical data. Exiting.")
        return
    
    logging.info("Starting live trading loop...")
    
    # Main trading loop
    while True:
        try:
            # --- SECTION 1: Fetch prices in batches to avoid rate limits ---
            all_symbols_chunks = [watchlist[i:i+5] for i in range(0, len(watchlist), 5)]
            prices = {}
            
            for chunk in all_symbols_chunks:
                chunk_prices = fetch_ltp(chunk)
                prices.update(chunk_prices)
                time.sleep(0.5)  # Small delay between batches
            
            current_time = time.time()
            
            # --- SECTION 2: Process each symbol ---
            for item in watchlist:
                symbol = item['symbol']
                token = item['token']
                
                try:
                    # Skip if we don't have historical data for this symbol
                    if symbol not in symbol_data:
                        continue
                    
                    df = symbol_data[symbol]
                    ltp = prices.get(symbol)
                    
                    if ltp is None:
                        logging.warning(f"[{symbol}] No LTP data, skipping this cycle")
                        continue
                    
                    # --- Update DataFrame with new live candle ---
                    prev_close = df['close'].iloc[-1] if len(df) > 0 else ltp
                    df = update_live_candle(df, ltp, prev_close)
                    symbol_data[symbol] = df  # Update the stored DataFrame
                    
                    # --- Calculate indicators on updated data ---
                    if len(df) < STRATEGY['ema_slow_period']:
                        logging.warning(f"[{symbol}] Not enough data for indicators, skipping")
                        continue
                    
                    ha_df = heikin_ashi(df)
                    last_trade = next((t for t in live_trades if t['symbol'] == symbol), None)
                    
                    # Get current and previous indicator values
                    signal, fast_ema, slow_ema, rsi, ha_close = analyze_entry_exit(df, ha_df, symbol, last_trade)
                    
                    if len(df) >= 2:  # Need at least 2 rows for previous values
                        prev_fast_ema = calculate_ema(df['close'], STRATEGY['ema_fast_period']).iloc[-2]
                        prev_slow_ema = calculate_ema(df['close'], STRATEGY['ema_slow_period']).iloc[-2]
                        prev_rsi = calculate_rsi(df['close'], STRATEGY['rsi_period']).iloc[-2]
                        prev_ha_close = ha_df['HA_Close'].iloc[-2]
                    else:
                        # For first iteration, use same values
                        prev_fast_ema = fast_ema
                        prev_slow_ema = slow_ema
                        prev_rsi = rsi
                        prev_ha_close = ha_close
                    
                    # --- Log indicators (every 60 seconds) ---
                    if current_time - last_log_time >= log_interval_sec:
                        log_indicator(symbol, fast_ema, slow_ema, rsi, ha_close, 
                                    prev_fast_ema, prev_slow_ema, prev_rsi, prev_ha_close)
                        
                        logging.info(
                            f"[{symbol}] Price: {ltp:.2f} | Signal: {signal} | "
                            f"Last Trade: {last_trade['side'] if last_trade else 'None'}"
                        )
                    
                    # --- SECTION 3: Entry Logic ---
                    if not last_trade and signal in ["long", "short"] and len(live_trades) < max_positions:
                        qty = int((trade_size * margin_multiplier) // ltp)
                        if qty > 0:  # Only enter if we can afford at least 1 share
                            entry = {
                                "symbol": symbol,
                                "side": signal,
                                "entry_price": ltp,
                                "entry_time": datetime.now().isoformat(),
                                "qty": qty,
                                "initial_sl": round(ltp * (1 - RISK['initial_stop_loss']) if signal == "long"
                                                  else ltp * (1 + RISK['initial_stop_loss']), 2),
                                "target": round(ltp * (1 + RISK['take_profit_pct']) if signal == "long"
                                              else ltp * (1 - RISK['take_profit_pct']), 2),
                                "trailing_sl": None,
                                "status": "open"
                            }
                            live_trades.append(entry)
                            save_live_trades(live_trades)
                            log_trade({**entry, "event": "entry"})
                            logging.info(format_entry_log(entry))
                            send_telegram_alert(format_entry_alert(entry))
                    
                    # --- SECTION 4: Exit Logic ---
                    elif last_trade and last_trade['status'] == "open":
                        side = last_trade['side']
                        sl = last_trade.get('trailing_sl', last_trade['initial_sl'])
                        target = last_trade['target']
                        exit_reason = None
                        
                        # Add safety check for None values
                        if sl is None or target is None:
                            logging.error(f"[{symbol}] Stop loss or target is None for trade: {last_trade}")
                            # Fix the problem by setting defaults
                            if sl is None:
                                sl = last_trade['initial_sl']
                                last_trade['trailing_sl'] = sl
                            if target is None:
                                target = round(last_trade['entry_price'] * (1 + RISK['take_profit_pct']) if side == "long"
                                                else last_trade['entry_price'] * (1 - RISK['take_profit_pct']), 2)
                                last_trade['target'] = target
                            save_live_trades(live_trades)
                            logging.info(f"[{symbol}] Fixed None values in trade: SL={sl}, Target={target}")
                        
                        # Check exit conditions
                        if side == "long":
                            if ltp <= sl:
                                exit_reason = "stop_loss"
                            elif ltp >= target:
                                exit_reason = "target"
                        elif side == "short":
                            if ltp >= sl:
                                exit_reason = "stop_loss"
                            elif ltp <= target:
                                exit_reason = "target"
                        
                        # Execute exit if conditions met
                        if exit_reason:
                            last_trade['exit_price'] = ltp
                            last_trade['exit_time'] = datetime.now().isoformat()
                            last_trade['exit_reason'] = exit_reason
                            last_trade['status'] = "closed"
                            pnl = ((ltp - last_trade['entry_price']) * last_trade['qty'] 
                                  if side == "long" else 
                                  (last_trade['entry_price'] - ltp) * last_trade['qty'])
                            last_trade['pnl'] = round(pnl, 2)
                            
                            log_trade({**last_trade, "event": "exit"})
                            logging.info(format_exit_log(last_trade))
                            send_telegram_alert(format_exit_alert(last_trade))
                            
                            # Remove closed trade from active trades
                            live_trades = [t for t in live_trades if not (t['symbol'] == symbol and t['status'] == "closed")]
                            save_live_trades(live_trades)
                        
                        # --- SECTION 5: Trailing Stop-Loss Logic ---
                        else:
                            # Initialize trailing stop if needed
                            if not last_trade.get('trailing_sl'):
                                last_trade['trailing_sl'] = last_trade['initial_sl']
                            
                            move_threshold = 0.005  # 0.5%
                            # For long positions, move stop loss up as price increases
                            if side == "long" and ha_close > last_trade['entry_price']:
                                new_trail = max(last_trade['trailing_sl'], 
                                              ha_close * (1 - RISK['initial_stop_loss']))
                                # Only update if significant movement
                                if new_trail > last_trade['trailing_sl'] + move_threshold * last_trade['entry_price']:
                                    last_trade['trailing_sl'] = round(new_trail, 2)
                                    log_trade({**last_trade, "event": "trail_update"})
                                    logging.info(f"[{symbol}] Trailing SL updated to {last_trade['trailing_sl']}")
                                    save_live_trades(live_trades)
                            
                            # For short positions, move stop loss down as price decreases
                            elif side == "short" and ha_close < last_trade['entry_price']:
                                new_trail = min(last_trade['trailing_sl'], 
                                              ha_close * (1 + RISK['initial_stop_loss']))
                                # Only update if significant movement
                                if new_trail < last_trade['trailing_sl'] - move_threshold * last_trade['entry_price']:
                                    last_trade['trailing_sl'] = round(new_trail, 2)
                                    log_trade({**last_trade, "event": "trail_update"})
                                    logging.info(f"[{symbol}] Trailing SL updated to {last_trade['trailing_sl']}")
                                    save_live_trades(live_trades)
                
                except Exception as symbol_error:
                    logging.error(f"Error processing symbol {symbol}: {symbol_error}")
                    continue  # Skip this symbol but continue with others
            
            # --- SECTION 6: Update log timer ---
            if current_time - last_log_time >= log_interval_sec:
                last_log_time = current_time
            
            # --- SECTION 7: Periodic Telegram Status Alert (every 15 minutes) ---
            if current_time - last_alert_time >= alert_interval_sec:
                funds = get_dhan_balance()
                status_msg = format_status_alert(live_trades, current_time - start_time, balance=funds)
                logging.info(format_status_log(live_trades, current_time - start_time, balance=funds))
                send_telegram_alert(status_msg)
                last_alert_time = current_time
            
            # Sleep for 60 seconds before next cycle
            time.sleep(60)
            
        except Exception as e:
            logging.error(f"Error in main loop: {e}")
            # Use different sleep times based on error type
            if "429" in str(e) or "rate limit" in str(e).lower():
                delay = 60  # Longer pause for rate limits
                logging.warning(f"Rate limit detected. Pausing for {delay} seconds")
            else:
                delay = 10
            time.sleep(delay)

# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Bot stopped by user")
        logging.info("Bot stopped by user")
    except Exception as e:
        print(f"Bot stopped due to error: {e}")
        logging.error(f"Bot stopped due to error: {e}")