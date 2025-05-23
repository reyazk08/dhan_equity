####--- Not Required for this bot, used just for testing purposes ---####
# --- Imports & Environment Setup ---
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
from config import STRATEGY, RISK, MARKET, TELEGRAM, WATCHLIST_FILE
from dotenv import load_dotenv
load_dotenv()

ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")
CLIENT_ID = os.getenv("CLIENT_ID")

from dhanhq import dhanhq
dhan = dhanhq(CLIENT_ID, ACCESS_TOKEN)

# --- Ensure UTF-8 Output for Terminal (optional, safe for Windows) ---
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding='utf-8')
# --- Logging Setup ---
class FlushFileHandler(logging.FileHandler):
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

# =========================
# FORMATTING & LOGGING FUNCTIONS
# =========================
# Add this after load_dotenv()

def format_startup_log(balance=0):
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

def format_status_log(live_trades, uptime_sec, balance=0):
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

def format_entry_log(entry):
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

def format_exit_log(trade):
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

def log_indicator(symbol, ema_fast, ema_slow, rsi, ha_close, prev_ema_fast, prev_ema_slow, prev_rsi, prev_ha_close):
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

# --- Telegram Alert Formatting (with emojis) ---
def format_startup_alert(balance=0):
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

def format_status_alert(live_trades, uptime_sec, balance=0):
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

def format_entry_alert(entry):
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

def format_exit_alert(trade):
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

# =========================
# API & DATA FUNCTIONS
# =========================

def load_watchlist(file_path=WATCHLIST_FILE):
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

def fetch_historical_data(symbol, security_id, interval):
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "access-token": ACCESS_TOKEN,
        "dhan-client-id": CLIENT_ID
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

def get_dhan_balance():
    try:
        fund_limits = dhan.get_fund_limits()
        # Dhan returns: {'status': ..., 'remarks': ..., 'data': {...}}
        data = fund_limits.get('data', {})
        return float(data.get('availabelBalance', 0))
    except Exception as e:
        logging.error(f"Error fetching Dhan balance: {e}")
        return 0

def fetch_ltp(symbols_tokens):
    tokens = [int(item['token']) for item in symbols_tokens]
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "access-token": ACCESS_TOKEN,
        "client-id": CLIENT_ID
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
                    # The Dhan API's LTP response uses the security_id (token) as a string key.
                    # Ensure we use the string version of the token for lookup.
                    token_str = str(item['token']) 
                    price_data = ltp_data["data"]["NSE_EQ"].get(token_str)
                    
                    if price_data:
                        last_price_raw = price_data.get("last_price")
                        if last_price_raw is not None:
                            try:
                                ltp_value = float(last_price_raw)
                                result[symbol] = ltp_value
                            except (ValueError, TypeError) as e_conv:
                                logging.warning(f"[{symbol}] LTP value '{last_price_raw}' (type {type(last_price_raw)}) couldn't be converted to float: {e_conv}. Skipping LTP for this symbol.")
                                # If conversion fails, we don't add it to result.
                                # prices.get(symbol) will then return None in the main loop.
                        # else: last_price key was missing or its value was None. Symbol won't be in result for LTP.
                    # else: price_data for this token_str was not found in the response.
                return result
            else:
                logging.warning(f"LTP response missing 'data' or 'NSE_EQ' field: {ltp_data}")
                return {}
        else:
            logging.error(f"LTP Error: {response.status_code} - {response.text}")
            return {}
    except requests.exceptions.RequestException as e_req:
        logging.error(f"LTP request failed (RequestException): {e_req}")
        return {}
    except json.JSONDecodeError as e_json:
        logging.error(f"LTP response JSON decode error: {e_json} - Response text: {response.text if 'response' in locals() else 'Response object not available'}")
        return {}
    except Exception as e:
        logging.error(f"Unexpected exception in LTP request: {e}")
        return {}

# =========================
# INDICATOR FUNCTIONS
# =========================

def calculate_ema(prices, period):
    if len(prices) < period:
        return pd.Series([np.nan]*len(prices))
    series = pd.Series(prices)
    ema = series.ewm(span=period, adjust=False).mean()
    return ema

def calculate_rsi(prices, period=14):
    series = pd.Series(prices)
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def heikin_ashi(df):
    ha_df = df.copy()
    ha_df['HA_Close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    ha_open = [(df['open'][0] + df['close'][0]) / 2]
    for i in range(1, len(df)):
        ha_open.append((ha_open[i-1] + ha_df['HA_Close'][i-1]) / 2)
    ha_df['HA_Open'] = ha_open
    ha_df['HA_High'] = ha_df[['HA_Open', 'HA_Close', 'high']].max(axis=1)
    ha_df['HA_Low'] = ha_df[['HA_Open', 'HA_Close', 'low']].min(axis=1)
    return ha_df

# =========================
# TELEGRAM ALERT FUNCTION
# =========================

def send_telegram_alert(message):
    if not TELEGRAM['enabled']:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM['token']}/sendMessage"
    payload = {"chat_id": TELEGRAM['chat_id'], "text": message, "parse_mode": "Markdown"}
    try:
        requests.post(url, json=payload)
    except Exception as e:
        logging.error(f"Telegram alert failed: {e}")

# =========================
# TRADE MANAGEMENT FUNCTIONS
# =========================

def load_live_trades():
    try:
        with open("live_trades.json", "r") as f:
            return json.load(f)
    except:
        return []

def save_live_trades(trades):
    with open("live_trades.json", "w") as f:
        json.dump(trades, f, indent=2)

def log_trade(trade, log_type="entry"):
    log_file = "trade_log.json"
    try:
        if os.path.exists(log_file):
            with open(log_file, "r") as f:
                logs = json.load(f)
        else:
            logs = []
        logs.append(trade)
        if len(logs) > 1000:
            logs = logs[-500:]
        with open(log_file, "w") as f:
            json.dump(logs, f, indent=2)
        logging.info(f"Trade log updated: {trade['symbol']} {trade.get('event', log_type)}")
    except Exception as e:
        logging.error(f"Trade log error: {e}")

# =========================
# SIGNAL/STRATEGY LOGIC
# =========================

def analyze_entry_exit(df, ha_df, symbol, last_trade):
    close = df['close']
    ema_fast = calculate_ema(close, STRATEGY['ema_fast_period'])
    ema_slow = calculate_ema(close, STRATEGY['ema_slow_period'])
    rsi = calculate_rsi(close, STRATEGY['rsi_period'])
    ha_close = ha_df['HA_Close']

    signal = None
    # Entry signals
    if len(ema_fast) > 1 and len(ema_slow) > 1:
        # Long entry
        if (ema_fast.iloc[-2] < ema_slow.iloc[-2] and ema_fast.iloc[-1] > ema_slow.iloc[-1]
            and rsi.iloc[-1] < STRATEGY['rsi_overbought']):
            signal = "long"
        # Short entry
        elif (ema_fast.iloc[-2] > ema_slow.iloc[-2] and ema_fast.iloc[-1] < ema_slow.iloc[-1]
              and rsi.iloc[-1] > STRATEGY['rsi_oversold']):
            signal = "short"
    return signal, ema_fast.iloc[-1], ema_slow.iloc[-1], rsi.iloc[-1], ha_close.iloc[-1]

# =========================
# MAIN BOT LOOP WITH PROPER LIVE DATA HANDLING
# =========================

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

def main():
    start_time = time.time()
    live_trades = load_live_trades()
    funds = get_dhan_balance()
    
    # Separate intervals for different functions
    log_interval_sec = 60          # Log every 60 seconds
    alert_interval_sec = 15 * 60   # Telegram alerts every 15 minutes
    
    logging.info(format_startup_log(balance=funds))
    send_telegram_alert(format_startup_alert(balance=funds))
    
    watchlist = load_watchlist()
    if not watchlist:
        logging.error("No symbols found in watchlist. Exiting.")
        return
    
    logging.info(f"Fetched Dhan balance: {funds:.2f} INR")
    max_positions = RISK['max_positions']
    margin_multiplier = RISK['margin_multiplier']
    trade_size = funds / max_positions
    
    # Initialize timers
    last_log_time = time.time()
    last_alert_time = time.time()
    
    # --- ONE-TIME: Initialize historical data for all symbols ---
    symbol_data = initialize_symbol_data(watchlist, STRATEGY['trading_interval'])
    
    if not symbol_data:
        logging.error("No symbols have sufficient historical data. Exiting.")
        return
    
    logging.info("Starting live trading loop...")
    
    while True:
        try:
            # --- Fetch live prices for all symbols ---
            prices = fetch_ltp(watchlist)
            current_time = time.time()
            
            if not prices:
                logging.warning("No LTP data received, retrying...")
                time.sleep(10)
                continue
            
            # --- Process each symbol with live data ---
            for item in watchlist:
                symbol = item['symbol']
                token = item['token']
                
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
                
                # --- Entry Logic ---
                if not last_trade and signal in ["long", "short"]:
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
                
                # --- Exit Logic ---
                elif last_trade and last_trade['status'] == "open":
                    side = last_trade['side']
                    sl = last_trade.get('trailing_sl', last_trade['initial_sl'])
                    target = last_trade['target']
                    exit_reason = None
                    
                    # Add this safety check
                    if sl is None or target is None:
                        logging.error(f"[{symbol}] Stop loss or target is None for trade: {last_trade}")
                        continue  # Skip this iteration for this symbol
                    
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
                    
                    # Execute exit
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
                        
                        # Remove closed trade
                        live_trades = [t for t in live_trades if not (t['symbol'] == symbol and t['status'] == "closed")]
                        save_live_trades(live_trades)
                    
                    else:
                        # --- Trailing SL Logic ---
                        if not last_trade.get('trailing_sl'):
                            last_trade['trailing_sl'] = last_trade['initial_sl']
                        
                        move_threshold = 0.005  # 0.5%
                        if side == "long" and ha_close > last_trade['entry_price']:
                            new_trail = max(last_trade['trailing_sl'], 
                                          ha_close * (1 - RISK['initial_stop_loss']))
                            if new_trail > last_trade['trailing_sl'] + move_threshold * last_trade['entry_price']:
                                last_trade['trailing_sl'] = round(new_trail, 2)
                                log_trade({**last_trade, "event": "trail_update"})
                                logging.info(f"[{symbol}] Trailing SL updated to {last_trade['trailing_sl']}")
                                save_live_trades(live_trades)
                        
                        elif side == "short" and ha_close < last_trade['entry_price']:
                            new_trail = min(last_trade['trailing_sl'], 
                                          ha_close * (1 + RISK['initial_stop_loss']))
                            if new_trail < last_trade['trailing_sl'] - move_threshold * last_trade['entry_price']:
                                last_trade['trailing_sl'] = round(new_trail, 2)
                                log_trade({**last_trade, "event": "trail_update"})
                                logging.info(f"[{symbol}] Trailing SL updated to {last_trade['trailing_sl']}")
                                save_live_trades(live_trades)
            
            # --- Update log timer ---
            if current_time - last_log_time >= log_interval_sec:
                last_log_time = current_time
            
            # --- Periodic Telegram Status Alert (every 15 minutes) ---
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
            time.sleep(10)  # Short sleep before retrying

# =========================
# ENTRY POINT
# =========================

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Bot stopped by user")
        logging.info("Bot stopped by user")
    except Exception as e:
        print(f"Bot stopped due to error: {e}")
        logging.error(f"Bot stopped due to error: {e}")