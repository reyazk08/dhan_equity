import os
import requests
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import datetime
import csv
import time
from tabulate import tabulate  # pip install tabulate if not installed

# Load environment variables
load_dotenv()
access_token = os.getenv("ACCESS_TOKEN")
client_id = os.getenv("CLIENT_ID")

# Function to load watchlist
def load_watchlist(file_path="watchlist.csv"):
    """Load symbols and tokens from watchlist CSV file"""
    watchlist = []
    
    try:
        print(f"Loading watchlist from {file_path}...")
        if not os.path.exists(file_path):
            print(f"❌ Watchlist file not found: {file_path}")
            return []
            
        with open(file_path, "r") as csvfile:
            # Skip comment lines that start with //
            csvfile = (line for line in csvfile if not line.startswith('//'))
            reader = csv.DictReader(csvfile)
            for row in reader:
                if 'symbol' in row and 'token' in row:
                    watchlist.append({
                        'symbol': row['symbol'],
                        'token': row['token'],
                        'label': row.get('label', row['symbol'])
                    })
        
        print(f"✅ Loaded {len(watchlist)} symbols from watchlist")
        return watchlist
    except Exception as e:
        print(f"❌ Error loading watchlist: {e}")
        import traceback
        traceback.print_exc()
        return []

def fetch_historical_data(symbol, security_id):
    """Fetch historical data using the approach from hist_data.py"""
    print(f"Fetching historical data for {symbol} (ID: {security_id})...")
    
    # Set up headers with dhan-client-id for historical data
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "access-token": access_token,
        "dhan-client-id": client_id  # Note the "dhan-" prefix
    }
    
    # Define payload
    payload = {
        "securityId": security_id,
        "exchangeSegment": "NSE_EQ",
        "instrument": "EQUITY",
        "fromDate": "2025-04-20",  # Extended date range to get more data
        "toDate": "2025-05-22"
    }
    
    # Make API request for historical data
    try:
        response = requests.post(
            "https://api.dhan.co/v2/charts/historical", 
            headers=headers, 
            json=payload
        )
        
        # Check response status
        if response.status_code == 200:
            data = response.json()
            print(f"✅ {symbol}: Got historical data with {len(data['close']) if 'close' in data else 0} candles")
            return data
        else:
            print(f"❌ {symbol}: Error {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Exception for {symbol}: {e}")
        return None

def fetch_ltp(symbols_tokens):
    """Fetch last traded price for multiple symbols at once"""
    print(f"Fetching LTP for {len(symbols_tokens)} symbols...")
    
    # Extract all tokens
    tokens = [int(item['token']) for item in symbols_tokens]
    
    # Set up headers
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "access-token": access_token,
        "client-id": client_id
    }
    
    # API endpoint for Market LTP feed
    url = "https://api.dhan.co/v2/marketfeed/ltp"
    
    # Build the payload for multiple tokens
    payload = {
        "NSE_EQ": tokens
    }
    
    # Make the API request
    try:
        response = requests.post(url, headers=headers, json=payload)
        
        # Check response status
        if response.status_code == 200:
            ltp_data = response.json()
            
            # Check if we received data successfully
            if "data" in ltp_data and "NSE_EQ" in ltp_data["data"]:
                result = {}
                
                # Process each token
                for item in symbols_tokens:
                    symbol = item['symbol']
                    token = item['token']
                    
                    price_data = ltp_data["data"]["NSE_EQ"].get(token)
                    if price_data:
                        last_price = price_data.get("last_price")
                        change_pct = price_data.get("change_percentage")
                        result[symbol] = {
                            'price': last_price,
                            'change_pct': change_pct
                        }
                
                print(f"✅ Successfully fetched prices for {len(result)} symbols")
                return result
            else:
                print("❌ Did not receive expected data structure.")
                return {}
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
            return {}
            
    except Exception as e:
        print(f"❌ Exception in LTP request: {e}")
        return {}

def calculate_ema(prices, period):
    """Calculate EMA using pandas"""
    if len(prices) < period:
        return None
        
    series = pd.Series(prices)
    ema = series.ewm(span=period, adjust=False).mean()
    return ema

def analyze_symbol(hist_data, current_price):
    """Analyze a single symbol using historical data and current price"""
    if not hist_data or 'close' not in hist_data or not hist_data['close']:
        return None
        
    close_prices = hist_data['close']
    
    # Calculate indicators
    fast_period = 9
    slow_period = 21
    
    # Check if we have enough data points
    if len(close_prices) < slow_period:
        return {
            'status': 'insufficient_data',
            'message': f"Not enough data points (need {slow_period}, got {len(close_prices)})"
        }
    
    # Calculate EMAs
    fast_ema = calculate_ema(close_prices, fast_period)
    slow_ema = calculate_ema(close_prices, slow_period)
    
    if fast_ema is None or slow_ema is None:
        return None
    
    # Get latest values
    fast_ema_value = fast_ema.iloc[-1]
    slow_ema_value = slow_ema.iloc[-1]
    
    # Check for crossover (need at least 2 data points)
    crossover = 'none'
    if len(fast_ema) > 1 and len(slow_ema) > 1:
        if fast_ema.iloc[-2] < slow_ema.iloc[-2] and fast_ema.iloc[-1] > slow_ema.iloc[-1]:
            crossover = 'bullish'
        elif fast_ema.iloc[-2] > slow_ema.iloc[-2] and fast_ema.iloc[-1] < slow_ema.iloc[-1]:
            crossover = 'bearish'
    
    # Determine trend based on EMA relationship
    if fast_ema_value > slow_ema_value:
        trend = 'bullish'
    else:
        trend = 'bearish'
    
    # Determine price position relative to EMAs
    if current_price > fast_ema_value and current_price > slow_ema_value:
        price_position = 'above_both'
    elif current_price < fast_ema_value and current_price < slow_ema_value:
        price_position = 'below_both'
    else:
        price_position = 'between'
    
    # Determine overall signal
    signal = 'neutral'
    
    # Strong bullish: price above both EMAs and fast EMA > slow EMA
    if trend == 'bullish' and price_position == 'above_both':
        signal = 'strong_buy'
    # Bullish pullback: price between EMAs but fast EMA > slow EMA
    elif trend == 'bullish' and price_position == 'between':
        signal = 'buy'
    # Bearish with resistance: price below EMAs but fast EMA > slow EMA
    elif trend == 'bullish' and price_position == 'below_both':
        signal = 'neutral'
    # Weak bearish: price above EMAs but fast EMA < slow EMA
    elif trend == 'bearish' and price_position == 'above_both':
        signal = 'neutral'
    # Bearish pullback: price between EMAs and fast EMA < slow EMA
    elif trend == 'bearish' and price_position == 'between':
        signal = 'sell'
    # Strong bearish: price below both EMAs and fast EMA < slow EMA
    elif trend == 'bearish' and price_position == 'below_both':
        signal = 'strong_sell'
    
    # Fresh signal from crossover
    if crossover == 'bullish':
        signal = 'fresh_buy'
    elif crossover == 'bearish':
        signal = 'fresh_sell'
    
    return {
        'status': 'ok',
        'fast_ema': fast_ema_value,
        'slow_ema': slow_ema_value,
        'trend': trend,
        'price_position': price_position,
        'crossover': crossover,
        'signal': signal
    }

def get_signal_emoji(signal):
    """Convert signal to emoji for better visualization"""
    if signal == 'strong_buy':
        return '🔥 STRONG BUY'
    elif signal == 'buy':
        return '✅ BUY'
    elif signal == 'fresh_buy':
        return '🚀 FRESH BUY'
    elif signal == 'neutral':
        return '⚪ NEUTRAL'
    elif signal == 'sell':
        return '❌ SELL'
    elif signal == 'strong_sell':
        return '💀 STRONG SELL'
    elif signal == 'fresh_sell':
        return '🔻 FRESH SELL'
    else:
        return '❓ UNKNOWN'

def main():
    """Main function to analyze all watchlist symbols"""
    print("=" * 70)
    print("🧪 WATCHLIST ANALYZER - TECHNICAL INDICATORS")
    print("=" * 70)
    
    # Step 1: Load watchlist
    watchlist = load_watchlist()
    if not watchlist:
        print("❌ No symbols found in watchlist. Exiting.")
        return
    
    # Initialize results table
    results = []
    
    # Step 2: Fetch all historical data first
    historical_data = {}
    for item in watchlist:
        symbol = item['symbol']
        token = item['token']
        
        hist_data = fetch_historical_data(symbol, token)
        if hist_data and 'close' in hist_data and hist_data['close']:
            historical_data[symbol] = hist_data
            candle_count = len(hist_data['close'])
            
            # Add a small delay to avoid API rate limits
            time.sleep(0.5)
        
    # Step 3: Fetch all current prices at once
    prices = fetch_ltp(watchlist)
    
    # Step 4: Process each symbol
    for item in watchlist:
        symbol = item['symbol']
        token = item['token']
        display_name = item.get('label', symbol)
        
        print(f"\nProcessing {display_name} ({symbol})...")
        
        # Skip if no historical data
        if symbol not in historical_data:
            print(f"⚠️ No historical data for {symbol}, skipping analysis")
            results.append([
                display_name,
                "N/A",
                "N/A",
                "N/A",
                "N/A",
                "❌ NO DATA"
            ])
            continue
        
        # Get current price or use last close
        current_price = None
        price_source = None
        
        if symbol in prices and 'price' in prices[symbol]:
            current_price = prices[symbol]['price']
            change_pct = prices[symbol].get('change_pct', 'N/A')
            price_source = "LTP API"
        elif 'close' in historical_data[symbol] and historical_data[symbol]['close']:
            current_price = historical_data[symbol]['close'][-1]
            change_pct = "N/A"
            price_source = "Historical"
        
        if not current_price:
            print(f"⚠️ No price data for {symbol}, skipping analysis")
            results.append([
                display_name,
                "N/A",
                "N/A",
                "N/A",
                "N/A",
                "❌ NO PRICE"
            ])
            continue
        
        # Analyze the symbol
        analysis = analyze_symbol(historical_data[symbol], current_price)
        
        if not analysis or analysis.get('status') != 'ok':
            message = analysis.get('message', 'Analysis failed') if analysis else 'Analysis failed'
            print(f"⚠️ {message}")
            results.append([
                display_name,
                f"₹{current_price:.2f}",
                "N/A",
                "N/A",
                "N/A",
                "⚠️ " + message
            ])
            continue
        
        # Get signal emoji
        signal_text = get_signal_emoji(analysis['signal'])
        
        # Add to results
        results.append([
            display_name,
            f"₹{current_price:.2f}",
            f"₹{analysis['fast_ema']:.2f}",
            f"₹{analysis['slow_ema']:.2f}",
            analysis['crossover'].upper() if analysis['crossover'] != 'none' else '-',
            signal_text
        ])
        
        # Print detailed analysis
        print(f"Price: ₹{current_price:.2f} ({price_source})")
        print(f"EMA(9): ₹{analysis['fast_ema']:.2f}, EMA(21): ₹{analysis['slow_ema']:.2f}")
        print(f"Signal: {signal_text}")
    
    # Step 5: Print summary table
    print("\n" + "=" * 70)
    print("📊 WATCHLIST SUMMARY")
    print("=" * 70)
    
    headers = ["Symbol", "Price", "EMA(9)", "EMA(21)", "Crossover", "Signal"]
    print(tabulate(results, headers=headers, tablefmt="fancy_grid"))
    
    # Count signals
    signal_counts = {}
    for row in results:
        signal = row[5]
        signal_counts[signal] = signal_counts.get(signal, 0) + 1
    
    # Print signal statistics
    print("\n" + "=" * 70)
    print("📈 SIGNAL DISTRIBUTION")
    print("=" * 70)
    
    for signal, count in signal_counts.items():
        print(f"{signal}: {count} symbol(s)")
    
    print("\n✅ Analysis completed!")

if __name__ == "__main__":
    main()