# Trading parameters
WATCHLIST_FILE = "watchlist.csv"
EXCHANGE = "NSE_EQ"

# Strategy parameters
STRATEGY = {
    # EMA settings
    'ema_fast_period': 9,    # Fast EMA period
    'ema_slow_period': 21,   # Slow EMA period
    
    # RSI settings
    'rsi_period': 14,        # RSI calculation period
    'rsi_oversold': 30,      # RSI oversold threshold
    'rsi_overbought': 70,    # RSI overbought threshold,
    
    # Timeframe settings
    'trading_interval': 15,  # Trading interval in minutes (5, 15, 30, 60, etc.)
}
# Risk management
RISK = {
    'max_positions': 3,            # Maximum number of concurrent positions
    'max_position_value': 50000,   # Maximum value per position in INR
    'margin_multiplier': 5,        # Leverage/margin multiplier (5x default)
    'initial_stop_loss': 0.015,    # 1.5% Initial Stop Loss
    'trail_step': 0.005,           # 0.5% Trailing Step Movement
    'max_loss_per_day': 1000,      # Maximum loss per day in INR
    'take_profit_pct': 0.03,       # 3% Take profit target (optional)
}

# Trading hours
MARKET = {
    'open_time': '09:15',    # Market open time
    'close_time': '15:30',   # Market close time
}

# Define WATCHLIST properly
WATCHLIST = [
    "PFC", "ADANIPOWER", "JSWENERGY", 
    "ADANIENT", "KALYANKJIL", 
    "LODHA", "ADANIGREEN", "BHEL", 
    "BANKINDIA", "CUB"
]

# Telegram notifications
TELEGRAM = {
    'enabled': True,  # Set to False to disable Telegram notifications
    'token': '7954091977:AAEq6QthmSN_B6EM6T1MZRFumjDyhd5pmu0',  # Get from BotFather on Telegram
    'chat_id': '7973091088',  # Chat ID to send messages to
    'status_interval': 15,  # Send status updates every 15 minutes
    'send_entry_signals': True,  # Send notifications for entry signals
    'send_exit_signals': True,   # Send notifications for exit signals
    'send_indicator_updates': False  # Send detailed indicator updates (can be noisy)
}