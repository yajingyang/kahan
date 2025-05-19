import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

class TechnicalIndicators:
    """
    A class to calculate various technical indicators from OHLCV data for multiple entities.
    """
    
    def __init__(self, csv_file_path, output_dir):
        """
        Initialize with the path to the CSV file containing OHLCV data for multiple entities.
        
        Args:
            csv_file_path (str): Path to the CSV file
        """
        self.csv_file_path = csv_file_path
        self.output_dir = output_dir
        self.data = None
        self.entities = {}
        self.load_data()
    
    def load_data(self):
        """
        Load the OHLCV data from the CSV file.
        Assumes the CSV has columns: Date, Product Name, Symbol, Open, High, Low, Close, Volume
        """
        try:
            # Load the data with date parsing
            self.data = pd.read_csv(self.csv_file_path, parse_dates=['Date'])
            
            # Verify required columns
            required_columns = ['Date', 'Product Name', 'Symbol', 'Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in required_columns if col not in self.data.columns]
            
            if missing_columns:
                print(f"Warning: Missing required columns: {missing_columns}")
                return
            
            # Sort data by Date
            self.data.sort_values(by='Date', inplace=True)
            
            # Get unique entities (Product Names)
            self.entity_names = self.data['Product Name'].unique()
            
            # Create a separate dataframe for each entity
            for entity in self.entity_names:
                entity_data = self.data[self.data['Product Name'] == entity].copy()
                entity_data.set_index('Date', inplace=True)
                self.entities[entity] = entity_data
            
            print(f"Loaded data for {len(self.entity_names)} entities.")
        
        except Exception as e:
            print(f"Error loading data: {e}")
    
    def standardize_columns(self, entity_data):
        """
        Standardize column names to ensure they match expected OHLCV format.
        This is less necessary with the predefined column structure but kept for flexibility.
        """
        # Map of possible column names to standard names
        column_mapping = {
            'open': 'Open',
            'high': 'High', 
            'low': 'Low', 
            'close': 'Close',
            'volume': 'Volume',
            'adj close': 'Close',
            'adjusted close': 'Close',
            'adj. close': 'Close',
        }
        
        # Create a mapping for this specific DataFrame
        rename_dict = {}
        for col in entity_data.columns:
            for key, value in column_mapping.items():
                if key in col.lower() and col != value:
                    rename_dict[col] = value
                    break
        
        # Rename columns
        if rename_dict:
            entity_data.rename(columns=rename_dict, inplace=True)
        
        return entity_data
    
    def calculate_all_indicators_for_entity(self, entity_data):
        """
        Calculate all technical indicators for a specific entity and add them to the DataFrame.
        
        Args:
            entity_data: DataFrame containing OHLCV data for a single entity
        """
        # Moving Averages
        self.calculate_sma(entity_data, [10, 20, 30, 50, 100, 200])
        self.calculate_ema(entity_data, [20, 50, 100])
        
        # Oscillators
        self.calculate_rsi(entity_data)
        self.calculate_macd(entity_data)
        self.calculate_stochastic(entity_data)
        
        # Volatility Indicators
        self.calculate_bollinger_bands(entity_data)
        self.calculate_atr(entity_data, 20)
        self.calculate_volatility(entity_data)
        self.calculate_standard_deviation(entity_data)
        
        # Volume Indicators
        self.calculate_volume_indicators(entity_data)
        
        # Price Indicators
        self.calculate_price_indicators(entity_data)
        
        # Support & Resistance
        self.calculate_support_resistance(entity_data)
        
        # Momentum
        self.calculate_momentum(entity_data, [10, 30])
        
        return entity_data
    
    def calculate_all_indicators(self):
        """
        Calculate all technical indicators for all entities.
        """
        for entity_name in self.entity_names:
            self.calculate_all_indicators_for_entity(self.entities[entity_name])
        
        return self.entities

    def calculate_sma(self, data, periods):
        """
        Calculate Simple Moving Averages for specified periods.
        
        Args:
            periods (list): List of periods to calculate SMA for
        """
        for period in periods:
            data[f'SMA_{period}'] = data['Close'].rolling(window=period).mean()
    
    def calculate_ema(self, data, periods):
        """
        Calculate Exponential Moving Averages for specified periods.
        
        Args:
            periods (list): List of periods to calculate EMA for
        """
        for period in periods:
            data[f'EMA_{period}'] = data['Close'].ewm(span=period, adjust=False).mean()
    
    def calculate_rsi(self, data, period=14):
        """
        Calculate Relative Strength Index.
        
        Args:
            period (int): Period for RSI calculation, default is 14
        """
        delta = data['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        # Calculate RS for the first period using SMA
        rs = avg_gain / avg_loss
        
        # Calculate RSI
        data[f'RSI_{period}'] = 100 - (100 / (1 + rs))
    
    def calculate_macd(self, data, fast_period=12, slow_period=26, signal_period=9):
        """
        Calculate Moving Average Convergence Divergence.
        
        Args:
            fast_period (int): Fast EMA period, default is 12
            slow_period (int): Slow EMA period, default is 26
            signal_period (int): Signal line period, default is 9
        """
        # Calculate MACD line
        fast_ema = data['Close'].ewm(span=fast_period, adjust=False).mean()
        slow_ema = data['Close'].ewm(span=slow_period, adjust=False).mean()
        
        data['MACD_Line'] = fast_ema - slow_ema
        
        # Calculate signal line
        data['MACD_Signal'] = data['MACD_Line'].ewm(span=signal_period, adjust=False).mean()
        
        # Calculate MACD histogram
        data['MACD_Histogram'] = data['MACD_Line'] - data['MACD_Signal']
    
    def calculate_stochastic(self, data, k_period=14, d_period=3):
        """
        Calculate Stochastic Oscillator.
        
        Args:
            k_period (int): K period, default is 14
            d_period (int): D period, default is 3
        """
        # Calculate %K
        low_min = data['Low'].rolling(window=k_period).min()
        high_max = data['High'].rolling(window=k_period).max()
        
        data['Stochastic_%K'] = 100 * ((data['Close'] - low_min) / (high_max - low_min))
        
        # Calculate %D (3-day SMA of %K)
        data['Stochastic_%D'] = data['Stochastic_%K'].rolling(window=d_period).mean()
    
    def calculate_bollinger_bands(self, data, period=20, std_dev=2):
        """
        Calculate Bollinger Bands.
        
        Args:
            period (int): Period for moving average, default is 20
            std_dev (int): Number of standard deviations, default is 2
        """
        # Calculate middle band (20-day SMA)
        data['BB_Middle'] = data['Close'].rolling(window=period).mean()
        
        # Calculate standard deviation
        rolling_std = data['Close'].rolling(window=period).std()
        
        # Calculate upper and lower bands
        data['BB_Upper'] = data['BB_Middle'] + (rolling_std * std_dev)
        data['BB_Lower'] = data['BB_Middle'] - (rolling_std * std_dev)
        
        # Calculate bandwidth
        data['BB_Width'] = (data['BB_Upper'] - data['BB_Lower']) / data['BB_Middle']
    
    def calculate_atr(self, data, period=14):
        """
        Calculate Average True Range.
        
        Args:
            period (int): Period for ATR calculation, default is 14
        """
        # Calculate True Range
        data['TR'] = np.maximum(
            data['High'] - data['Low'],
            np.maximum(
                abs(data['High'] - data['Close'].shift(1)),
                abs(data['Low'] - data['Close'].shift(1))
            )
        )
        
        # Calculate ATR as the simple moving average of TR
        data[f'ATR_{period}'] = data['TR'].rolling(window=period).mean()
        
        # Drop the temporary TR column
        data.drop('TR', axis=1, inplace=True)
    
    def calculate_volatility(self, data):
        """
        Calculate volatility for different timeframes.
        """
        # Daily volatility (daily returns standard deviation)
        data['Daily_Return'] = data['Close'].pct_change()
        data['Daily_Volatility'] = data['Daily_Return'].rolling(window=20).std() * np.sqrt(252)
        
        # Weekly volatility
        if isinstance(data.index, pd.DatetimeIndex):
            # Only calculate if we have datetime index
            data['Weekly_Volatility'] = data['Daily_Return'].rolling(window=5).std() * np.sqrt(52)
        
        # Monthly volatility
        if isinstance(data.index, pd.DatetimeIndex):
            # Only calculate if we have datetime index
            data['Monthly_Volatility'] = data['Daily_Return'].rolling(window=21).std() * np.sqrt(12)
    
    def calculate_standard_deviation(self, data, period=30):
        """
        Calculate standard deviation for prices.
        
        Args:
            period (int): Period for standard deviation calculation, default is 30
        """
        # Daily standard deviation
        data['Daily_StdDev'] = data['Close'].pct_change().rolling(window=1).std()
        
        # 30-day standard deviation
        data[f'{period}Day_StdDev'] = data['Close'].pct_change().rolling(window=period).std()
    
    def calculate_volume_indicators(self, data):
        """
        Calculate volume-based indicators.
        """
        # Daily volume
        # Already exists as 'Volume'
        
        # Moving averages of volume
        for period in [10, 20, 30]:
            data[f'Volume_MA_{period}'] = data['Volume'].rolling(window=period).mean()
        
        # Volume rate of change
        data['Volume_ROC'] = data['Volume'].pct_change() * 100
        
        # Volume / Price correlation
        data['Vol_Price_Corr_20'] = data['Close'].rolling(window=20).corr(data['Volume'])
    
    def calculate_price_indicators(self, data):
        """
        Calculate price-based indicators.
        """
        # Daily price range
        data['Daily_Range'] = data['High'] - data['Low']
        data['Daily_Range_Pct'] = data['Daily_Range'] / data['Close'] * 100
        
        # Intraday range (if we have data at that granularity)
        # This is equivalent to Daily_Range if data is daily
        data['Intraday_Range'] = data['Daily_Range']
        
        # Weekly range (if possible)
        if isinstance(data.index, pd.DatetimeIndex):
            data['Weekly_High'] = data['High'].rolling(window=5).max()
            data['Weekly_Low'] = data['Low'].rolling(window=5).min()
            data['Weekly_Range'] = data['Weekly_High'] - data['Weekly_Low']
            data['Weekly_Range_Pct'] = data['Weekly_Range'] / data['Close'] * 100
        
        # 30-period range
        data['30Day_High'] = data['High'].rolling(window=30).max()
        data['30Day_Low'] = data['Low'].rolling(window=30).min()
        data['30Day_Range'] = data['30Day_High'] - data['30Day_Low']
        data['30Day_Range_Pct'] = data['30Day_Range'] / data['Close'] * 100
        
        # Price changes
        for period in [1, 5, 7, 30]:
            data[f'Price_Change_{period}d'] = data['Close'].diff(periods=period)
            data[f'Price_Change_Pct_{period}d'] = data['Close'].pct_change(periods=period) * 100
    
    def calculate_support_resistance(self, data, period=20):
        """
        Calculate basic support and resistance levels using pivot points.
        
        Args:
            period (int): Lookback period, default is 20
        """
        # Calculate pivot points
        data['Pivot'] = (data['High'] + data['Low'] + data['Close']) / 3
        
        # Calculate support and resistance levels
        data['Support1'] = (2 * data['Pivot']) - data['High']
        data['Support2'] = data['Pivot'] - (data['High'] - data['Low'])
        data['Resistance1'] = (2 * data['Pivot']) - data['Low']
        data['Resistance2'] = data['Pivot'] + (data['High'] - data['Low'])
        
        # Alternative support and resistance based on prior lows and highs
        data['Support_Level'] = data['Low'].rolling(window=period).min()
        data['Resistance_Level'] = data['High'].rolling(window=period).max()
    
    def calculate_momentum(self, data, periods):
        """
        Calculate momentum indicators for specified periods.
        
        Args:
            periods (list): List of periods to calculate momentum for
        """
        for period in periods:
            # Momentum as rate of change
            data[f'Momentum_{period}'] = data['Close'].diff(periods=period)
            
            # Percentage momentum
            data[f'Momentum_Pct_{period}'] = data['Close'].pct_change(periods=period) * 100
    
    def export_last_day_indicators(self):
        """
        Calculate indicators for all entities and export the last day's values to a text file.
        """
        output_file_path = Path(self.output_dir) / f"{self.csv_file_path.stem}.txt"

        with open(output_file_path, 'w') as f:
            
            # Process each entity
            for entity_name in self.entity_names:
                f.write(f"Entity: {entity_name}\n")
                # f.write("-"*50 + "\n")
                
                # Get the entity data
                entity_data = self.entities[entity_name].copy()
                
                # Skip if there's no data
                if len(entity_data) == 0:
                    f.write("No data available for this entity.\n\n")
                    continue
                
                # Calculate all indicators for this entity
                self.calculate_all_indicators_for_entity(entity_data)
                
                # Get the latest date's data
                latest_date = entity_data.index.max()
                latest_data = entity_data.loc[latest_date]
                
                # Write price information
                f.write("Price Information:\n")
                f.write(f"  Open: {latest_data['Open']:.2f}\n" if not pd.isna(latest_data['Open']) else "")    
                f.write(f"  High: {latest_data['High']:.2f}\n" if not pd.isna(latest_data['High']) else "")
                f.write(f"  Low: {latest_data['Low']:.2f}\n" if not pd.isna(latest_data['Low']) else "")
                f.write(f"  Close: {latest_data['Close']:.2f}\n" if not pd.isna(latest_data['Close']) else "")
                f.write(f"  Volume: {latest_data['Volume']:.0f}\n\n" if not pd.isna(latest_data['Volume']) else "\n")
                
                # Group indicators by category and write to file
                self.write_indicator_group(f, latest_data, "Moving Averages", 
                                           ['SMA_10', 'SMA_20', 'SMA_30', 'SMA_50', 'SMA_100', 'SMA_200',
                                            'EMA_20', 'EMA_50', 'EMA_100'])
                
                self.write_indicator_group(f, latest_data, "Oscillators", 
                                           ['RSI_14', 'MACD_Line', 'MACD_Signal', 'MACD_Histogram',
                                            'Stochastic_%K', 'Stochastic_%D'])
                
                self.write_indicator_group(f, latest_data, "Volatility Indicators", 
                                           ['BB_Upper', 'BB_Middle', 'BB_Lower', 'BB_Width',
                                            'ATR_20', 'Daily_Volatility', 'Weekly_Volatility', 'Monthly_Volatility',
                                            'Daily_StdDev', '30Day_StdDev'])
                
                self.write_indicator_group(f, latest_data, "Volume Indicators", 
                                           ['Volume', 'Volume_MA_10', 'Volume_MA_20', 'Volume_MA_30',
                                            'Volume_ROC', 'Vol_Price_Corr_20'])
                
                self.write_indicator_group(f, latest_data, "Price Indicators", 
                                           ['Daily_Range', 'Daily_Range_Pct', 'Weekly_Range', 'Weekly_Range_Pct',
                                            '30Day_Range', '30Day_Range_Pct',
                                            'Price_Change_1d', 'Price_Change_Pct_1d',
                                            'Price_Change_5d', 'Price_Change_Pct_5d',
                                            'Price_Change_7d', 'Price_Change_Pct_7d',
                                            'Price_Change_30d', 'Price_Change_Pct_30d'])
                
                self.write_indicator_group(f, latest_data, "Support & Resistance", 
                                           ['Pivot', 'Support1', 'Support2', 'Resistance1', 'Resistance2',
                                            'Support_Level', 'Resistance_Level'])
                
                self.write_indicator_group(f, latest_data, "Momentum", 
                                           ['Momentum_10', 'Momentum_Pct_10', 'Momentum_30', 'Momentum_Pct_30'])
                
                f.write("\n")
                # f.write("\n" + "="*50 + "\n\n")
            
            # f.write("\nAnalysis completed.\n")
        
        print(f"Last day indicators for all entities exported to: {output_file_path}")
        return output_file_path
    
    def write_indicator_group(self, file, data, group_name, indicators):
        """
        Helper method to write a group of indicators to the file.
        
        Args:
            file: The open file to write to
            data: DataFrame row with the indicator values
            group_name: Name of the indicator group
            indicators: List of indicator column names
        """
        file.write(f"{group_name}:\n")
        
        for indicator in indicators:
            if indicator in data.index:
                value = data[indicator]
                if value == np.nan or pd.isna(value):
                    continue
                if isinstance(value, (int, float, np.number)):
                    if 'Pct' in indicator or 'RSI' in indicator:
                        file.write(f"  {indicator}: {value:.2f}%\n")
                    else:
                        file.write(f"  {indicator}: {value:.4f}\n")
                else:
                    file.write(f"  {indicator}: {value}\n")
        
        file.write("\n")


def main():
    data_dir = Path("data\datatales")
    sub_data_dirs = [p for p in data_dir.iterdir() if p.is_dir()]
    for sub_data_dir in sub_data_dirs:
        sub_output_dir = Path("results/metric_values") / sub_data_dir.stem
        sub_output_dir.mkdir(exist_ok=True, parents=True)
        for data_path in (sub_data_dir / 'test').iterdir():
            try:
                indicators = TechnicalIndicators(data_path, sub_output_dir)
                
                indicators.calculate_all_indicators()   

                indicators.export_last_day_indicators()    
                
            except Exception as e:
                print(f"Error: {e}")


if __name__ == "__main__":
    main()