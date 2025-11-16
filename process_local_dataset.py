import pandas as pd
import os
import glob
import sqlite3
from tqdm import tqdm
from datetime import datetime, timedelta

def process_stock_file(file_path, five_years_ago):
    """Reads a single stock CSV, processes it, and returns a DataFrame."""
    try:
        df = pd.read_csv(file_path)
        if df.empty:
            return None

        # Rename columns to match project requirements
        df.rename(columns={
            'Date': 'date', 'Open': 'open', 'High': 'high', 'Low': 'low',
            'Close': 'close', 'Volume': 'volume'
        }, inplace=True)

        # Convert date and filter for the last 5 years
        df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')
        df = df[df['date'] >= five_years_ago].copy()

        if df.empty:
            return None

        # Get ticker from filename
        ticker = os.path.basename(file_path).split('.')[0]
        df['code'] = ticker

        # Calculate 'amount'
        df['amount'] = df['close'] * df['volume']

        # Select and reorder columns
        required_cols = ['date', 'open', 'high', 'low', 'close', 'volume', 'amount', 'code']
        return df[required_cols]

    except Exception as e:
        print(f"Could not process file {os.path.basename(file_path)}: {e}")
        return None

def save_to_db(df, db_name='stock_large.db'):
    """Saves the final DataFrame to an SQLite database."""
    print(f"\nSaving data to SQLite database: {db_name}...")
    try:
        conn = sqlite3.connect(db_name)
        df.to_sql('raw_data', conn, if_exists='replace', index=False)
        conn.close()
        print(f"Data saved successfully to '{db_name}' in table 'raw_data'.")
    except Exception as e:
        print(f"Failed to save data to database: {e}")

def main():
    """Main function to process the local dataset."""
    base_path = r'C:\Users\2\.cache\kagglehub\datasets\paultimothymooney\stock-market-data\versions\74\stock_market_data'
    
    print("--- Starting Local Dataset Processing ---")
    print(f"Reading from base path: {base_path}")

    # Check if the path exists
    if not os.path.exists(base_path):
        print(f"Error: Dataset path does not exist: {base_path}")
        return

    # 1. Get all stock file paths from nasdaq and nyse
    nasdaq_files = glob.glob(os.path.join(base_path, 'nasdaq', 'csv', '*.csv'))
    nyse_files = glob.glob(os.path.join(base_path, 'nyse', 'csv', '*.csv'))
    all_files = nasdaq_files + nyse_files

    if not all_files:
        print("Error: No CSV files found in 'nasdaq/csv' or 'nyse/csv' directories.")
        return

    print(f"Found {len(all_files)} total stock files.")

    # 2. Select 100 stocks
    # For reproducibility, we sort the files and take the first 100.
    # You could use random.sample(all_files, 100) for random selection.
    all_files.sort()
    selected_files = all_files[:100]
    
    print(f"Selected {len(selected_files)} stocks for processing.")

    # 3. Process each file
    all_data_frames = []
    five_years_ago = datetime.now() - timedelta(days=5*365)

    for file_path in tqdm(selected_files, desc="Processing stock files"):
        processed_df = process_stock_file(file_path, five_years_ago)
        if processed_df is not None:
            all_data_frames.append(processed_df)

    if not all_data_frames:
        print("Error: No data could be processed. Exiting.")
        return

    # 4. Combine and save
    final_df = pd.concat(all_data_frames, ignore_index=True)
    final_df = final_df.sort_values(['code', 'date'])
    
    print(f"\nSuccessfully processed {len(all_data_frames)} stocks.")
    print(f"Final dataset contains {len(final_df)} rows.")

    save_to_db(final_df)

    print("\n--- Dataset Creation Complete ---")
    print("New database 'stock_large.db' has been created in the project directory.")
    print("To use it, please modify the database name in your project's data loading script (e.g., in 'data.py').")


if __name__ == '__main__':
    main()
