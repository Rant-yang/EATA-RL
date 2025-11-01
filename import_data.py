import pandas as pd
import os
import sqlite3
import glob

# --- 配置 ---
# 新数据的源路径
SOURCE_DATA_PATH = "C:\\Users\\2\\Desktop\\ETS-SDA\\ETS-SDA\\data"
# EATA项目中的数据库文件
DB_PATH = "C:\\Users\\2\\Desktop\\eata\\stock.db"
# 要存入的表名
TABLE_NAME = "downloaded"

def import_and_process_data():
    """
    读取、处理外部CSV数据，并将其导入到项目的SQLite数据库中。
    """
    data_folder = os.path.join(SOURCE_DATA_PATH, "djia30")
    
    if not os.path.isdir(data_folder):
        print(f"错误: 数据文件夹 {data_folder} 不存在。")
        return

    csv_files = glob.glob(os.path.join(data_folder, "*.csv"))
    
    if not csv_files:
        print(f"错误: 在 {data_folder} 中没有找到任何 .csv 文件。")
        return

    print(f"在 {data_folder} 中找到 {len(csv_files)} 个股票的CSV文件。开始处理...")

    all_stocks_df = []

    for file_path in csv_files:
        try:
            ticker = os.path.basename(file_path).replace(".csv", "")
            df = pd.read_csv(file_path)

            # 1. 重命名列 (更稳健的方式)
            rename_dict = {}
            for col in df.columns:
                # 将 'Date' 转为 'date', 'close_aapl' 转为 'close'
                rename_dict[col] = col.split('_')[0].lower()
            df.rename(columns=rename_dict, inplace=True)

            # 2. 添加 'code' 列
            df['code'] = ticker

            # 3. 计算 'amount' 列
            if 'close' in df.columns and 'volume' in df.columns:
                df['amount'] = df['close'] * df['volume']
            else:
                df['amount'] = 0

            # 4. 整理并选择最终需要的列
            required_cols = ['date', 'open', 'high', 'low', 'close', 'volume', 'amount', 'code']
            
            # 确保所有需要的列都存在，如果不存在则补0
            for col in required_cols:
                if col not in df.columns:
                    df[col] = 0
            
            final_df = df[required_cols]
            all_stocks_df.append(final_df)
            print(f"  已处理: {ticker}")

        except Exception as e:
            print(f"  处理文件 {file_path} 时出错: {e}")

    if not all_stocks_df:
        print("错误: 未能成功处理任何股票数据。")
        return

    # 5. 合并所有数据
    combined_df = pd.concat(all_stocks_df, ignore_index=True)
    print(f"\n数据合并完成，共 {len(combined_df)} 条记录，包含 {combined_df['code'].nunique()} 支股票。")

    # 6. 存入数据库
    try:
        print(f"\n正在连接数据库 {DB_PATH} 并写入数据到表 '{TABLE_NAME}'...")
        conn = sqlite3.connect(DB_PATH)
        combined_df.to_sql(TABLE_NAME, conn, if_exists='replace', index=False)
        conn.close()
        print("成功！真实股票数据已导入到项目中。")
        print("现在可以再次尝试运行 predict.py 功能测试了。")

    except Exception as e:
        print(f"存入数据库时发生严重错误: {e}")

if __name__ == "__main__":
    import_and_process_data()