import os
import sys
import datetime
from pathlib import Path
import qlib
from qlib.data import D

# ================= 配置区域 =================
# 指向你的 qlib 数据目录
QLIB_DATA_DIR = r"C:\Users\K1ndred\.qlib\qlib_data\cn_data"
# Qlib 脚本的位置 (根据你的实际位置修改)
COLLECTOR_SCRIPT = r"D:\Code_new\Finance\qlib\scripts\data_collector\akshare\collector.py"
# ===========================================

def get_last_date():
    """获取当前数据库中最新的交易日"""
    try:
        qlib.init(provider_uri=QLIB_DATA_DIR)
        # 随便取一个指数（如沪深300）来看看最新日期
        # 如果你的数据里没有 SH000300，换个一直存在的股票代码，比如 'SH600000'
        df = D.features(["SH000300"], ["$close"], start_time="2024-01-01")
        if df.empty:
            return "2020-01-01" # 兜底
        last_date = df.index.get_level_values("datetime").max()
        return last_date.strftime("%Y-%m-%d")
    except Exception as e:
        print(f"无法读取现有数据日期，默认回溯7天: {e}")
        return (datetime.date.today() - datetime.timedelta(days=7)).strftime("%Y-%m-%d")

def run_update():
    # 1. 获取数据库里最新的日期
    last_date = get_last_date()
    # 2. 获取今天
    today = datetime.date.today().strftime("%Y-%m-%d")
    
    print(f"Current Data Last Date: {last_date}")
    print(f"Target Date: {today}")

    if last_date >= today:
        print("数据已经是新的，无需更新。")
        return

    # 3. 构造命令
    # 注意：Yahoo 的 collector 不一定支持精确的断点续传，通常建议多覆盖几天
    # start_date 设为 last_date，会重新下载最后一天以确保数据完整（比如昨天的收盘价修正）
    cmd = (
        f"python {COLLECTOR_SCRIPT} update_data_to_bin "
        f"--qlib_data_1d_dir {QLIB_DATA_DIR} "
        f"--trading_date {last_date} "
        f"--end_date {today}"
    )
    
    print(f"Running command: {cmd}")
    os.system(cmd)

if __name__ == "__main__":
    run_update()
