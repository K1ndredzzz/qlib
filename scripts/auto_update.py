#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Qlib 数据自动更新脚本 - 使用 akshare 数据源

功能:
    - 自动检测现有数据的最新日期
    - 增量下载个股日线数据
    - 同步更新指数数据（沪深300、中证500、中证1000等）
    - 数据标准化并转换为 qlib bin 格式

使用方法:
    python auto_update.py                    # 增量更新到最新
    python auto_update.py --full             # 全量更新（从2012年开始）
    python auto_update.py --start 2024-01-01 # 从指定日期开始更新

依赖:
    pip install akshare pandas loguru tqdm
"""

import os
import sys
import time
import datetime
import argparse
import multiprocessing
from pathlib import Path
from typing import List, Optional

import pandas as pd
import numpy as np
from tqdm import tqdm
from loguru import logger

try:
    import akshare as ak
except ImportError:
    raise ImportError("请先安装 akshare: pip install akshare")

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

try:
    import qlib
    from qlib.data import D
except ImportError:
    qlib = None
    D = None

# ================= 配置区域 =================
DEFAULT_QLIB_DATA_DIR = Path.home() / ".qlib" / "qlib_data" / "cn_data"
DEFAULT_START_DATE = "2012-01-01"

INDEX_CONFIG = {
    "csi300": ("000300", "sh000300"),
    "csi500": ("000905", "sh000905"),
    "csi1000": ("000852", "sh000852"),
    "csi100": ("000903", "sh000903"),
    "sse50": ("000016", "sh000016"),
}

MAX_WORKERS = 4
DOWNLOAD_DELAY = 0.1
# ===========================================


class AkShareDataCollector:
    """使用 akshare 收集股票数据"""

    def __init__(self, qlib_data_dir, start_date=None, end_date=None, max_workers=MAX_WORKERS):
        self.qlib_data_dir = Path(qlib_data_dir)
        self.start_date = start_date or DEFAULT_START_DATE
        self.end_date = end_date or datetime.date.today().strftime("%Y-%m-%d")
        self.max_workers = max_workers
        self.source_dir = self.qlib_data_dir / "_temp_source"
        self.normalize_dir = self.qlib_data_dir / "_temp_normalize"
        self.source_dir.mkdir(parents=True, exist_ok=True)
        self.normalize_dir.mkdir(parents=True, exist_ok=True)

    def get_stock_list(self) -> List[str]:
        """获取A股股票列表"""
        logger.info("获取A股股票列表...")
        try:
            df_sh = ak.stock_sh_a_spot_em()
            df_sz = ak.stock_sz_a_spot_em()
            sh_codes = df_sh["代码"].tolist() if "代码" in df_sh.columns else []
            sz_codes = df_sz["代码"].tolist() if "代码" in df_sz.columns else []
            symbols = []
            for code in sh_codes:
                if code.startswith(("6", "688")):
                    symbols.append(f"sh{code}")
            for code in sz_codes:
                if code.startswith(("0", "3")):
                    symbols.append(f"sz{code}")
            logger.info(f"共获取 {len(symbols)} 只股票")
            return sorted(symbols)
        except Exception as e:
            logger.error(f"获取股票列表失败: {e}")
            return self._get_symbols_from_qlib()

    def _get_symbols_from_qlib(self) -> List[str]:
        """从现有 qlib 数据获取股票列表"""
        instruments_file = self.qlib_data_dir / "instruments" / "all.txt"
        if instruments_file.exists():
            df = pd.read_csv(instruments_file, sep="\t", header=None, names=["symbol", "start", "end"])
            return df["symbol"].str.lower().tolist()
        return []

    def download_stock_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """下载单只股票的日线数据"""
        try:
            code = symbol[2:]
            df = ak.stock_zh_a_hist(
                symbol=code,
                period="daily",
                start_date=self.start_date.replace("-", ""),
                end_date=self.end_date.replace("-", ""),
                adjust="qfq"
            )
            if df is None or df.empty:
                return None
            df = df.rename(columns={
                "日期": "date", "开盘": "open", "收盘": "close",
                "最高": "high", "最低": "low", "成交量": "volume",
            })
            required_cols = ["date", "open", "close", "high", "low", "volume"]
            df = df[[c for c in required_cols if c in df.columns]]
            df["symbol"] = symbol.upper()
            df["date"] = pd.to_datetime(df["date"])
            return df
        except Exception as e:
            logger.debug(f"下载 {symbol} 失败: {e}")
            return None

    def download_index_data(self, index_code: str, symbol: str) -> Optional[pd.DataFrame]:
        """下载指数日线数据"""
        try:
            df = ak.stock_zh_index_daily(symbol=f"sh{index_code}")
            if df is None or df.empty:
                return None
            df["date"] = pd.to_datetime(df["date"])
            df = df[(df["date"] >= pd.Timestamp(self.start_date)) &
                    (df["date"] <= pd.Timestamp(self.end_date))]
            df["symbol"] = symbol.upper()
            required_cols = ["date", "open", "close", "high", "low", "volume", "symbol"]
            df = df[[c for c in required_cols if c in df.columns]]
            return df
        except Exception as e:
            logger.error(f"下载指数 {index_code} 失败: {e}")
            return None

    def normalize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """标准化数据为 qlib 格式"""
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.drop_duplicates(subset=["date"], keep="last")
        df = df.sort_values("date")
        if "close" in df.columns:
            df["factor"] = 1.0
            df["change"] = df["close"].pct_change()
        df.loc[(df["volume"] <= 0) | df["volume"].isna(),
               ["open", "high", "low", "close", "volume"]] = np.nan
        return df

    def save_to_csv(self, df: pd.DataFrame, symbol: str):
        """保存数据到 CSV 文件"""
        if df is None or df.empty:
            return
        file_path = self.normalize_dir / f"{symbol.upper()}.csv"
        if file_path.exists():
            old_df = pd.read_csv(file_path)
            old_df["date"] = pd.to_datetime(old_df["date"])
            df = pd.concat([old_df, df], ignore_index=True)
            df = df.drop_duplicates(subset=["date"], keep="last")
            df = df.sort_values("date")
        df.to_csv(file_path, index=False)

    def collect_all_stocks(self, symbols=None):
        """收集所有股票数据"""
        if symbols is None:
            symbols = self.get_stock_list()
        logger.info(f"开始下载 {len(symbols)} 只股票数据...")
        success_count = 0
        failed_symbols = []
        with tqdm(total=len(symbols), desc="下载股票数据") as pbar:
            for symbol in symbols:
                df = self.download_stock_data(symbol)
                if df is not None and not df.empty:
                    df = self.normalize_data(df)
                    self.save_to_csv(df, symbol)
                    success_count += 1
                else:
                    failed_symbols.append(symbol)
                pbar.update(1)
                time.sleep(DOWNLOAD_DELAY)
        logger.info(f"下载完成: 成功 {success_count}, 失败 {len(failed_symbols)}")
        if failed_symbols and len(failed_symbols) <= 20:
            logger.warning(f"失败的股票: {failed_symbols}")

    def collect_all_indices(self):
        """收集所有指数数据"""
        logger.info("开始下载指数数据...")
        for name, (code, symbol) in INDEX_CONFIG.items():
            logger.info(f"下载 {name} ({symbol})...")
            df = self.download_index_data(code, symbol)
            if df is not None and not df.empty:
                df = self.normalize_data(df)
                self.save_to_csv(df, symbol)
                logger.info(f"{name} 下载成功")
            else:
                logger.warning(f"{name} 下载失败")
            time.sleep(DOWNLOAD_DELAY)

    def dump_to_bin(self):
        """转换数据为 qlib bin 格式"""
        logger.info("转换数据为 qlib bin 格式...")
        try:
            calendar_file = self.qlib_data_dir / "calendars" / "day.txt"
            max_workers = max(multiprocessing.cpu_count() - 2, 1)

            if calendar_file.exists():
                # 增量更新模式：日历文件已存在
                from dump_bin import DumpDataUpdate
                logger.info("使用增量更新模式 (DumpDataUpdate)...")
                _dump = DumpDataUpdate(
                    data_path=str(self.normalize_dir),
                    qlib_dir=str(self.qlib_data_dir),
                    exclude_fields="symbol,date",
                    max_workers=max_workers,
                )
            else:
                # 全量初始化模式：首次创建数据
                from dump_bin import DumpDataAll
                logger.info("使用全量初始化模式 (DumpDataAll)...")
                _dump = DumpDataAll(
                    data_path=str(self.normalize_dir),
                    qlib_dir=str(self.qlib_data_dir),
                    exclude_fields="symbol,date",
                    max_workers=max_workers,
                )

            _dump.dump()
            logger.info("数据转换完成")
        except Exception as e:
            logger.error(f"数据转换失败: {e}")
            raise

    def cleanup(self):
        """清理临时文件"""
        import shutil
        try:
            if self.source_dir.exists():
                shutil.rmtree(self.source_dir)
            if self.normalize_dir.exists():
                shutil.rmtree(self.normalize_dir)
            logger.info("临时文件清理完成")
        except Exception as e:
            logger.warning(f"清理临时文件失败: {e}")


def get_last_date(qlib_data_dir):
    """获取当前数据库中最新的交易日"""
    try:
        calendar_file = Path(qlib_data_dir) / "calendars" / "day.txt"
        if calendar_file.exists():
            df = pd.read_csv(calendar_file, header=None)
            last_date = pd.Timestamp(df.iloc[-1, 0])
            return last_date.strftime("%Y-%m-%d")
    except Exception as e:
        logger.warning(f"无法读取现有数据日期: {e}")
    return (datetime.date.today() - datetime.timedelta(days=7)).strftime("%Y-%m-%d")


def run_update(qlib_data_dir=None, start_date=None, end_date=None, full_update=False,
               skip_stock=False, skip_index=False, skip_dump=False, cleanup=True):
    """运行数据更新"""
    if qlib_data_dir is None:
        qlib_data_dir = DEFAULT_QLIB_DATA_DIR
    qlib_data_dir = Path(qlib_data_dir)

    # 如果数据目录不存在，自动创建并进行全量下载
    if not qlib_data_dir.exists():
        logger.warning(f"数据目录不存在: {qlib_data_dir}")
        logger.info("将自动创建目录并进行全量下载...")
        qlib_data_dir.mkdir(parents=True, exist_ok=True)
        full_update = True  # 强制全量更新

    if full_update:
        start_date = DEFAULT_START_DATE
    elif start_date is None:
        last_date = get_last_date(qlib_data_dir)
        start_date = (pd.Timestamp(last_date) - pd.Timedelta(days=3)).strftime("%Y-%m-%d")

    if end_date is None:
        end_date = datetime.date.today().strftime("%Y-%m-%d")

    logger.info(f"数据目录: {qlib_data_dir}")
    logger.info(f"更新范围: {start_date} -> {end_date}")

    if start_date >= end_date:
        logger.info("数据已是最新，无需更新")
        return

    collector = AkShareDataCollector(
        qlib_data_dir=qlib_data_dir,
        start_date=start_date,
        end_date=end_date,
    )

    try:
        if not skip_stock:
            collector.collect_all_stocks()
        if not skip_index:
            collector.collect_all_indices()
        if not skip_dump:
            collector.dump_to_bin()
        logger.info("数据更新完成!")
    except Exception as e:
        logger.error(f"更新失败: {e}")
        raise
    finally:
        if cleanup:
            collector.cleanup()


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(description="Qlib 数据自动更新脚本 (akshare 数据源)")
    parser.add_argument("--qlib_data_dir", type=str, default=None, help="qlib 数据目录")
    parser.add_argument("--start", type=str, default=None, help="开始日期 (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None, help="结束日期 (YYYY-MM-DD)")
    parser.add_argument("--full", action="store_true", help="全量更新 (从2010年开始)")
    parser.add_argument("--skip-stock", action="store_true", help="跳过股票数据下载")
    parser.add_argument("--skip-index", action="store_true", help="跳过指数数据下载")
    parser.add_argument("--skip-dump", action="store_true", help="跳过 bin 格式转换")
    parser.add_argument("--no-cleanup", action="store_true", help="不清理临时文件")
    parser.add_argument("--debug", action="store_true", help="启用调试模式")

    args = parser.parse_args()

    if args.debug:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")
    else:
        logger.remove()
        logger.add(sys.stderr, level="INFO")

    run_update(
        qlib_data_dir=args.qlib_data_dir,
        start_date=args.start,
        end_date=args.end,
        full_update=args.full,
        skip_stock=args.skip_stock,
        skip_index=args.skip_index,
        skip_dump=args.skip_dump,
        cleanup=not args.no_cleanup,
    )


if __name__ == "__main__":
    main()
