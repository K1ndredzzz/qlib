# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
AkShare Data Collector for Qlib

This module provides data collection functionality using AkShare API,
supporting Chinese A-share market data collection.

AkShare is an open-source financial data interface library for Python,
providing easy access to Chinese stock market data.

Usage Examples:
    # Download daily data
    $ python collector.py download_data --source_dir ~/.qlib/stock_data/source --start 2020-01-01 --end 2024-01-01 --delay 0.5

    # Normalize data
    $ python collector.py normalize_data --source_dir ~/.qlib/stock_data/source --normalize_dir ~/.qlib/stock_data/normalize

    # Update data to bin format
    $ python collector.py update_data_to_bin --qlib_data_1d_dir ~/.qlib/qlib_data/cn_data
"""

import abc
import sys
import copy
import time
import datetime
import importlib
import multiprocessing
from abc import ABC
from pathlib import Path
from typing import Iterable, List, Optional

import fire
import numpy as np
import pandas as pd
from loguru import logger

CUR_DIR = Path(__file__).resolve().parent
sys.path.append(str(CUR_DIR.parent.parent))

from data_collector.base import BaseCollector, BaseNormalize, BaseRun, Normalize
from data_collector.utils import (
    deco_retry,
    get_calendar_list,
)

try:
    import akshare as ak
except ImportError:
    raise ImportError("Please install akshare: pip install akshare")


# Index benchmark URL from eastmoney for downloading index data
INDEX_BENCH_URL = "http://push2his.eastmoney.com/api/qt/stock/kline/get?secid=1.{index_code}&fields1=f1%2Cf2%2Cf3%2Cf4%2Cf5&fields2=f51%2Cf52%2Cf53%2Cf54%2Cf55%2Cf56%2Cf57%2Cf58&klt=101&fqt=0&beg={begin}&end={end}"


def get_cn_stock_symbols() -> List[str]:
    """Get all Chinese A-share stock symbols using AkShare.

    Returns
    -------
    List[str]
        List of stock symbols in format like 'sh600000', 'sz000001'
    """
    logger.info("Getting CN stock symbols from AkShare...")

    @deco_retry(retry=5, retry_sleep=3)
    def _get_symbols():
        # Get all A-share stock info
        df = ak.stock_zh_a_spot_em()
        symbols = df["代码"].tolist()
        return symbols

    try:
        symbols = _get_symbols()
        # Format symbols: add exchange prefix
        formatted_symbols = []
        for s in symbols:
            if s.startswith("6"):
                formatted_symbols.append(f"sh{s}")
            elif s.startswith(("0", "3")):
                formatted_symbols.append(f"sz{s}")
            elif s.startswith("8") or s.startswith("4"):
                # Beijing stock exchange or other
                formatted_symbols.append(f"bj{s}")

        logger.info(f"Got {len(formatted_symbols)} symbols.")
        return sorted(set(formatted_symbols))
    except Exception as e:
        logger.error(f"Failed to get stock symbols: {e}")
        raise


class AkshareCollector(BaseCollector):
    """Base collector class for AkShare data source."""

    retry = 5  # Number of retries for network failures

    def __init__(
        self,
        save_dir: [str, Path],
        start=None,
        end=None,
        interval="1d",
        max_workers=4,
        max_collector_count=2,
        delay=0,
        check_data_length: int = None,
        limit_nums: int = None,
        adjust: str = "qfq",
    ):
        """
        Parameters
        ----------
        save_dir: str
            Directory to save stock data
        max_workers: int
            Number of workers, default 4
        max_collector_count: int
            Maximum retry count for collection, default 2
        delay: float
            Delay between requests (seconds), default 0
        interval: str
            Data frequency, value from ['1d'], default '1d'
        start: str
            Start datetime, default None (uses DEFAULT_START_DATETIME_1D)
        end: str
            End datetime, default None (uses current date)
        check_data_length: int
            Minimum data length check, default None
        limit_nums: int
            Limit number of symbols for debugging, default None
        adjust: str
            Price adjustment type: 'qfq' (forward), 'hfq' (backward), '' (none)
            Default is 'qfq' (forward adjusted)
        """
        super(AkshareCollector, self).__init__(
            save_dir=save_dir,
            start=start,
            end=end,
            interval=interval,
            max_workers=max_workers,
            max_collector_count=max_collector_count,
            delay=delay,
            check_data_length=check_data_length,
            limit_nums=limit_nums,
        )
        self.adjust = adjust
        self.init_datetime()

    def init_datetime(self):
        """Initialize datetime settings."""
        if self.interval == self.INTERVAL_1min:
            self.start_datetime = max(self.start_datetime, self.DEFAULT_START_DATETIME_1MIN)
        elif self.interval == self.INTERVAL_1d:
            pass
        else:
            raise ValueError(f"interval error: {self.interval}")

    @property
    @abc.abstractmethod
    def _timezone(self):
        raise NotImplementedError("rewrite _timezone")

    @staticmethod
    def get_data_from_remote(
        symbol: str,
        interval: str,
        start: str,
        end: str,
        adjust: str = "qfq"
    ) -> Optional[pd.DataFrame]:
        """
        Fetch stock data from AkShare.

        Parameters
        ----------
        symbol: str
            Stock symbol with exchange prefix (e.g., 'sh600000', 'sz000001')
        interval: str
            Data frequency ('1d' for daily)
        start: str
            Start date in format 'YYYYMMDD'
        end: str
            End date in format 'YYYYMMDD'
        adjust: str
            Price adjustment type: 'qfq', 'hfq', or ''

        Returns
        -------
        pd.DataFrame or None
            DataFrame with columns: date, open, high, low, close, volume, symbol
        """
        error_msg = f"{symbol}-{interval}-{start}-{end}"

        try:
            # Extract pure stock code (remove exchange prefix)
            if symbol.startswith(("sh", "sz", "bj")):
                stock_code = symbol[2:]
            else:
                stock_code = symbol

            if interval in ["1d", "day", "daily"]:
                # Use stock_zh_a_hist for daily data
                df = ak.stock_zh_a_hist(
                    symbol=stock_code,
                    period="daily",
                    start_date=start,
                    end_date=end,
                    adjust=adjust
                )

                if df is None or df.empty:
                    logger.warning(f"No data returned for {symbol}")
                    return None

                # Rename columns to standard format
                df = df.rename(columns={
                    "日期": "date",
                    "开盘": "open",
                    "收盘": "close",
                    "最高": "high",
                    "最低": "low",
                    "成交量": "volume",
                    "成交额": "amount",
                    "振幅": "amplitude",
                    "涨跌幅": "pct_change",
                    "涨跌额": "change",
                    "换手率": "turnover"
                })

                # Add symbol column
                df["symbol"] = symbol

                # Select and order columns
                columns_to_keep = ["date", "open", "high", "low", "close", "volume", "symbol"]
                if "amount" in df.columns:
                    columns_to_keep.append("amount")

                df = df[[col for col in columns_to_keep if col in df.columns]]

                return df
            else:
                raise ValueError(f"Unsupported interval: {interval}")

        except Exception as e:
            logger.warning(f"get data error: {error_msg}, exception: {e}")
            return None

    def get_data(
        self,
        symbol: str,
        interval: str,
        start_datetime: pd.Timestamp,
        end_datetime: pd.Timestamp
    ) -> pd.DataFrame:
        """
        Get stock data with retry mechanism.

        Parameters
        ----------
        symbol: str
            Stock symbol
        interval: str
            Data frequency
        start_datetime: pd.Timestamp
            Start datetime
        end_datetime: pd.Timestamp
            End datetime

        Returns
        -------
        pd.DataFrame
            Stock data DataFrame
        """
        @deco_retry(retry_sleep=self.delay, retry=self.retry)
        def _get_simple(start_, end_):
            self.sleep()
            # Format dates for AkShare API (YYYYMMDD)
            start_str = pd.Timestamp(start_).strftime("%Y%m%d")
            end_str = pd.Timestamp(end_).strftime("%Y%m%d")

            resp = self.get_data_from_remote(
                symbol,
                interval=interval,
                start=start_str,
                end=end_str,
                adjust=self.adjust
            )
            if resp is None or resp.empty:
                raise ValueError(
                    f"get data error: {symbol}--{start_}--{end_}. "
                    "The stock may be delisted or data unavailable."
                )
            return resp

        _result = None
        if interval == self.INTERVAL_1d:
            try:
                _result = _get_simple(start_datetime, end_datetime)
            except ValueError as e:
                logger.warning(f"Failed to get data for {symbol}: {e}")
        elif interval == self.INTERVAL_1min:
            raise ValueError("AkShare collector currently only supports daily data (1d)")
        else:
            raise ValueError(f"cannot support {self.interval}")

        return pd.DataFrame() if _result is None else _result

    def collector_data(self):
        """Collect data from AkShare."""
        super(AkshareCollector, self).collector_data()
        self.download_index_data()

    @abc.abstractmethod
    def download_index_data(self):
        """Download index data."""
        raise NotImplementedError("rewrite download_index_data")


class AkshareCollectorCN(AkshareCollector, ABC):
    """Chinese A-share market data collector."""

    def get_instrument_list(self):
        """Get list of Chinese A-share stock symbols."""
        logger.info("Getting CN stock symbols...")
        symbols = get_cn_stock_symbols()
        logger.info(f"Got {len(symbols)} symbols.")
        return symbols

    def normalize_symbol(self, symbol):
        """
        Normalize symbol format.

        Input: 'sh600000' or 'sz000001'
        Output: 'SH600000' or 'SZ000001'
        """
        return symbol.upper()

    @property
    def _timezone(self):
        return "Asia/Shanghai"


class AkshareCollectorCN1d(AkshareCollectorCN):
    """Chinese A-share daily data collector."""

    def download_index_data(self):
        """Download index data (CSI300, CSI500, CSI100)."""
        import requests

        _format = "%Y%m%d"
        _begin = self.start_datetime.strftime(_format)
        _end = self.end_datetime.strftime(_format)

        index_map = {
            "csi300": "000300",
            "csi100": "000903",
            "csi500": "000905"
        }

        for _index_name, _index_code in index_map.items():
            logger.info(f"Getting index data: {_index_name}({_index_code})...")
            try:
                # Use AkShare to get index data
                df = ak.index_zh_a_hist(
                    symbol=_index_code,
                    period="daily",
                    start_date=_begin,
                    end_date=_end
                )

                if df is None or df.empty:
                    logger.warning(f"No data for index {_index_name}")
                    continue

                # Rename columns
                df = df.rename(columns={
                    "日期": "date",
                    "开盘": "open",
                    "收盘": "close",
                    "最高": "high",
                    "最低": "low",
                    "成交量": "volume",
                    "成交额": "amount"
                })

                df["date"] = pd.to_datetime(df["date"])
                df = df.astype(float, errors="ignore")
                df["adjclose"] = df["close"]
                df["symbol"] = f"sh{_index_code}"

                _path = self.save_dir.joinpath(f"sh{_index_code}.csv")
                if _path.exists():
                    _old_df = pd.read_csv(_path)
                    df = pd.concat([_old_df, df], sort=False)
                    df = df.drop_duplicates(subset=["date"])
                df.to_csv(_path, index=False)

                logger.info(f"Saved index {_index_name} data to {_path}")
                time.sleep(1)

            except Exception as e:
                logger.warning(f"Failed to get {_index_name} index data: {e}")
                continue


class AkshareNormalize(BaseNormalize):
    """Base normalize class for AkShare data."""

    COLUMNS = ["open", "close", "high", "low", "volume"]
    DAILY_FORMAT = "%Y-%m-%d"

    @staticmethod
    def calc_change(df: pd.DataFrame, last_close: float) -> pd.Series:
        """Calculate daily price change percentage."""
        df = df.copy()
        _tmp_series = df["close"].ffill()
        _tmp_shift_series = _tmp_series.shift(1)
        if last_close is not None:
            _tmp_shift_series.iloc[0] = float(last_close)
        change_series = _tmp_series / _tmp_shift_series - 1
        return change_series

    @staticmethod
    def normalize_akshare(
        df: pd.DataFrame,
        calendar_list: list = None,
        date_field_name: str = "date",
        symbol_field_name: str = "symbol",
        last_close: float = None,
    ) -> pd.DataFrame:
        """
        Normalize AkShare data.

        Parameters
        ----------
        df: pd.DataFrame
            Raw data from AkShare
        calendar_list: list
            Trading calendar list
        date_field_name: str
            Date field name
        symbol_field_name: str
            Symbol field name
        last_close: float
            Last close price for calculating change

        Returns
        -------
        pd.DataFrame
            Normalized data
        """
        if df.empty:
            return df

        symbol = df.loc[df[symbol_field_name].first_valid_index(), symbol_field_name]
        columns = copy.deepcopy(AkshareNormalize.COLUMNS)
        df = df.copy()

        df.set_index(date_field_name, inplace=True)
        df.index = pd.to_datetime(df.index)
        df = df[~df.index.duplicated(keep="first")]

        if calendar_list is not None:
            df = df.reindex(
                pd.DataFrame(index=calendar_list)
                .loc[
                    pd.Timestamp(df.index.min()).date() : pd.Timestamp(df.index.max()).date()
                    + pd.Timedelta(hours=23, minutes=59)
                ]
                .index
            )

        df.sort_index(inplace=True)
        df.loc[(df["volume"] <= 0) | np.isnan(df["volume"]), list(set(df.columns) - {symbol_field_name})] = np.nan

        # Calculate change
        change_series = AkshareNormalize.calc_change(df, last_close)
        df["change"] = change_series

        columns += ["change"]
        df.loc[(df["volume"] <= 0) | np.isnan(df["volume"]), columns] = np.nan

        df[symbol_field_name] = symbol
        df.index.names = [date_field_name]
        return df.reset_index()

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize data."""
        df = self.normalize_akshare(df, self._calendar_list, self._date_field_name, self._symbol_field_name)
        df = self.adjusted_price(df)
        return df

    @abc.abstractmethod
    def adjusted_price(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply price adjustment."""
        raise NotImplementedError("rewrite adjusted_price")


class AkshareNormalize1d(AkshareNormalize, ABC):
    """Normalize class for daily data."""

    DAILY_FORMAT = "%Y-%m-%d"

    def adjusted_price(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply price adjustment for daily data.

        Since AkShare already provides adjusted prices (qfq/hfq),
        we mainly calculate the factor for compatibility with Qlib.
        """
        if df.empty:
            return df

        df = df.copy()
        df.set_index(self._date_field_name, inplace=True)

        # For AkShare with forward adjustment (qfq),
        # factor is already applied, so we set factor=1
        df["factor"] = 1.0

        df.index.names = [self._date_field_name]
        return df.reset_index()

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize daily data."""
        df = super(AkshareNormalize1d, self).normalize(df)
        df = self._manual_adj_data(df)
        return df

    def _get_first_close(self, df: pd.DataFrame) -> float:
        """Get first valid close price."""
        df = df.loc[df["close"].first_valid_index():]
        _close = df["close"].iloc[0]
        return _close

    def _manual_adj_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Manual adjust data: standardize all fields according to
        the close of the first day.
        """
        if df.empty:
            return df

        df = df.copy()
        df.sort_values(self._date_field_name, inplace=True)
        df = df.set_index(self._date_field_name)
        _close = self._get_first_close(df)

        for _col in df.columns:
            if _col in [self._symbol_field_name, "adjclose", "change", "factor"]:
                continue
            if _col == "volume":
                df[_col] = df[_col] * _close
            else:
                df[_col] = df[_col] / _close

        return df.reset_index()


class AkshareNormalizeCN:
    """Mixin class for Chinese market calendar."""

    def _get_calendar_list(self) -> Iterable[pd.Timestamp]:
        """Get Chinese market trading calendar."""
        return get_calendar_list("ALL")


class AkshareNormalizeCN1d(AkshareNormalizeCN, AkshareNormalize1d):
    """Normalize class for Chinese A-share daily data."""
    pass


class AkshareNormalize1dExtend(AkshareNormalize1d):
    """Extended normalize class for incremental updates."""

    def __init__(
        self,
        old_qlib_data_dir: [str, Path],
        date_field_name: str = "date",
        symbol_field_name: str = "symbol",
        **kwargs
    ):
        """
        Parameters
        ----------
        old_qlib_data_dir: str, Path
            Existing qlib data directory to extend
        date_field_name: str
            Date field name, default 'date'
        symbol_field_name: str
            Symbol field name, default 'symbol'
        """
        super(AkshareNormalize1dExtend, self).__init__(date_field_name, symbol_field_name)
        self.column_list = ["open", "high", "low", "close", "volume", "factor", "change"]
        self.old_qlib_data = self._get_old_data(old_qlib_data_dir)

    def _get_old_data(self, qlib_data_dir: [str, Path]):
        """Load existing qlib data."""
        import qlib
        from qlib.data import D

        qlib_data_dir = str(Path(qlib_data_dir).expanduser().resolve())
        qlib.init(provider_uri=qlib_data_dir, expression_cache=None, dataset_cache=None)
        df = D.features(D.instruments("all"), ["$" + col for col in self.column_list])
        df.columns = self.column_list
        return df

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize with extension to existing data."""
        df = super(AkshareNormalize1dExtend, self).normalize(df)
        df.set_index(self._date_field_name, inplace=True)
        symbol_name = df[self._symbol_field_name].iloc[0]
        old_symbol_list = self.old_qlib_data.index.get_level_values("instrument").unique().tolist()

        if str(symbol_name).upper() not in old_symbol_list:
            return df.reset_index()

        old_df = self.old_qlib_data.loc[str(symbol_name).upper()]
        latest_date = old_df.index[-1]
        df = df.loc[latest_date:]

        if len(df) < 2:
            return pd.DataFrame()

        new_latest_data = df.iloc[0]
        old_latest_data = old_df.loc[latest_date]

        for col in self.column_list[:-1]:
            if col == "volume":
                df[col] = df[col] / (new_latest_data[col] / old_latest_data[col])
            else:
                df[col] = df[col] * (old_latest_data[col] / new_latest_data[col])

        return df.drop(df.index[0]).reset_index()


class AkshareNormalizeCN1dExtend(AkshareNormalizeCN, AkshareNormalize1dExtend):
    """Extended normalize class for Chinese A-share incremental updates."""
    pass


class Run(BaseRun):
    """Main entry point for AkShare data collection."""

    def __init__(
        self,
        source_dir=None,
        normalize_dir=None,
        max_workers=1,
        interval="1d",
        adjust="qfq"
    ):
        """
        Parameters
        ----------
        source_dir: str
            Directory for raw data, default "Path(__file__).parent/source"
        normalize_dir: str
            Directory for normalized data, default "Path(__file__).parent/normalize"
        max_workers: int
            Number of workers, default 1; recommended 1 when collecting data
        interval: str
            Data frequency, value from ['1d'], default '1d'
        adjust: str
            Price adjustment: 'qfq' (forward), 'hfq' (backward), '' (none)
        """
        super().__init__(source_dir, normalize_dir, max_workers, interval)
        self.adjust = adjust

    @property
    def collector_class_name(self):
        return f"AkshareCollectorCN{self.interval}"

    @property
    def normalize_class_name(self):
        return f"AkshareNormalizeCN{self.interval}"

    @property
    def default_base_dir(self) -> [Path, str]:
        return CUR_DIR

    def download_data(
        self,
        max_collector_count=2,
        delay=0.5,
        start=None,
        end=None,
        check_data_length=None,
        limit_nums=None,
    ):
        """
        Download data from AkShare.

        Parameters
        ----------
        max_collector_count: int
            Maximum retry count, default 2
        delay: float
            Delay between requests (seconds), default 0.5
        start: str
            Start date, default "2000-01-01"; closed interval (including start)
        end: str
            End date, default current date; open interval (excluding end)
        check_data_length: int
            Minimum data length check, default None
        limit_nums: int
            Limit symbols for debugging, default None

        Examples
        --------
            # Get daily data
            $ python collector.py download_data --source_dir ~/.qlib/stock_data/source --start 2020-01-01 --end 2024-01-01 --delay 0.5 --interval 1d
        """
        if end is None:
            end = pd.Timestamp(datetime.datetime.now()).strftime("%Y-%m-%d")

        if self.interval == "1d" and pd.Timestamp(end) > pd.Timestamp(datetime.datetime.now().strftime("%Y-%m-%d")):
            end = pd.Timestamp(datetime.datetime.now()).strftime("%Y-%m-%d")
            logger.warning(f"End date adjusted to current date: {end}")

        _class = getattr(self._cur_module, self.collector_class_name)
        _class(
            self.source_dir,
            max_workers=self.max_workers,
            max_collector_count=max_collector_count,
            delay=delay,
            start=start,
            end=end,
            interval=self.interval,
            check_data_length=check_data_length,
            limit_nums=limit_nums,
            adjust=self.adjust,
        ).collector_data()

    def normalize_data(
        self,
        date_field_name: str = "date",
        symbol_field_name: str = "symbol",
        end_date: str = None,
        qlib_data_1d_dir: str = None,
    ):
        """
        Normalize data.

        Parameters
        ----------
        date_field_name: str
            Date field name, default 'date'
        symbol_field_name: str
            Symbol field name, default 'symbol'
        end_date: str
            End date for normalization, default None
        qlib_data_1d_dir: str
            Not used for daily data, reserved for compatibility

        Examples
        --------
            $ python collector.py normalize_data --source_dir ~/.qlib/stock_data/source --normalize_dir ~/.qlib/stock_data/normalize
        """
        super(Run, self).normalize_data(
            date_field_name, symbol_field_name, end_date=end_date, qlib_data_1d_dir=qlib_data_1d_dir
        )

    def normalize_data_1d_extend(
        self,
        old_qlib_data_dir,
        date_field_name: str = "date",
        symbol_field_name: str = "symbol"
    ):
        """
        Normalize data with extension to existing qlib data.

        Parameters
        ----------
        old_qlib_data_dir: str
            Existing qlib data directory
        date_field_name: str
            Date field name, default 'date'
        symbol_field_name: str
            Symbol field name, default 'symbol'

        Examples
        --------
            $ python collector.py normalize_data_1d_extend --old_qlib_dir ~/.qlib/qlib_data/cn_data --source_dir ~/.qlib/stock_data/source --normalize_dir ~/.qlib/stock_data/normalize
        """
        _class = getattr(self._cur_module, f"{self.normalize_class_name}Extend")
        yc = Normalize(
            source_dir=self.source_dir,
            target_dir=self.normalize_dir,
            normalize_class=_class,
            max_workers=self.max_workers,
            date_field_name=date_field_name,
            symbol_field_name=symbol_field_name,
            old_qlib_data_dir=old_qlib_data_dir,
        )
        yc.normalize()

    def download_today_data(
        self,
        max_collector_count=2,
        delay=0.5,
        check_data_length=None,
        limit_nums=None,
    ):
        """
        Download today's data.

        Parameters
        ----------
        max_collector_count: int
            Maximum retry count, default 2
        delay: float
            Delay between requests (seconds), default 0.5
        check_data_length: int
            Minimum data length check, default None
        limit_nums: int
            Limit symbols for debugging, default None

        Examples
        --------
            $ python collector.py download_today_data --source_dir ~/.qlib/stock_data/source --delay 0.5
        """
        start = datetime.datetime.now().date()
        end = pd.Timestamp(start + pd.Timedelta(days=1)).date()
        self.download_data(
            max_collector_count,
            delay,
            start.strftime("%Y-%m-%d"),
            end.strftime("%Y-%m-%d"),
            check_data_length,
            limit_nums,
        )

    def update_data_to_bin(
        self,
        qlib_data_1d_dir: str,
        end_date: str = None,
        check_data_length: int = None,
        delay: float = 1,
        exists_skip: bool = False,
    ):
        """
        Update AkShare data to qlib bin format.

        Parameters
        ----------
        qlib_data_1d_dir: str
            Qlib data directory to update
        end_date: str
            End date, default current date
        check_data_length: int
            Minimum data length check, default None
        delay: float
            Delay between requests (seconds), default 1
        exists_skip: bool
            Skip if data exists, default False

        Examples
        --------
            $ python collector.py update_data_to_bin --qlib_data_1d_dir ~/.qlib/qlib_data/cn_data --end_date 2024-01-01
        """
        from qlib.tests.data import GetData
        from qlib.utils import exists_qlib_data

        sys.path.append(str(CUR_DIR.parent.parent))
        from dump_bin import DumpDataUpdate

        if self.interval.lower() != "1d":
            logger.warning("Currently supports 1d data updates: --interval 1d")

        # Download qlib 1d data if not exists
        qlib_data_1d_dir = str(Path(qlib_data_1d_dir).expanduser().resolve())
        if not exists_qlib_data(qlib_data_1d_dir):
            GetData().qlib_data(
                target_dir=qlib_data_1d_dir, interval=self.interval, region="CN", exists_skip=exists_skip
            )

        # Get start/end date from calendar
        calendar_df = pd.read_csv(Path(qlib_data_1d_dir).joinpath("calendars/day.txt"))
        trading_date = (pd.Timestamp(calendar_df.iloc[-1, 0]) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        if end_date is None:
            end_date = (pd.Timestamp(trading_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        # Download data from AkShare
        self.download_data(delay=delay, start=trading_date, end=end_date, check_data_length=check_data_length)

        # Use more workers for normalization
        self.max_workers = (
            max(multiprocessing.cpu_count() - 2, 1)
            if self.max_workers is None or self.max_workers <= 1
            else self.max_workers
        )

        # Normalize data
        self.normalize_data_1d_extend(qlib_data_1d_dir)

        # Dump to bin format
        _dump = DumpDataUpdate(
            data_path=self.normalize_dir,
            qlib_dir=qlib_data_1d_dir,
            exclude_fields="symbol,date",
            max_workers=self.max_workers,
        )
        _dump.dump()

        # Parse index components
        try:
            get_instruments = getattr(
                importlib.import_module("data_collector.cn_index.collector"), "get_instruments"
            )
            for _index in ["CSI100", "CSI300"]:
                get_instruments(str(qlib_data_1d_dir), _index, market_index="cn_index")
        except Exception as e:
            logger.warning(f"Failed to update index components: {e}")


if __name__ == "__main__":
    fire.Fire(Run)
