# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
AkShare Data Collector for Qlib

This module provides data collection functionality using AkShare API
for Chinese A-share market data.
"""

from .collector import (
    AkshareCollector,
    AkshareCollectorCN,
    AkshareCollectorCN1d,
    AkshareNormalize,
    AkshareNormalize1d,
    AkshareNormalizeCN,
    AkshareNormalizeCN1d,
    AkshareNormalizeCN1dExtend,
    Run,
    get_cn_stock_symbols,
)

__all__ = [
    "AkshareCollector",
    "AkshareCollectorCN",
    "AkshareCollectorCN1d",
    "AkshareNormalize",
    "AkshareNormalize1d",
    "AkshareNormalizeCN",
    "AkshareNormalizeCN1d",
    "AkshareNormalizeCN1dExtend",
    "Run",
    "get_cn_stock_symbols",
]
