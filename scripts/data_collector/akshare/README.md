# AkShare Data Collector for Qlib

This module provides data collection functionality using [AkShare](https://github.com/akfamily/akshare) API for Chinese A-share market data.

## Features

- **Chinese A-share Market Support**: Collect data for all A-share stocks (SH, SZ, BJ exchanges)
- **Index Data**: Automatic download of major index data (CSI300, CSI500, CSI100)
- **Price Adjustment**: Support for forward (qfq) and backward (hfq) price adjustment
- **Incremental Update**: Support for incremental data updates to existing Qlib data

## Requirements

```bash
pip install akshare
```

## Usage

### 1. Download Data

Download historical daily data:

```bash
# Download all A-share daily data from 2020-01-01 to 2024-01-01
python collector.py download_data \
    --source_dir ~/.qlib/stock_data/source \
    --start 2020-01-01 \
    --end 2024-01-01 \
    --delay 0.5 \
    --interval 1d

# Download with forward price adjustment (default)
python collector.py download_data \
    --source_dir ~/.qlib/stock_data/source \
    --start 2020-01-01 \
    --end 2024-01-01 \
    --adjust qfq

# Download with backward price adjustment
python collector.py download_data \
    --source_dir ~/.qlib/stock_data/source \
    --start 2020-01-01 \
    --end 2024-01-01 \
    --adjust hfq

# Download with no price adjustment
python collector.py download_data \
    --source_dir ~/.qlib/stock_data/source \
    --start 2020-01-01 \
    --end 2024-01-01 \
    --adjust ""
```

### 2. Normalize Data

Normalize downloaded data for Qlib:

```bash
python collector.py normalize_data \
    --source_dir ~/.qlib/stock_data/source \
    --normalize_dir ~/.qlib/stock_data/normalize
```

### 3. Dump to Qlib Binary Format

After normalization, dump data to Qlib binary format:

```bash
# Using the general dump_bin script
python scripts/dump_bin.py dump_all \
    --csv_path ~/.qlib/stock_data/normalize \
    --qlib_dir ~/.qlib/qlib_data/cn_data \
    --freq day \
    --exclude_fields date,symbol
```

### 4. Update Existing Qlib Data

Update existing Qlib data incrementally:

```bash
python collector.py update_data_to_bin \
    --qlib_data_1d_dir ~/.qlib/qlib_data/cn_data \
    --end_date 2024-01-01
```

### 5. Download Today's Data

Download the latest trading day's data:

```bash
python collector.py download_today_data \
    --source_dir ~/.qlib/stock_data/source \
    --delay 0.5
```

## Parameters

### Common Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `source_dir` | Directory for raw data | `./source` |
| `normalize_dir` | Directory for normalized data | `./normalize` |
| `max_workers` | Number of concurrent workers | `1` |
| `interval` | Data frequency (`1d`) | `1d` |
| `adjust` | Price adjustment type | `qfq` |

### download_data Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `start` | Start date (YYYY-MM-DD) | `2000-01-01` |
| `end` | End date (YYYY-MM-DD) | Today |
| `delay` | Delay between requests (seconds) | `0.5` |
| `max_collector_count` | Maximum retry count | `2` |
| `check_data_length` | Minimum data length check | `None` |
| `limit_nums` | Limit symbols (for debugging) | `None` |

### Price Adjustment Options

| Value | Description |
|-------|-------------|
| `qfq` | Forward adjustment (前复权) |
| `hfq` | Backward adjustment (后复权) |
| `""` | No adjustment |

## Data Format

### Raw Data (source)

CSV files with columns:
- `date`: Trading date
- `open`: Opening price
- `high`: Highest price
- `low`: Lowest price
- `close`: Closing price
- `volume`: Trading volume
- `symbol`: Stock symbol

### Normalized Data

CSV files with additional columns:
- `change`: Daily return
- `factor`: Adjustment factor

## Symbol Format

- Shanghai Exchange: `sh600000`, `SH600000`
- Shenzhen Exchange: `sz000001`, `SZ000001`
- Beijing Exchange: `bj430047`, `BJ430047`

## Notes

1. **Rate Limiting**: AkShare may have rate limits. Use `delay` parameter to avoid being blocked.
2. **Network Issues**: The collector has built-in retry mechanism for network failures.
3. **Data Quality**: AkShare data is sourced from public APIs and may have occasional issues.
4. **Index Data**: Index data (CSI300, CSI500, CSI100) is automatically downloaded.

## Comparison with Yahoo Collector

| Feature | AkShare | Yahoo |
|---------|---------|-------|
| Data Source | Chinese APIs | Yahoo Finance |
| CN Market | Excellent | Good |
| US Market | Limited | Excellent |
| Rate Limit | Moderate | Strict |
| Price Adjustment | Built-in | Manual |
| Network Access | No VPN needed | May need VPN |

## Troubleshooting

### Common Issues

1. **Import Error**: Make sure `akshare` is installed
   ```bash
   pip install akshare
   ```

2. **Network Timeout**: Increase delay or retry
   ```bash
   python collector.py download_data --delay 1.0
   ```

3. **Missing Data**: Some stocks may be delisted or have limited history

4. **Rate Limiting**: Reduce `max_workers` to 1 and increase `delay`

## License

MIT License - see the main Qlib repository for details.
