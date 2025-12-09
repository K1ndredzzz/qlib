@echo off
:: 获取今天的日期，格式为 YYYY-MM-DD
for /f %%i in ('powershell -c "Get-Date -Format 'yyyy-MM-dd'"') do set TODAY=%%i

:: 获取7天前的日期（作为起始日期，多更新几天防止漏数据，Qlib会自动覆盖）
for /f %%i in ('powershell -c "(Get-Date).AddDays(-7).ToString('yyyy-MM-dd')"') do set START_DATE=%%i

echo ==========================================
echo Starting Qlib Data Update...
echo From: %START_DATE%
echo To:   %TODAY%
echo ==========================================

:: 这里替换为你自己的 Qlib 路径和数据路径！
:: 记得激活你的 python 环境，比如：call conda activate qlib
python D:\Code_new\Finance\qlib\scripts\data_collector\akshare\collector.py update_data_to_bin ^
    --qlib_data_1d_dir C:\Users\K1ndred\.qlib\qlib_data\cn_data ^
    --trading_date 2010.01.01 ^
    --end_date 2015.01.01

pause
