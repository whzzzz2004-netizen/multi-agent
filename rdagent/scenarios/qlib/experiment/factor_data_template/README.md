# How to read files.
For example, if you want to read `filename.h5`
```Python
import pandas as pd
df = pd.read_hdf("filename.h5", key="data")
```
NOTE: **key is always "data" for all hdf5 files **.

# Here is a short description about the data

| Filename       | Description                                                      |
| -------------- | -----------------------------------------------------------------|
| "daily_pv.h5"  | Adjusted daily price and volume data.                            |
| "minute_pv_sample.h5"  | Synthetic minute-level OHLCV/VWAP sample derived from daily data for pipeline testing. |
| "minute_quote_sample.h5"  | Synthetic minute-level bid/ask and size sample derived from daily data for pipeline testing. |


# For different data, We have some basic knowledge for them

## Daily price and volume data
$open: open price of the stock on that day.
$close: close price of the stock on that day.
$high: high price of the stock on that day.
$low: low price of the stock on that day.
$volume: volume of the stock on that day.
$factor: factor value of the stock on that day.

## Minute price and volume sample data
$open: minute open price.
$close: minute close price.
$high: minute high price.
$low: minute low price.
$volume: minute traded volume.
$vwap: minute volume weighted average price.

## Minute quote sample data
$bid1: best bid price at that minute.
$ask1: best ask price at that minute.
$bid1_size: best bid size at that minute.
$ask1_size: best ask size at that minute.
$mid_price: midpoint price computed from bid1 and ask1.
$spread_bps: bid-ask spread in basis points.

These minute files are synthetic samples generated from the daily dataset, intended to let the factor-mining pipeline test minute-level and quote-style logic. They are not true exchange-grade market microstructure data.
