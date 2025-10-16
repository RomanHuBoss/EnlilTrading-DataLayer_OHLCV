# C3 · Features.Core — спецификация

## Назначение

Детерминированные признаки поверх валидированных OHLCV (артефакты C2). Без внешних TA‑зависимостей; расчёты в pandas.

## Вход

Колонки (канон): `timestamp_ms:int64`, `start_time_iso:str(UTC)`, `open:float64`, `high:float64`, `low:float64`, `close:float64`, `volume:float64`, `turnover?:float64`.
Допустима внутренняя схема `o,h,l,c,v,(t?)` — автопереименование.

## Инварианты

* `min_periods = window` для всех rolling.
* `ddof=1` для std.
* `ε = 1e-12` для стабилизации делителей.
* Wilder‑сглаживание: `ewm(alpha=1/n, adjust=False)` для RSI/ATR/ADX.

## Формулы

* `logret_t = log(close_t/close_{t-1})`
* `rv_n = stdev(logret, n)`
* `TR = max(high-low, |high-close_{t-1}|, |low-close_{t-1}|)`; `ATR_n = EMA_Wilder(TR, n)`
* `RSI_n = 100 - 100/(1 + EMA_Wilder(up,n)/EMA_Wilder(down,n))`
* `±DI_n = 100*EMA_Wilder(±DM, n)/ATR_n`; `ADX_n = EMA_Wilder(DX, n)`
* Donchian_n: верх/низ окна; `break_dir ∈ {-1,0,1}`; `width_pct=(hi-lo)/close`
* `Z_n(x) = (x - mean_n(x))/(std_n(x)+ε)`
* `VWAP_roll_n = Σ(tp*vol)/Σ(vol)`; `VWAP_session` — кумулятивно в пределах UTC‑дня; `dev_pct = (close - vwap)/vwap`

## Набор признаков (по умолчанию)

* Доходности/вола: `f_ret1`, `f_logret1`, `f_rv_{20,60}`
* Свечные: `f_range_pct`, `f_body_pct`, `f_wick_upper_pct`, `f_wick_lower_pct`, `f_tr`, `f_atr_14`, `f_atr_pct_14`
* Тренд/моментум: `f_ema_20`, `f_ema_slope_20`, `f_mom_20`, `f_rsi14`, `f_pdi14`, `f_mdi14`, `f_adx14`
* Donchian 20: `f_donch_h_20`, `f_donch_l_20`, `f_donch_break_dir_20`, `f_donch_width_pct_20`
* Z‑score: `f_close_z_{20,60}`, `f_range_z_{20,60}`, `f_vol_z_{20,60}`
* Объёмы: `f_upvol_20`, `f_downvol_20`, `f_vol_balance_20`, `f_obv`
* VWAP: `f_vwap_roll_96`, `f_vwap_dev_pct_96`, `f_vwap_session`, `f_vwap_session_dev_pct`

## Версионирование сборок признаков

`f_build_version = C3.Features.Core@<semver>+<hash12>`; хэш по git‑ревизии и параметрам окна.

## Строгая проверка входа

CLI флаг `--strict` и параметр `strict` в `compute_features`:

* Наличие всех канонических колонок;
* Приведение типов;
* Отказ при NaN/inf в базовых колонках `open,high,low,close,volume`.
