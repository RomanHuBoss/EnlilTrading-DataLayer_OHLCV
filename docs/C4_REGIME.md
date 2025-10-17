# C4.Regime — онлайн-детекция режима

Назначение: классификация баров рабочего ТФ в режимы `trend`/`flat` + флаг `high_rv`.
Ансамбль детекторов: ADX (D1), Donchian-ширина (D2), BOCPD (D3), HMM по дневной RV (D4) с правилом голосования 3‑из‑4 и гистерезисом.

## Входы

— Признаки C3 рабочего ТФ: минимум `f_adx14, f_pdi14, f_mdi14, f_donch_width_pct_20, f_logret1` или `f_close_z_60`, служебные `timestamp_ms,start_time_iso,symbol,tf`.
— Контекст C5 (рекомендуется): `ctx_rv_daily`, `ctx_rv_daily_pct`, `ctx_rv_daily_state`. При отсутствии включается резервная агрегация дневной RV из рабочих баров.

## Выходы

`regime`, `high_rv`, `regime_confidence`, `votes_trend`, `votes_flat`, `det_used`, `chgpt`, `p_bocpd_trend`, `p_bocpd_flat`, `s_adx`, `s_donch`, `s_hmm_rv`, `hysteresis_state`, `symbol`, `tf`, `build_version`.

## Производственные нормы

— Детерминированность при фиксированных входах и конфиге.
— Отсутствие сетевых вызовов.
— Время: ≤ 1 мс/бар без переобучения HMM; HMM переобучается вне горячего пути.

## Ограничения

— BOCPD реализован упрощённо, подменяет онлайн‑байесовский апдейт эвристикой run‑length.
— При отсутствии C5 high_rv вычисляется по перцентилю дневной RV (fallback).
