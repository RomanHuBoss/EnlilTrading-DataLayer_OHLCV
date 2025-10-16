import pandas as pd
import numpy as np

from ohlcv.quality.validator import validate, QualityConfig, DQ_BITS


def _mk_1m_index(n, start="2024-01-01T00:00:00Z"):
    return pd.date_range(pd.Timestamp(start), periods=n, freq="min", tz="UTC")


def _mk_df(n=10, start="2024-01-01T00:00:00Z"):
    idx = _mk_1m_index(n, start)
    o = np.linspace(100, 100 + n - 1, n)
    c = o + 0.1
    h = np.maximum(o, c)
    low = np.minimum(o, c)
    v = np.ones(n)
    df = pd.DataFrame({"o": o, "h": h, "l": low, "c": c, "v": v}, index=idx)
    return df


def test_validate_empty_returns_empty():
    df = pd.DataFrame(columns=["o", "h", "l", "c", "v"]).set_index(pd.DatetimeIndex([], tz="UTC"))
    clean, issues = validate(df)
    assert clean.shape[0] == 0
    assert issues.shape[0] == 0


def test_fix_ohlc_invariants_and_neg_volume():
    df = _mk_df(3)
    # Нарушим инварианты на середине и отрицательный объём на последней
    i1, i2 = df.index[1], df.index[2]
    df.loc[i1, "h"] = min(df.loc[i1, "o"], df.loc[i1, "c"]) - 1.0
    df.loc[i1, "l"] = max(df.loc[i1, "o"], df.loc[i1, "c"]) + 1.0
    df.loc[i2, "v"] = -5.0

    clean, issues = validate(df, tf="1m", repair=True)
    # Инварианты должны быть исправлены
    assert (
        clean.loc[i1, "h"] >= max(clean.loc[i1, "o"], clean.loc[i1, "c"])
        and clean.loc[i1, "l"] <= min(clean.loc[i1, "o"], clean.loc[i1, "c"])
        and clean.loc[i1, "l"] <= clean.loc[i1, "h"]
    )
    # Объём неотрицателен
    assert clean.loc[i2, "v"] == 0.0
    # dq_flags должен быть ненулевым хотя бы на этих строках
    assert (clean.loc[[i1, i2], "dq_flags"] != 0).all()


def test_missing_small_fill_sets_is_gap_and_flags():
    df = _mk_df(10)
    # Удалим две минуты внутри — сознательно большой порог, чтобы автозаполнение сработало
    drop_idx = [df.index[3], df.index[7]]
    df = df.drop(drop_idx)

    cfg = QualityConfig(missing_fill_threshold=0.5)  # разрешим заполнение
    clean, issues = validate(df, tf="1m", repair=True, config=cfg)

    # Должны восстановиться 10 строк
    assert clean.shape[0] == 10
    # И на восстановленных минутах is_gap=True и есть флаги MISSING_*
    assert clean.loc[drop_idx, "is_gap"].all()
    for ts in drop_idx:
        assert clean.loc[ts, "dq_flags"] & (1 << DQ_BITS["MISSING_BARS"]) != 0
        assert clean.loc[ts, "dq_flags"] & (1 << DQ_BITS["MISSING_FILLED"]) != 0


def test_misaligned_timestamps_are_aligned_to_right_boundary():
    # Создадим метки, сдвинутые на +10 секунд — должны округлиться к правой границе минуты
    base = pd.Timestamp("2024-01-01T00:00:00Z")
    idx = pd.DatetimeIndex(
        [base + pd.Timedelta(seconds=10), base + pd.Timedelta(minutes=1, seconds=10)], tz="UTC"
    )
    df = pd.DataFrame(
        {
            "o": [100.0, 101.0],
            "h": [101.0, 102.0],
            "l": [99.0, 100.5],
            "c": [100.5, 101.5],
            "v": [1.0, 2.0],
        },
        index=idx,
    )

    cfg = QualityConfig(misaligned_tolerance_seconds=1)
    clean, issues = validate(df, tf="1m", repair=True, config=cfg)

    # Ожидаем метки ровно на границах минут: 00:01 и 00:02 (правая граница окна)
    assert (
        clean.index
        == pd.DatetimeIndex(
            [base + pd.Timedelta(minutes=1), base + pd.Timedelta(minutes=2)], tz="UTC"
        )
    ).all()
    # Наличие issue MISALIGNED_TS
    assert (issues["code"] == "MISALIGNED_TS").any()


def test_dq_notes_correspond_to_flags():
    df = _mk_df(1)
    # Сгенерируем конфликт инвариантов
    df.loc[df.index[0], ["h", "l"]] = [
        df.loc[df.index[0], "o"] - 1.0,
        df.loc[df.index[0], "c"] + 1.0,
    ]
    clean, issues = validate(df, tf="1m", repair=True)
    flags = int(clean.iloc[0]["dq_flags"])
    notes = clean.iloc[0]["dq_notes"]
    if flags == 0:
        assert notes == ""
    else:
        # Каждая пометка из notes должна соответствовать установленному биту
        for tag in notes.split(";"):
            assert (flags & (1 << DQ_BITS[tag])) != 0
