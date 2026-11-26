"""Load CERT r4.2 (or synthetic sample) logs and build daily early-warning features."""

from pathlib import Path
import numpy as np
import pandas as pd

import config


def _is_after_hours(dt_series: pd.Series) -> pd.Series:
    hour = dt_series.dt.hour
    is_weekend = dt_series.dt.weekday >= 5
    return (hour < config.WORK_START_HOUR) | (hour >= config.WORK_END_HOUR) | is_weekend


def _aggregate_logon(path: Path) -> pd.DataFrame:
    parts = []
    for chunk in pd.read_csv(path, usecols=["date", "user", "pc", "activity"], chunksize=config.CHUNK_SIZE):
        dt = pd.to_datetime(chunk["date"], format="%m/%d/%Y %H:%M:%S", errors="coerce")
        chunk = chunk.assign(
            day=dt.dt.floor("D"),
            is_logon=(chunk["activity"] == "Logon").astype(int),
            is_after_hours=((chunk["activity"] == "Logon") & _is_after_hours(dt)).astype(int),
        ).dropna(subset=["day"])

        agg = chunk.groupby(["user", "day"], as_index=False).agg(
            logon_count=("is_logon", "sum"),
            after_hours_logon_count=("is_after_hours", "sum"),
            unique_pcs=("pc", "nunique"),
        )
        parts.append(agg)

    if not parts:
        return pd.DataFrame(columns=["user", "day", "logon_count", "after_hours_logon_count", "unique_pcs"])

    df = pd.concat(parts, ignore_index=True)
    return df.groupby(["user", "day"], as_index=False).agg(
        logon_count=("logon_count", "sum"),
        after_hours_logon_count=("after_hours_logon_count", "sum"),
        unique_pcs=("unique_pcs", "max"),
    )


def _aggregate_device(path: Path) -> pd.DataFrame:
    parts = []
    for chunk in pd.read_csv(path, usecols=["date", "user", "activity"], chunksize=config.CHUNK_SIZE):
        dt = pd.to_datetime(chunk["date"], format="%m/%d/%Y %H:%M:%S", errors="coerce")
        is_connect = chunk["activity"] == "Connect"
        chunk = chunk.assign(
            day=dt.dt.floor("D"),
            usb_connect_count=is_connect.astype(int),
            after_hours_usb_count=(is_connect & _is_after_hours(dt)).astype(int),
        ).dropna(subset=["day"])

        agg = chunk.groupby(["user", "day"], as_index=False).agg(
            usb_connect_count=("usb_connect_count", "sum"),
            after_hours_usb_count=("after_hours_usb_count", "sum"),
        )
        parts.append(agg)

    if not parts:
        return pd.DataFrame(columns=["user", "day", "usb_connect_count", "after_hours_usb_count"])

    df = pd.concat(parts, ignore_index=True)
    return df.groupby(["user", "day"], as_index=False).sum()


def _aggregate_file(path: Path) -> pd.DataFrame:
    parts = []
    for chunk in pd.read_csv(path, usecols=["date", "user"], chunksize=config.CHUNK_SIZE):
        dt = pd.to_datetime(chunk["date"], format="%m/%d/%Y %H:%M:%S", errors="coerce")
        chunk = chunk.assign(
            day=dt.dt.floor("D"),
            file_access_count=1,
            after_hours_file_count=_is_after_hours(dt).astype(int),
        ).dropna(subset=["day"])

        agg = chunk.groupby(["user", "day"], as_index=False).agg(
            file_access_count=("file_access_count", "sum"),
            after_hours_file_count=("after_hours_file_count", "sum"),
        )
        parts.append(agg)

    if not parts:
        return pd.DataFrame(columns=["user", "day", "file_access_count", "after_hours_file_count"])

    df = pd.concat(parts, ignore_index=True)
    return df.groupby(["user", "day"], as_index=False).sum()


def _aggregate_email(path: Path) -> pd.DataFrame:
    parts = []
    usecols = ["date", "user", "to", "cc", "bcc", "size", "attachments"]
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=config.CHUNK_SIZE):
        dt = pd.to_datetime(chunk["date"], format="%m/%d/%Y %H:%M:%S", errors="coerce")
        recipients = (
            chunk["to"].fillna("").astype(str)
            + ";"
            + chunk["cc"].fillna("").astype(str)
            + ";"
            + chunk["bcc"].fillna("").astype(str)
        )
        has_external = recipients.apply(
            lambda s: int(any(addr.strip() and not addr.strip().lower().endswith(config.COMPANY_DOMAIN) for addr in s.split(";")))
        )
        chunk = chunk.assign(
            day=dt.dt.floor("D"),
            email_count=1,
            external_email_count=has_external,
            total_attachments=pd.to_numeric(chunk["attachments"], errors="coerce").fillna(0),
            total_email_size=pd.to_numeric(chunk["size"], errors="coerce").fillna(0),
        ).dropna(subset=["day"])

        agg = chunk.groupby(["user", "day"], as_index=False).agg(
            email_count=("email_count", "sum"),
            external_email_count=("external_email_count", "sum"),
            total_attachments=("total_attachments", "sum"),
            total_email_size=("total_email_size", "sum"),
        )
        parts.append(agg)

    if not parts:
        return pd.DataFrame(
            columns=["user", "day", "email_count", "external_email_count", "total_attachments", "total_email_size"]
        )

    df = pd.concat(parts, ignore_index=True)
    return df.groupby(["user", "day"], as_index=False).sum()


def _aggregate_http(path: Path) -> pd.DataFrame:
    """Stream http.csv in chunks (skipping the huge content column) to extract web signals."""
    cols = ["user", "day", "http_count", "after_hours_http_count", "job_search_http_count", "upload_cloud_http_count"]
    if not path.exists():
        return pd.DataFrame(columns=cols)

    parts = []
    chunk_num = 0
    for chunk in pd.read_csv(path, usecols=["date", "user", "url"], chunksize=config.CHUNK_SIZE):
        chunk_num += 1
        if chunk_num % 20 == 0:
            print(f"  ...parsed {chunk_num * config.CHUNK_SIZE:,} http.csv rows")

        dt = pd.to_datetime(chunk["date"], format="%m/%d/%Y %H:%M:%S", errors="coerce")
        url_str = chunk["url"].fillna("").astype(str)

        chunk = chunk.assign(
            day=dt.dt.floor("D"),
            http_count=1,
            after_hours_http_count=_is_after_hours(dt).astype(int),
            job_search_http_count=url_str.str.contains(config.JOB_SEARCH_URL_PATTERN, case=False, regex=True).astype(int),
            upload_cloud_http_count=url_str.str.contains(config.UPLOAD_CLOUD_URL_PATTERN, case=False, regex=True).astype(int),
        ).dropna(subset=["day"])

        agg = chunk.groupby(["user", "day"], as_index=False).agg(
            http_count=("http_count", "sum"),
            after_hours_http_count=("after_hours_http_count", "sum"),
            job_search_http_count=("job_search_http_count", "sum"),
            upload_cloud_http_count=("upload_cloud_http_count", "sum"),
        )
        parts.append(agg)

    if not parts:
        return pd.DataFrame(columns=cols)

    df = pd.concat(parts, ignore_index=True)
    return df.groupby(["user", "day"], as_index=False).sum()


def _load_psychometric(path: Path) -> pd.DataFrame:
    cols = ["user", "O", "C", "E", "A", "N"]
    if not path.exists():
        return pd.DataFrame(columns=cols)

    df = pd.read_csv(path)
    if "user_id" in df.columns and "user" not in df.columns:
        df = df.rename(columns={"user_id": "user"})
    keep = [c for c in cols if c in df.columns]
    return df[keep].drop_duplicates(subset=["user"])


def load_insiders(data_dir: Path) -> pd.DataFrame:
    insiders_path = data_dir / "insiders.csv"
    ins = pd.read_csv(insiders_path)
    ins = ins[ins["dataset"].astype(str).str.strip() == "4.2"].copy()
    ins["attack_start_day"] = pd.to_datetime(ins["start"], format="%m/%d/%Y %H:%M:%S", errors="coerce").dt.floor("D")
    ins = ins.dropna(subset=["attack_start_day"])
    ins = ins.sort_values("attack_start_day").groupby("user", as_index=False).first()
    return ins[["user", "scenario", "attack_start_day"]]


def _load_or_build_daily_base(data_dir: Path, use_cache: bool, rebuild_cache: bool) -> pd.DataFrame:
    cache_path = data_dir / "daily_base_cache.csv"
    if use_cache and not rebuild_cache and cache_path.exists():
        print(f"Loading cached daily aggregates from {cache_path} (use --rebuild-cache to re-parse raw CSVs)...")
        return pd.read_csv(cache_path, parse_dates=["day"])

    print("Aggregating logon.csv...")
    logon = _aggregate_logon(data_dir / "logon.csv")
    print("Aggregating device.csv...")
    device = _aggregate_device(data_dir / "device.csv")
    print("Aggregating file.csv...")
    files = _aggregate_file(data_dir / "file.csv")
    print("Aggregating email.csv...")
    email = _aggregate_email(data_dir / "email.csv")
    print("Aggregating http.csv...")
    http = _aggregate_http(data_dir / "http.csv")

    df = logon.merge(device, on=["user", "day"], how="outer")
    df = df.merge(files, on=["user", "day"], how="outer")
    df = df.merge(email, on=["user", "day"], how="outer")
    df = df.merge(http, on=["user", "day"], how="outer")

    if use_cache:
        df.to_csv(cache_path, index=False)
        print(f"Saved daily base cache to {cache_path}")

    return df


def build_daily_dataset(
    data_dir: Path,
    use_cache: bool = False,
    rebuild_cache: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Return (labeled_daily_df, insiders_df, feature_columns)."""
    df = _load_or_build_daily_base(data_dir, use_cache=use_cache, rebuild_cache=rebuild_cache)

    base_cols = [
        "logon_count",
        "after_hours_logon_count",
        "unique_pcs",
        "usb_connect_count",
        "after_hours_usb_count",
        "file_access_count",
        "after_hours_file_count",
        "email_count",
        "external_email_count",
        "total_attachments",
        "total_email_size",
        "http_count",
        "after_hours_http_count",
        "job_search_http_count",
        "upload_cloud_http_count",
    ]
    for col in base_cols:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = df[col].fillna(0.0)

    df = df.sort_values(["user", "day"]).reset_index(drop=True)

    # Strictly past-looking rolling 7-day features + per-user historical baseline deviation
    rolling_cols = [
        "after_hours_logon_count",
        "usb_connect_count",
        "after_hours_usb_count",
        "file_access_count",
        "external_email_count",
        "after_hours_http_count",
        "job_search_http_count",
        "upload_cloud_http_count",
    ]
    feature_cols = list(base_cols)

    for col in rolling_cols:
        r_sum_col = f"{col}_roll{config.ROLLING_WINDOW_DAYS}d_sum"
        r_mean_col = f"{col}_roll{config.ROLLING_WINDOW_DAYS}d_mean"
        diff_col = f"{col}_vs_user_baseline"

        df[r_sum_col] = df.groupby("user")[col].transform(
            lambda s: s.rolling(config.ROLLING_WINDOW_DAYS, min_periods=1).sum()
        )
        df[r_mean_col] = df.groupby("user")[col].transform(
            lambda s: s.rolling(config.ROLLING_WINDOW_DAYS, min_periods=1).mean()
        )
        past_mean = (
            df.groupby("user")[col]
            .transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
            .fillna(0.0)
        )
        df[diff_col] = df[col] - past_mean
        feature_cols.extend([r_sum_col, r_mean_col, diff_col])

    # Merge static OCEAN psychometric traits
    psych = _load_psychometric(data_dir / "psychometric.csv")
    if not psych.empty:
        df = df.merge(psych, on="user", how="left")
        for trait in ["O", "C", "E", "A", "N"]:
            if trait in df.columns:
                feature_cols.append(trait)

    # Merge insider ground truth and create early-warning label
    insiders = load_insiders(data_dir)
    df = df.merge(insiders, on="user", how="left")

    days_before_attack = (df["attack_start_day"] - df["day"]).dt.days
    df["days_before_attack"] = days_before_attack

    # Drop days on or after attack start for malicious insiders (strictly early warning)
    df = df[df["days_before_attack"].isna() | (df["days_before_attack"] > 0)].copy()

    df["label"] = (
        df["days_before_attack"].notna()
        & (df["days_before_attack"] >= 1)
        & (df["days_before_attack"] <= config.PREDICTION_HORIZON_DAYS)
    ).astype(int)

    return df.reset_index(drop=True), insiders, feature_cols


def chronological_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split strictly by date into Train / Validation / Test."""
    unique_days = np.sort(df["day"].unique())
    n_days = len(unique_days)
    train_end_idx = int(n_days * config.TRAIN_RATIO)
    val_end_idx = int(n_days * (config.TRAIN_RATIO + config.VAL_RATIO))

    train_cutoff = unique_days[train_end_idx - 1]
    val_cutoff = unique_days[val_end_idx - 1]

    train_df = df[df["day"] <= train_cutoff].copy()
    val_df = df[(df["day"] > train_cutoff) & (df["day"] <= val_cutoff)].copy()
    test_df = df[df["day"] > val_cutoff].copy()
    return train_df, val_df, test_df