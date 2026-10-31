"""Shared constants and preprocessing functions for the Delhivery Delay Pipeline.
Reused across Part A, Part B, Part C, Part D, and production inference.
"""
import numpy as np
import pandas as pd

PROMISED_HOURS = {"Express": 36, "Standard": 60, "Economy": 96}

FEATURES = [
    "promised_hours", "distance_km", "parcel_weight_kg", "log_declared_value", "n_stops",
    "rain_flag", "sale_event_flag", "is_cod", "is_weekend", "dispatch_hour",
    "hub_tier", "hub_daily_capacity",
    "label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no",
]

NOISE_COLUMNS = ["label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no"]

MEDIAN_COLUMNS = ["distance_km", "parcel_weight_kg", "declared_value_inr"]


def clean_shipments(raw: pd.DataFrame, medians: dict = None) -> tuple[pd.DataFrame, dict]:
    """Clean a raw shipments export DataFrame.
    
    Steps applied in order:
    1. Drop exact duplicate rows.
    2. Standardize service_type: strip whitespace and convert to title case.
    3. Fill missing payment_mode with 'Unknown'.
    4. Clean declared_value (remove rupee sign and commas) -> numeric declared_value_inr, drop text column.
    5. Convert dispatch_time to datetime.
    6. Treat parcel_weight_kg > 100 as scanner-fault sentinel -> set to NaN.
    7. Fill missing values in distance_km, parcel_weight_kg, and declared_value_inr with medians.
       If medians is None, compute from the non-missing values; otherwise use passed medians.
       
    Returns:
        tuple: (clean_df, medians_dict)
    """
    df = raw.copy()
    
    # 1. Drop exact duplicate rows
    df = df.drop_duplicates().reset_index(drop=True)
    
    # 2. service_type: strip and title case
    df["service_type"] = df["service_type"].astype(str).str.strip().str.title()
    
    # 3. payment_mode: replace missing values with "Unknown"
    df["payment_mode"] = df["payment_mode"].fillna("Unknown")
    
    # 4. declared_value: remove rupee sign and commas, convert to numeric, drop text column
    cleaned_value = (
        df["declared_value"].astype("string")
        .str.replace("\u20b9", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
    )
    df["declared_value_inr"] = pd.to_numeric(cleaned_value, errors="coerce").astype(float)
    df = df.drop(columns=["declared_value"])
    
    # 5. dispatch_time: convert to datetime
    df["dispatch_time"] = pd.to_datetime(df["dispatch_time"])
    
    # 6. parcel_weight_kg: values > 100 are scanner-fault sentinels -> treat as missing
    df["parcel_weight_kg"] = df["parcel_weight_kg"].astype(float)
    df.loc[df["parcel_weight_kg"] > 100, "parcel_weight_kg"] = np.nan
    
    # 7. Fill missing distance_km, parcel_weight_kg, declared_value_inr with median
    if medians is None:
        medians = {col: float(df[col].median()) for col in MEDIAN_COLUMNS}
    
    for col in MEDIAN_COLUMNS:
        df[col] = df[col].fillna(medians[col])
        
    return df, medians


def build_features(clean: pd.DataFrame, hubs: pd.DataFrame) -> pd.DataFrame:
    """Attach hub columns and derive model features.
    
    Never uses delivery_hours so it works for unlabelled dispatches.
    
    Returns:
        pd.DataFrame with hub attributes and engineered features.
    """
    hub_cols = ["hub_id", "hub_city", "region", "hub_tier", "hub_daily_capacity"]
    out = clean.merge(hubs[hub_cols], on="hub_id", how="left")
    
    out["promised_hours"] = out["service_type"].map(PROMISED_HOURS).astype(int)
    out["is_cod"] = (out["payment_mode"] == "COD").astype(int)
    out["is_weekend"] = (out["dispatch_time"].dt.dayofweek >= 5).astype(int)
    out["dispatch_hour"] = out["dispatch_time"].dt.hour.astype(int)
    out["log_declared_value"] = np.log(out["declared_value_inr"])
    
    return out
