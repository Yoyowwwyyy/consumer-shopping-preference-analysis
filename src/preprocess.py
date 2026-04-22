import pandas as pd


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )
    return df


def convert_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    numeric_columns = [
        "age",
        "monthly_income",
        "daily_internet_hours",
        "smartphone_usage_years",
        "social_media_hours",
        "online_payment_trust_score",
        "tech_savvy_score",
        "monthly_online_orders",
        "monthly_store_visits",
        "avg_online_spend",
        "avg_store_spend",
        "discount_sensitivity",
        "return_frequency",
        "avg_delivery_days",
        "delivery_fee_sensitivity",
        "free_return_importance",
        "product_availability_online",
        "impulse_buying_score",
        "need_touch_feel_score",
        "brand_loyalty_score",
        "environmental_awareness",
        "time_pressure_level",
    ]

    categorical_columns = [
        "gender",
        "city_tier",
        "shopping_preference",
    ]

    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in categorical_columns:
        if col in df.columns:
            df[col] = df[col].astype("category")

    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    numeric_columns = df.select_dtypes(include=["number"]).columns
    categorical_columns = df.select_dtypes(include=["object", "category"]).columns

    for col in numeric_columns:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    for col in categorical_columns:
        if df[col].isna().any():
            df[col] = df[col].astype("object").fillna("Unknown").astype("category")

    return df

def handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    numeric_columns = df.select_dtypes(include=["number"]).columns

    for col in numeric_columns:
        lower = df[col].quantile(0.01)
        upper = df[col].quantile(0.99)
        df[col] = df[col].clip(lower=lower, upper=upper)

    return df


def basic_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Total shopping activity
    if {"monthly_online_orders", "monthly_store_visits"}.issubset(df.columns):
        df["total_shopping_activity"] = (
            df["monthly_online_orders"] + df["monthly_store_visits"]
        )

        # Avoid division by zero
        total_activity = df["total_shopping_activity"].replace(0, float("nan"))

        df["online_activity_share"] = (
            df["monthly_online_orders"].div(total_activity).fillna(0.0)
        )
        df["store_activity_share"] = (
            df["monthly_store_visits"].div(total_activity).fillna(0.0)
        )

    # Income group
    if "monthly_income" in df.columns:
        df["income_group"] = pd.qcut(
            df["monthly_income"],
            q=4,
            labels=["Low", "Lower-Middle", "Upper-Middle", "High"],
            duplicates="drop"
        ).astype("category")

    return df


def preprocess_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    df = clean_column_names(df)
    df = convert_types(df)
    df = handle_missing_values(df)
    df = handle_outliers(df)
    df = basic_feature_engineering(df)
    return df


# * Test Method

# Return a simple summary of the processed dataset
def get_data_summary(df: pd.DataFrame) -> dict:
    summary = {
        "shape": df.shape,
        "total_missing_values": int(df.isna().sum().sum()),
        "numeric_columns": df.select_dtypes(include=["number"]).columns.tolist(),
        "categorical_columns": df.select_dtypes(include=["object", "category"]).columns.tolist(),
    }
    return summary

