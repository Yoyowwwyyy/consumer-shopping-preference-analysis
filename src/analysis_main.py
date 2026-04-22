from __future__ import annotations

from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import pandas as pd

from src.visualization import (
    get_display_name,
    set_plot_style,
    plot_preference_distribution,
    plot_single_metric_by_preference,
    plot_stacked_preference_mix,
    save_figure,
)


# * Variable

TARGET_COL = "shopping_preference"

DEMOGRAPHIC_COLS = [
    "age",
    "monthly_income",
    "income_group",
    "gender",
    "city_tier",
]

DIGITAL_COLS = [
    "daily_internet_hours",
    "smartphone_usage_years",
    "social_media_hours",
    "online_payment_trust_score",
    "tech_savvy_score",
]

PSYCHOLOGICAL_COLS = [
    "discount_sensitivity",
    "delivery_fee_sensitivity",
    "free_return_importance",
    "product_availability_online",
    "impulse_buying_score",
    "need_touch_feel_score",
    "brand_loyalty_score",
    "environmental_awareness",
    "time_pressure_level",
]

BEHAVIOR_COLS = [
    "monthly_online_orders",
    "monthly_store_visits",
    "avg_online_spend",
    "avg_store_spend",
    "return_frequency",
    "avg_delivery_days",
    "total_shopping_activity",
    "online_activity_share",
    "store_activity_share",
]


# * Main Functions

# * Analyze the overall distribution of shopping_preference.
def analyze_preference_overview(
    df: pd.DataFrame,
    target_col: str = "shopping_preference",
    plot: bool = True,
    save_fig: bool = False
) -> dict:

    if target_col not in df.columns:
        raise ValueError(f"{target_col} not found in dataframe.")

    count_table = df[target_col].value_counts().rename("count")

    percentage_table = (
        df[target_col]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
        .rename("percentage")
    )

    summary_table = pd.concat([count_table, percentage_table], axis=1)

    ordered_index = [x for x in ["Store", "Online", "Hybrid"] if x in summary_table.index]
    remaining_index = [x for x in summary_table.index if x not in ordered_index]
    summary_table = summary_table.loc[ordered_index + remaining_index]

    max_count = summary_table["count"].max()
    min_count = summary_table["count"].min()
    imbalance_ratio = round(max_count / min_count, 2) if min_count > 0 else None

    saved_path = None

    if plot:
        set_plot_style()
        fig, ax = plot_preference_distribution(
            summary_table=summary_table,
            count_col="count",
            percentage_col="percentage",
            title="Shopping Preference Distribution",
            xlabel="Preference",
            ylabel="Count",
            figsize=(7, 4),
        )

        if save_fig:
            saved_path = save_figure(
                fig=fig,
                filename="shopping_preference_distribution.png",
                subfolder="main_analysis"
            )

        plt.show()
        plt.close(fig)

    most_common = summary_table["count"].idxmax()
    least_common = summary_table["count"].idxmin()

    insights = f"""
The distribution of shopping preference shows a clear imbalance:

- {most_common} is the dominant shopping mode by a large margin.
- {least_common} is the least preferred option, accounting for only {summary_table.loc[least_common, 'percentage']:.2f}% of consumers.
- This indicates that most consumers in the dataset still rely primarily on physical stores.
- The key analytical question is therefore not whether Hybrid is a mainstream pattern, but what differentiates the smaller Online and Hybrid groups from the dominant Store group.
"""

    return {
        "count_table": count_table,
        "percentage_table": percentage_table,
        "summary_table": summary_table,
        "imbalance_ratio": imbalance_ratio,
        "saved_path": saved_path,
        "insights": insights.strip()
    }


# * Analyze how demographic variables influence shopping preference
def analyze_demographic_drivers(
    df: pd.DataFrame,
    target_col: str = "shopping_preference",
    demographic_cols: list = None,
    plot: bool = True,
    save_fig: bool = False
) -> dict:

    if demographic_cols is None:
        raise ValueError("demographic_cols must be provided.")

    tables = {}
    driver_strength = {}
    saved_paths = {}

    for col in demographic_cols:
        if col not in df.columns:
            continue

        plot_col = col
        temp_df = df.copy()

        if col == "age":
            temp_df["age_group"] = pd.cut(
                temp_df["age"],
                bins=[17, 29, 39, 49, 59, 100],
                labels=["18-29", "30-39", "40-49", "50-59", "60+"]
            )
            plot_col = "age_group"

        elif col == "monthly_income":
            temp_df["income_band"] = pd.qcut(
                temp_df["monthly_income"],
                q=4,
                labels=["Low", "Lower-Mid", "Upper-Mid", "High"],
                duplicates="drop"
            )
            plot_col = "income_band"

        crosstab = pd.crosstab(
            temp_df[plot_col],
            temp_df[target_col],
            normalize="index"
        ).round(3)

        tables[col] = crosstab

        if "Store" in crosstab.columns:
            store_share = crosstab["Store"]
            deviation = (1 - store_share).mean()
            driver_strength[col] = deviation

        if plot:
            set_plot_style()
            display_plot_col = get_display_name(plot_col)

            fig, ax = plot_stacked_preference_mix(
                crosstab=crosstab,
                title=f"{display_plot_col} vs Shopping Preference",
                ylabel="Proportion",
                xlabel=display_plot_col,
                figsize=(7, 4),
                legend_title="Preference",
            )

            if save_fig:
                saved_paths[col] = save_figure(
                    fig=fig,
                    filename=f"{plot_col}_vs_shopping_preference.png",
                    subfolder="main_analysis"
                )

            plt.show()
            plt.close(fig)

    driver_summary = (
        pd.Series(driver_strength)
        .sort_values(ascending=False)
        .rename("deviation_from_store")
    )

    top_driver = driver_summary.index[0]
    top_value = driver_summary.iloc[0]

    if top_value < 0.05:
        top_driver = None

    if top_driver is None:
        insights = """
Demographic analysis shows that shopping preference is highly consistent across all demographic groups.

There is no strong evidence that age, income, gender, or city tier significantly influence shopping preference.

This suggests that demographic factors alone are insufficient to explain why consumers choose Store over Online or Hybrid.

Therefore, deeper drivers such as digital capability and psychological factors should be explored next.
"""
    else:
        insights = f"""
Demographic analysis reveals that:

- Most groups are still dominated by Store preference.
- The strongest demographic driver is: {top_driver}.

However, the overall differences remain relatively small, suggesting limited explanatory power.
"""

    return {
        "tables": tables,
        "driver_summary": driver_summary,
        "saved_paths": saved_paths,
        "insights": insights.strip()
    }


# Analyze how digital capability variables influence shopping preference.
def analyze_digital_drivers(
    df: pd.DataFrame,
    target_col: str = "shopping_preference",
    digital_cols: list = None,
    plot: bool = True,
    save_fig: bool = False
) -> dict:

    if digital_cols is None:
        raise ValueError("digital_cols must be provided.")

    tables = {}
    driver_strength = {}
    saved_paths = {}

    for col in digital_cols:
        if col not in df.columns:
            continue

        summary = (
            df.groupby(target_col)[col]
            .mean()
            .round(2)
        )

        tables[col] = summary

        strength = summary.max() - summary.min()
        driver_strength[col] = strength

        if plot:
            set_plot_style()
            display_col = get_display_name(col)

            fig, ax = plot_single_metric_by_preference(
                summary=summary,
                title=f"{display_col} by Shopping Preference",
                ylabel="Average Value",
                xlabel="Preference",
                narrow_y=True,
                padding_ratio=0.20,
                min_visible_span_ratio=0.05,
                show_values=True,
                value_fmt="{:.2f}",
            )

            if save_fig:
                saved_paths[col] = save_figure(
                    fig=fig,
                    filename=f"{col}_by_shopping_preference.png",
                    subfolder="main_analysis"
                )

            plt.show()
            plt.close(fig)

    driver_summary = (
        pd.Series(driver_strength)
        .sort_values(ascending=False)
        .rename("driver_strength")
    )

    top_driver = driver_summary.index[0]
    top_value = driver_summary.iloc[0]

    if top_value < 0.5:
        insights = f"""
Digital capability variables show only modest differences across shopping preferences.

- The strongest digital factor is: {top_driver}, but the overall variation remains limited.

This suggests that while digital capability may influence shopping behavior to some extent,
it is not a dominant driver in this dataset.

Therefore, other factors — particularly psychological preferences — are likely to play a more significant role.
"""
    else:
        insights = f"""
Digital analysis reveals meaningful differences across shopping preferences.

- The strongest digital driver is: {top_driver}.

Consumers with higher {top_driver} are more likely to shift away from Store towards Online or Hybrid shopping.

This indicates that digital capability plays a significant role in shaping shopping behavior.
"""

    return {
        "tables": tables,
        "driver_summary": driver_summary,
        "saved_paths": saved_paths,
        "insights": insights.strip()
    }


# Analyze psychological drivers of shopping preference.
def analyze_psychological_drivers(
    df: pd.DataFrame,
    target_col: str = "shopping_preference",
    psych_cols: list = None,
    plot: bool = True,
    save_fig: bool = False
) -> dict:

    if psych_cols is None:
        raise ValueError("psych_cols must be provided.")

    tables = {}
    driver_strength = {}
    saved_paths = {}

    for col in psych_cols:
        if col not in df.columns:
            continue

        summary = (
            df.groupby(target_col)[col]
            .mean()
            .round(2)
        )

        tables[col] = summary

        strength = summary.max() - summary.min()
        driver_strength[col] = strength

        if plot:
            set_plot_style()
            display_col = get_display_name(col)

            fig, ax = plot_single_metric_by_preference(
                summary=summary,
                title=f"{display_col} by Shopping Preference",
                ylabel="Average Value",
                xlabel="Preference",
                narrow_y=True,
                padding_ratio=0.20,
                min_visible_span_ratio=0.05,
                show_values=True,
                value_fmt="{:.2f}",
            )

            if save_fig:
                saved_paths[col] = save_figure(
                    fig=fig,
                    filename=f"{col}_by_shopping_preference.png",
                    subfolder="main_analysis"
                )

            plt.show()
            plt.close(fig)

    driver_summary = (
        pd.Series(driver_strength)
        .sort_values(ascending=False)
        .rename("driver_strength")
    )

    top_driver = driver_summary.index[0]
    top_value = driver_summary.iloc[0]

    if top_value < 0.5:
        insights = f"""
Psychological variables show limited variation across shopping preferences.

- The strongest factor is {top_driver}, but overall differences are modest.

This suggests that psychological factors are not the dominant drivers in this dataset.
"""
    else:
        insights = f"""
Psychological analysis reveals strong differences across shopping preferences.

- The most influential factor is: {top_driver}.

This indicates that consumer preferences are strongly shaped by underlying motivations and perceptions.

In particular, differences in {top_driver} play a key role in determining whether consumers prefer Store, Online, or Hybrid shopping.
"""

    return {
        "tables": tables,
        "driver_summary": driver_summary,
        "saved_paths": saved_paths,
        "insights": insights.strip()
    }


# * Auxiliary Methods

# Return only columns that actually exist in df.
def get_existing_columns(df: pd.DataFrame, columns: List[str]) -> List[str]:
    existing = set(df.columns)
    return [col for col in columns if col in existing]


# Return the finalized variable groups for main analysis
def get_variable_groups(df: pd.DataFrame) -> Dict[str, List[str]]:
    groups = {
        "target": [TARGET_COL] if TARGET_COL in df.columns else [],
        "demographic": get_existing_columns(df, DEMOGRAPHIC_COLS),
        "digital": get_existing_columns(df, DIGITAL_COLS),
        "psychological": get_existing_columns(df, PSYCHOLOGICAL_COLS),
        "behavior": get_existing_columns(df, BEHAVIOR_COLS),
    }
    return groups


# Print grouped variables for quick inspection in notebook/debugging
def print_variable_groups(groups: Dict[str, List[str]]) -> None:
    for group_name, cols in groups.items():
        print(f"\n[{group_name.upper()}] ({len(cols)})")
        for col in cols:
            print(f" - {col}")


# Return all columns used by the main analysis module.
def get_all_main_analysis_columns(df: pd.DataFrame) -> List[str]:
    groups = get_variable_groups(df)

    all_cols = []
    for cols in groups.values():
        all_cols.extend(cols)

    # remove duplicates while preserving order
    seen = set()
    ordered_cols = []
    for col in all_cols:
        if col not in seen:
            seen.add(col)
            ordered_cols.append(col)

    return ordered_cols