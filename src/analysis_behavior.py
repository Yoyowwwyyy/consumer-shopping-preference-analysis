import pandas as pd
import matplotlib.pyplot as plt


# * Main Functions

# Run the full behavior analysis module.
def run_behavior_analysis(
    df: pd.DataFrame,
    target_col: str = "shopping_preference",
    plot: bool = True,
    round_digits: int = 3
) -> dict:
    
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

    variable_groups = get_behavior_variable_groups(df)

    all_behavior_cols = (
        variable_groups["core_behavior"]
        + variable_groups["behavior_structure"]
        + variable_groups["behavior_context"]
    )

    overall_summary = summarize_behavior_by_preference(
        df=df,
        target_col=target_col,
        behavior_cols=all_behavior_cols,
        round_digits=round_digits
    )

    activity_structure = analyze_activity_structure(
        df=df,
        target_col=target_col,
        plot=plot,
        round_digits=round_digits
    )

    hybrid_position = analyze_hybrid_position(
        df=df,
        target_col=target_col,
        behavior_cols=variable_groups["core_behavior"] + variable_groups["behavior_structure"],
        round_digits=round_digits
    )

    return {
        "variable_groups": variable_groups,
        "overall_summary": overall_summary,
        "activity_structure": activity_structure,
        "hybrid_position": hybrid_position,
    }


# Analyze activity structure across shopping preference groups.
def analyze_activity_structure(
    df: pd.DataFrame,
    target_col: str = "shopping_preference",
    plot: bool = True,
    round_digits: int = 3
) -> dict:

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

    required_cols = [
        "total_shopping_activity",
        "online_activity_share",
        "store_activity_share",
    ]

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"These required activity structure columns are missing: {missing_cols}"
        )

    activity_df = df[[target_col] + required_cols].copy()

    summary_table = (
        activity_df.groupby(target_col)[required_cols]
        .agg(["mean", "median"])
        .round(round_digits)
    )

    preferred_order = ["Store", "Online", "Hybrid"]
    existing_order = [x for x in preferred_order if x in summary_table.index]
    remaining_order = [x for x in summary_table.index if x not in existing_order]
    summary_table = summary_table.loc[existing_order + remaining_order]

    plot_data = (
        activity_df.groupby(target_col)[required_cols]
        .mean()
        .round(round_digits)
    )
    plot_data = plot_data.loc[existing_order + remaining_order]

    key_patterns = []

    if "store_activity_share" in plot_data.columns:
        max_store_group = plot_data["store_activity_share"].idxmax()
        key_patterns.append(
            f"{max_store_group} shows the highest average store activity share."
        )

    if "online_activity_share" in plot_data.columns:
        max_online_group = plot_data["online_activity_share"].idxmax()
        key_patterns.append(
            f"{max_online_group} shows the highest average online activity share."
        )

    if all(group in plot_data.index for group in ["Store", "Online", "Hybrid"]):
        hybrid_online = plot_data.loc["Hybrid", "online_activity_share"]
        store_online = plot_data.loc["Store", "online_activity_share"]
        online_online = plot_data.loc["Online", "online_activity_share"]

        hybrid_store = plot_data.loc["Hybrid", "store_activity_share"]
        store_store = plot_data.loc["Store", "store_activity_share"]
        online_store = plot_data.loc["Online", "store_activity_share"]

        if min(store_online, online_online) <= hybrid_online <= max(store_online, online_online):
            key_patterns.append(
                "Hybrid appears to sit between Store and Online in online activity share."
            )

        if min(store_store, online_store) <= hybrid_store <= max(store_store, online_store):
            key_patterns.append(
                "Hybrid appears to sit between Store and Online in store activity share."
            )

    if plot:
        # Chart 1: activity share comparison
        share_cols = ["online_activity_share", "store_activity_share"]
        share_plot = plot_data[share_cols]

        ax = share_plot.plot(kind="bar", figsize=(8, 5))
        ax.set_title("Activity Share by Shopping Preference")
        ax.set_xlabel("Shopping Preference")
        ax.set_ylabel("Average Share")
        ax.legend(title="Metric", loc="upper right")
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.show()

        # Chart 2: total activity comparison
        ax = plot_data[["total_shopping_activity"]].plot(kind="bar", figsize=(7, 4))

        ax.set_title("Total Shopping Activity by Shopping Preference")
        ax.set_xlabel("Shopping Preference")
        ax.set_ylabel("Average Total Activity")

        min_val = plot_data["total_shopping_activity"].min()
        max_val = plot_data["total_shopping_activity"].max()
        ax.set_ylim(min_val * 0.95, max_val * 1.05)

        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.show()

    return {
        "summary_table": summary_table,
        "key_patterns": key_patterns,
        "plot_data": plot_data,
    }


# Analyze whether Hybrid is closer to Store or Online for each behavior variable
def analyze_hybrid_position(
    df,
    target_col: str = "shopping_preference",
    behavior_cols: list[str] | None = None,
    round_digits: int = 3
) -> pd.DataFrame:

    if target_col not in df.columns:
        raise ValueError(f"{target_col} not found in DataFrame.")

    # Default Value
    if behavior_cols is None:
        groups = get_behavior_variable_groups(df)
        behavior_cols = (
            groups["core_behavior"]
            + groups["behavior_structure"]
        )

    # Group Mean
    mean_table = (
        df.groupby(target_col)[behavior_cols]
        .mean()
        .round(round_digits)
    )

    required_groups = ["Store", "Online", "Hybrid"]
    for g in required_groups:
        if g not in mean_table.index:
            raise ValueError(f"Group '{g}' not found in data.")

    results = []

    for col in behavior_cols:
        store_val = mean_table.loc["Store", col]
        online_val = mean_table.loc["Online", col]
        hybrid_val = mean_table.loc["Hybrid", col]

        dist_store = abs(hybrid_val - store_val)
        dist_online = abs(online_val - hybrid_val)

        # position
        if abs(store_val - online_val) < 1e-6:
            position = "indistinguishable"
        elif min(store_val, online_val) <= hybrid_val <= max(store_val, online_val):
            position = "in-between"
        else:
            position = "outside-range"

        # check closer
        if dist_store < dist_online:
            closer_to = "Store"
        elif dist_online < dist_store:
            closer_to = "Online"
        else:
            closer_to = "Equal"

        results.append({
            "variable": col,
            "store_mean": round(store_val, round_digits),
            "hybrid_mean": round(hybrid_val, round_digits),
            "online_mean": round(online_val, round_digits),
            "dist_to_store": round(dist_store, round_digits),
            "dist_to_online": round(dist_online, round_digits),
            "position": position,
            "closer_to": closer_to
        })

    result_df = pd.DataFrame(results)

    return result_df


# * Auxiliary Methods

# Define and validate variable groups for the behavior analysis module.
def get_behavior_variable_groups(df: pd.DataFrame) -> dict:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    target_col = "shopping_preference"

    core_behavior = [
        "monthly_online_orders",
        "monthly_store_visits",
        "avg_online_spend",
        "avg_store_spend",
        "return_frequency",
        "avg_delivery_days",
    ]

    behavior_structure = [
        "total_shopping_activity",
        "online_activity_share",
        "store_activity_share",
    ]

    behavior_context = [
        "time_pressure_level",
    ]

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

    def existing_cols(columns: list[str]) -> list[str]:
        return [col for col in columns if col in df.columns]

    groups = {
        "target": target_col,
        "core_behavior": existing_cols(core_behavior),
        "behavior_structure": existing_cols(behavior_structure),
        "behavior_context": existing_cols(behavior_context),
    }

    return groups


# Summarize behavior variables by shopping preference group.
def summarize_behavior_by_preference(
    df: pd.DataFrame,
    target_col: str = "shopping_preference",
    behavior_cols: list[str] | None = None,
    round_digits: int = 2
) -> pd.DataFrame:

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

    if behavior_cols is None:
        groups = get_behavior_variable_groups(df)
        behavior_cols = (
            groups["core_behavior"]
            + groups["behavior_structure"]
            + groups["behavior_context"]
        )

    missing_cols = [col for col in behavior_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"These behavior columns are missing: {missing_cols}")

    if len(behavior_cols) == 0:
        raise ValueError("No valid behavior columns were provided.")

    agg_dict = {col: ["mean", "median"] for col in behavior_cols}

    summary = (
        df.groupby(target_col)
        .agg(agg_dict)
        .round(round_digits)
    )

    # Add group size
    counts = df.groupby(target_col).size().rename(("group_info", "count"))
    summary[("group_info", "count")] = counts

    # Reorder so count appears first
    ordered_cols = [("group_info", "count")] + [
        col for col in summary.columns if col != ("group_info", "count")
    ]
    summary = summary[ordered_cols]

    # Sort rows in a stable order if possible
    preferred_order = ["Store", "Online", "Hybrid"]
    existing_order = [x for x in preferred_order if x in summary.index]
    remaining_order = [x for x in summary.index if x not in existing_order]
    summary = summary.loc[existing_order + remaining_order]

    return summary