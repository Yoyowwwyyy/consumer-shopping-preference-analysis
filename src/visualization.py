from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence

import os
import matplotlib.pyplot as plt
import pandas as pd


# * Constants

DEFAULT_PREFERENCE_ORDER = ["Store", "Online", "Hybrid"]

PREFERENCE_PALETTE = {
    "Store": "#4E79A7",
    "Online": "#F28E2B",
    "Hybrid": "#59A14F",
}

DISPLAY_NAME_MAP = {
    "shopping_preference": "Shopping Preference",
    "age": "Age",
    "monthly_income": "Monthly Income",
    "income_group": "Income Group",
    "gender": "Gender",
    "city_tier": "City Tier",
    "daily_internet_hours": "Daily Internet Hours",
    "smartphone_usage_years": "Smartphone Usage Years",
    "social_media_hours": "Social Media Hours",
    "online_payment_trust_score": "Online Payment Trust Score",
    "tech_savvy_score": "Tech Savvy Score",
    "discount_sensitivity": "Discount Sensitivity",
    "delivery_fee_sensitivity": "Delivery Fee Sensitivity",
    "free_return_importance": "Free Return Importance",
    "product_availability_online": "Product Availability Online",
    "impulse_buying_score": "Impulse Buying Score",
    "need_touch_feel_score": "Need for Touch and Feel",
    "brand_loyalty_score": "Brand Loyalty Score",
    "environmental_awareness": "Environmental Awareness",
    "time_pressure_level": "Time Pressure Level",
    "monthly_online_orders": "Monthly Online Orders",
    "monthly_store_visits": "Monthly Store Visits",
    "avg_online_spend": "Average Online Spend",
    "avg_store_spend": "Average Store Spend",
    "return_frequency": "Return Frequency",
    "avg_delivery_days": "Average Delivery Days",
    "total_shopping_activity": "Total Shopping Activity",
    "online_activity_share": "Online Activity Share",
    "store_activity_share": "Store Activity Share",
    "age_group": "Age Group",
    "income_band": "Income Band",
}



# * Main Functions

# Plot the overall shopping preference distribution using count bars + percent labels.
def plot_preference_distribution(
    summary_table: pd.DataFrame,
    count_col: str = "count",
    percentage_col: str = "percentage",
    title: str = "Shopping Preference Distribution",
    xlabel: str = "Preference",
    ylabel: str = "Count",
    figsize: tuple[int, int] = (7, 4),
):

    if not isinstance(summary_table, pd.DataFrame):
        raise TypeError("summary_table must be a pandas DataFrame.")

    required_cols = [count_col, percentage_col]
    missing_cols = [col for col in required_cols if col not in summary_table.columns]
    if missing_cols:
        raise ValueError(f"summary_table is missing required columns: {missing_cols}")

    plot_df = summary_table.copy()
    ordered_index = get_preference_order(plot_df.index)
    plot_df = plot_df.loc[ordered_index]

    colors = [get_preference_palette(plot_df.index)[idx] for idx in plot_df.index]

    fig, ax = plt.subplots(figsize=figsize)
    plot_df[count_col].plot(
        kind="bar",
        ax=ax,
        color=colors,
        edgecolor="black",
        linewidth=0.6
    )

    _format_axis(
        ax=ax,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        rotate_xticks=0,
        show_y_grid=True,
    )

    _add_percent_labels_to_bars(
        ax=ax,
        fmt="{:.2f}%"
    )

    # overwrite labels using percentage column, not count
    for text, pct in zip(ax.texts, plot_df[percentage_col].tolist()):
        text.set_text(f"{pct:.2f}%")

    fig.tight_layout()
    return fig, ax

# Plot one metric across shopping preference groups
def plot_single_metric_by_preference(
    summary: pd.Series,
    title: str,
    ylabel: str = "Average Value",
    xlabel: str = "Preference",
    figsize: tuple[int, int] = (6, 4),
    narrow_y: bool = True,
    padding_ratio: float = 0.20,
    min_visible_span_ratio: float = 0.05,
    show_values: bool = True,
    value_fmt: str = "{:.2f}",
):

    if not isinstance(summary, pd.Series):
        raise TypeError("summary must be a pandas Series.")

    plot_series = summary.dropna().copy()
    if plot_series.empty:
        raise ValueError("summary is empty after dropping missing values.")

    ordered_index = get_preference_order(plot_series.index)
    plot_series = plot_series.loc[ordered_index]

    colors = [get_preference_palette(plot_series.index)[idx] for idx in plot_series.index]

    fig, ax = plt.subplots(figsize=figsize)
    plot_series.plot(kind="bar", ax=ax, color=colors, edgecolor="black", linewidth=0.6)

    _format_axis(
        ax=ax,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        rotate_xticks=0,
        show_y_grid=True,
    )

    if narrow_y:
        _apply_narrow_y_axis(
            ax=ax,
            values=plot_series.values,
            padding_ratio=padding_ratio,
            min_visible_span_ratio=min_visible_span_ratio,
            force_zero_floor=False,
        )

    if show_values:
        _add_bar_value_labels(ax=ax, fmt=value_fmt)

    fig.tight_layout()
    return fig, ax


# Plot stacked proportion bars for preference composition by group
def plot_stacked_preference_mix(
    crosstab: pd.DataFrame,
    title: str,
    ylabel: str = "Proportion",
    xlabel: str = "",
    figsize: tuple[int, int] = (7, 4),
    legend_title: str = "Preference",
):

    if not isinstance(crosstab, pd.DataFrame):
        raise TypeError("crosstab must be a pandas DataFrame.")

    plot_df = crosstab.copy()
    if plot_df.empty:
        raise ValueError("crosstab is empty.")

    ordered_cols = get_preference_order(plot_df.columns)
    plot_df = plot_df[ordered_cols]

    palette = get_preference_palette(plot_df.columns)
    colors = [palette[col] for col in plot_df.columns]

    fig, ax = plt.subplots(figsize=figsize)
    plot_df.plot(
        kind="bar",
        stacked=True,
        ax=ax,
        color=colors,
        edgecolor="black",
        linewidth=0.4
    )

    _format_axis(
        ax=ax,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        rotate_xticks=0,
        show_y_grid=True,
    )

    ax.set_ylim(0, 1.0)
    ax.legend(title=legend_title, loc="upper right")

    fig.tight_layout()
    return fig, ax


def plot_activity_structure_chart(
    plot_data: pd.DataFrame,
    figsize_share: tuple[int, int] = (8, 5),
    figsize_total: tuple[int, int] = (7, 4),
    narrow_total_y: bool = True,
    total_padding_ratio: float = 0.20,
    total_min_visible_span_ratio: float = 0.05,
    show_values: bool = True,
):

    if not isinstance(plot_data, pd.DataFrame):
        raise TypeError("plot_data must be a pandas DataFrame.")

    required_cols = [
        "online_activity_share",
        "store_activity_share",
        "total_shopping_activity",
    ]
    missing_cols = [col for col in required_cols if col not in plot_data.columns]
    if missing_cols:
        raise ValueError(f"plot_data is missing required columns: {missing_cols}")

    ordered_index = get_preference_order(plot_data.index)
    plot_df = plot_data.loc[ordered_index].copy()

    # Chart 1: activity share comparison
    share_df = plot_df[["online_activity_share", "store_activity_share"]]

    fig1, ax1 = plt.subplots(figsize=figsize_share)
    share_df.plot(
        kind="bar",
        ax=ax1,
        color=["#76B7B2", "#E15759"],
        edgecolor="black",
        linewidth=0.6
    )

    _format_axis(
        ax=ax1,
        title="Activity Share by Shopping Preference",
        xlabel="Shopping Preference",
        ylabel="Average Share",
        rotate_xticks=0,
        show_y_grid=True,
    )
    ax1.legend(title="Metric", loc="upper right")

    if show_values:
        _add_bar_value_labels(ax=ax1, fmt="{:.2f}")

    fig1.tight_layout()

    # Chart 2: total activity comparison
    total_series = plot_df["total_shopping_activity"]
    colors = [get_preference_palette(total_series.index)[idx] for idx in total_series.index]

    fig2, ax2 = plt.subplots(figsize=figsize_total)
    total_series.plot(
        kind="bar",
        ax=ax2,
        color=colors,
        edgecolor="black",
        linewidth=0.6
    )

    _format_axis(
        ax=ax2,
        title="Total Shopping Activity by Shopping Preference",
        xlabel="Shopping Preference",
        ylabel="Average Total Activity",
        rotate_xticks=0,
        show_y_grid=True,
    )

    if narrow_total_y:
        _apply_narrow_y_axis(
            ax=ax2,
            values=total_series.values,
            padding_ratio=total_padding_ratio,
            min_visible_span_ratio=total_min_visible_span_ratio,
            force_zero_floor=False,
        )

    if show_values:
        _add_bar_value_labels(ax=ax2, fmt="{:.2f}")

    fig2.tight_layout()

    return (fig1, ax1), (fig2, ax2)


# Plot top-N ML feature importance values
def plot_ml_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 10,
    title: str = "Top Features Influencing Shopping Preference",
    figsize: tuple = (9, 5.5),
    save_fig: bool = False,
    filename: str = "ml_feature_importance_top10.png",
):

    required_cols = {"feature", "importance"}
    missing_cols = required_cols - set(importance_df.columns)
    if missing_cols:
        raise ValueError(
            f"importance_df is missing required columns: {missing_cols}"
        )

    plot_df = importance_df.head(top_n).copy()
    if plot_df.empty:
        raise ValueError("importance_df is empty.")

    plot_df = plot_df.iloc[::-1]  # reverse for barh top-to-bottom display

    set_plot_style()
    fig, ax = plt.subplots(figsize=figsize)

    # y labels
    if "feature_group" in plot_df.columns:
        display_labels = [
            f"{feature} ({group})"
            for feature, group in zip(plot_df["feature"], plot_df["feature_group"])
        ]
    else:
        display_labels = plot_df["feature"].tolist()

    y_positions = range(len(plot_df))

    ax.barh(
        y=y_positions,
        width=plot_df["importance"],
        color="#4E79A7",
        edgecolor="black",
        linewidth=0.6,
    )

    ax.set_yticks(list(y_positions))
    ax.set_yticklabels(display_labels)

    _format_axis(
        ax=ax,
        title=title,
        xlabel="Importance (average absolute coefficient)",
        ylabel="Feature",
        rotate_xticks=0,
        show_y_grid=False,
    )

    ax.grid(axis="x", linestyle="--", alpha=0.3)

    # value labels
    x_max = float(plot_df["importance"].max())
    label_offset = x_max * 0.01 if x_max > 0 else 0.01

    for i, value in enumerate(plot_df["importance"]):
        ax.text(
            x=value + label_offset,
            y=i,
            s=f"{value:.3f}",
            va="center",
            ha="left",
            fontsize=9,
        )

    plt.tight_layout()

    if save_fig:
        save_path = save_figure(
            fig=fig,
            filename=filename,
            subfolder="ml_analysis",
            close=False,
        )

    return fig, ax


# * Helpers

# Apply a unified matplotlib style for the whole project.
def set_plot_style() -> None:
    plt.style.use("default")
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#333333",
        "axes.labelcolor": "#222222",
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "legend.title_fontsize": 10,
        "grid.color": "#D9D9D9",
        "grid.linestyle": "--",
        "grid.linewidth": 0.7,
    })


# Convert raw column name into a more readable display name
def get_display_name(name: str) -> str:
    return DISPLAY_NAME_MAP.get(name, name.replace("_", " ").title())


# Return a stable preference order
def get_preference_order(
    values: Optional[Iterable[str]] = None,
    preferred_order: Optional[Sequence[str]] = None
) -> list[str]:

    if preferred_order is None:
        preferred_order = DEFAULT_PREFERENCE_ORDER

    if values is None:
        return list(preferred_order)

    values_list = list(pd.Index(values).astype(str))
    existing = [x for x in preferred_order if x in values_list]
    remaining = [x for x in values_list if x not in existing]
    return existing + remaining


# Return a palette dictionary for preference groups
def get_preference_palette(
    values: Optional[Iterable[str]] = None
) -> dict[str, str]:

    fallback_color = "#9E9E9E"

    if values is None:
        return dict(PREFERENCE_PALETTE)

    palette = {}
    for value in values:
        palette[str(value)] = PREFERENCE_PALETTE.get(str(value), fallback_color)
    return palette

# Save figure to figures/<subfolder>/filename and return the path
def save_figure(
    fig,
    filename: str,
    subfolder: str = "general",
    base_dir: str = "figures",
    close: bool = False,
) -> Path:

    save_dir = Path(base_dir) / subfolder
    save_dir.mkdir(parents=True, exist_ok=True)

    save_path = save_dir / filename
    fig.savefig(save_path, bbox_inches="tight")

    if close:
        plt.close(fig)

    return save_path


# Apply consistent axis formatting
def _format_axis(
    ax,
    title: str,
    xlabel: str,
    ylabel: str,
    rotate_xticks: int = 0,
    show_y_grid: bool = True
) -> None:

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=rotate_xticks)

    if show_y_grid:
        ax.grid(axis="y", alpha=0.7)
    else:
        ax.grid(False)

    ax.set_axisbelow(True)


# Add numeric labels above bars
def _add_bar_value_labels(
    ax,
    fmt: str = "{:.2f}",
    padding_ratio: float = 0.01
) -> None:

    y_min, y_max = ax.get_ylim()
    y_span = y_max - y_min
    label_offset = y_span * padding_ratio if y_span > 0 else 0.02

    for patch in ax.patches:
        height = patch.get_height()

        if pd.isna(height):
            continue

        x = patch.get_x() + patch.get_width() / 2
        ax.text(
            x,
            height + label_offset,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=9
        )

# Add percent labels above bars for already-percent-scaled values
def _add_percent_labels_to_bars(
    ax,
    fmt: str = "{:.1f}%",
    padding_ratio: float = 0.01
) -> None:

    y_min, y_max = ax.get_ylim()
    y_span = y_max - y_min
    label_offset = y_span * padding_ratio if y_span > 0 else 0.5

    for patch in ax.patches:
        height = patch.get_height()

        if pd.isna(height):
            continue

        x = patch.get_x() + patch.get_width() / 2
        ax.text(
            x,
            height + label_offset,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=9
        )



# Narrow the y-axis for subtle differences, but avoid dishonest over-zooming
def _apply_narrow_y_axis(
    ax,
    values: Sequence[float],
    padding_ratio: float = 0.20,
    min_visible_span_ratio: float = 0.05,
    force_zero_floor: bool = False,
) -> None:

    numeric_values = pd.Series(values, dtype="float64").dropna()

    if numeric_values.empty:
        return

    y_min = float(numeric_values.min())
    y_max = float(numeric_values.max())
    span = y_max - y_min

    if span == 0:
        base = max(abs(y_max), 1.0)
        span = base * min_visible_span_ratio

    reference_scale = max(abs(y_min), abs(y_max), 1.0)
    min_span = reference_scale * min_visible_span_ratio
    visible_span = max(span, min_span)

    padding = visible_span * padding_ratio
    lower = y_min - padding
    upper = y_max + padding

    if force_zero_floor:
        lower = max(0.0, lower)

    if lower == upper:
        upper = lower + 1.0

    ax.set_ylim(lower, upper)