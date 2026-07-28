
# ==========================================
# ADVANCED EXPLORATORY DATA ANALYSIS (EDA)
# ==========================================
# Note: This script assumes that 'df' is already loaded in your environment.
# All analysis steps are encapsulated inside the single function `eda_by_ai`.

import warnings
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Styling configurations
warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)


def eda_by_ai(df):
    """Performs an exhaustive, end-to-end Advanced Exploratory Data Analysis (EDA)

    directly on the already-loaded pandas DataFrame `df`.
    """

    # ==========================================
    # 1. INITIAL INSPECTION & DESCRIPTIVE STATS
    # ==========================================
    print("=" * 50)
    print("1. INITIAL INSPECTION & DESCRIPTIVE STATISTICS")
    print("=" * 50)

    print(f"\nDataset Shape: {df.shape}")
    print("\nMissing Values:\n", df.isnull().sum())
    print(f"\nDuplicate Rows: {df.duplicated().sum()}")

    print("\n--- Descriptive Statistics (Numerical) ---")
    print(df.describe())

    print("\n--- Descriptive Statistics (Categorical/Object) ---")
    try:
        print(df.describe(include=["O"]))
    except Exception as e:
        print(
            "Could not describe categorical columns with include=['O']: ", e
        )

    # ==========================================
    # 2. CORRELATION ANALYSIS
    # ==========================================
    print("\n" + "=" * 50)
    print("2. CORRELATION ANALYSIS")
    print("=" * 50)

    num_df = df.select_dtypes(include=[np.number])
    if not num_df.empty and num_df.shape[1] > 1:
        corr_matrix = num_df.corr(method="pearson")

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5
        )
        plt.title("Pearson Correlation Matrix of Numerical Features")
        plt.tight_layout()
        plt.show()
    else:
        print("Not enough numerical columns to compute a correlation matrix.")

    # ==========================================
    # 3. UNIVARIATE ANALYSIS
    # ==========================================
    print("\n" + "=" * 50)
    print("3. UNIVARIATE ANALYSIS")
    print("=" * 50)

    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns

    # Numerical distributions & outliers
    for col in num_cols:
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        sns.histplot(df[col], kde=True, ax=axes[0], color="skyblue")
        axes[0].set_title(f"Distribution & KDE of {col}")

        sns.boxplot(x=df[col], ax=axes[1], color="lightgreen")
        axes[1].set_title(f"Boxplot (Outlier Detection) of {col}")
        plt.tight_layout()
        plt.show()

    # Categorical counts
    for col in cat_cols:
        plt.figure(figsize=(8, 4))
        order = df[col].value_counts().index
        sns.countplot(
            data=df, x=col, order=order, palette="Set2", hue=col, legend=False
        )
        plt.title(f"Frequency Count of {col}")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    # ==========================================
    # 4. TIME SERIES ANALYSIS (Conditional)
    # ==========================================
    print("\n" + "=" * 50)
    print("4. TIME SERIES ANALYSIS")
    print("=" * 50)

    date_cols = [
        col
        for col in df.columns
        if "date" in col.lower() or "time" in col.lower()
    ]

    if date_cols and len(num_cols) > 0:
        date_col = date_cols[0]
        print(f"Datetime column identified: '{date_col}'")

        # Create a copy to avoid mutating the original dataframe inplace permanently if unintended
        temp_df = df.copy()
        temp_df[date_col] = pd.to_datetime(temp_df[date_col], errors="coerce")

        # Pick the first numeric column for time series metric plotting
        metric_col = num_cols[0]
        ts_df = (
            temp_df.set_index(date_col)
            .resample("D")[metric_col]
            .sum()
            .reset_index()
        )

        plt.figure(figsize=(12, 5))
        sns.lineplot(data=ts_df, x=date_col, y=metric_col, color="b")
        plt.title(f"Daily Trend Analysis for {metric_col}")
        plt.xlabel("Date")
        plt.ylabel(f"Total {metric_col}")
        plt.tight_layout()
        plt.show()
    else:
        print(
            "No date/time columns detected or no numerical columns available for time series analysis."
        )

    # ==========================================
    # 5. MULTIVARIATE ANALYSIS (Bar Plot with Hue)
    # ==========================================
    print("\n" + "=" * 50)
    print("5. MULTIVARIATE ANALYSIS (Bar Plot with Hue)")
    print("=" * 50)

    # Check if we have at least one numeric and two categorical columns to perform multivariate grouping
    if len(num_cols) > 0 and len(cat_cols) >= 2:
        val_col = num_cols[0]
        cat1 = cat_cols[0]
        cat2 = cat_cols[1]

        plt.figure(figsize=(10, 6))

        # Grouping to find mean of metric across the two categories
        grouped_df = (
            df.groupby([cat1, cat2])[val_col].mean().reset_index()
        )

        ax = sns.barplot(
            data=grouped_df,
            x=cat1,
            y=val_col,
            hue=cat2,
            palette="muted",
        )

        plt.title(
            f"Average {val_col} by {cat1} and {cat2}", fontsize=14
        )
        plt.xlabel(cat1, fontsize=12)
        plt.ylabel(f"Average {val_col}", fontsize=12)
        plt.legend(title=cat2, bbox_to_anchor=(1.05, 1), loc="upper left")

        # Data label annotations on bars
        for p in ax.patches:
            height = p.get_height()
            if not np.isnan(height) and height > 0:
                ax.annotate(
                    f"{height:.1f}",
                    (p.get_x() + p.get_width() / 2.0, height),
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    color="black",
                    xytext=(0, 3),
                    textcoords="offset points",
                )

        plt.tight_layout()
        plt.show()
    else:
        print(
            "Insufficient numerical or categorical columns found for multivariate grouped bar plot."
        )

    # ==========================================
    # 6. AUTOMATED SUMMARY
    # ==========================================
    print("\n" + "=" * 50)
    print("6. AUTOMATED SUMMARY")
    print("=" * 50)
    print(
        "EDA execution completed. Review the generated plots above for feature distributions,"
    )
    print(
        "potential outlier anomalies, correlations, and multi-category interactions."
    )
