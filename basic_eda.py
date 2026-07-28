
import pandas as pd


def perform_eda(df: pd.DataFrame):
    """Performs basic Exploratory Data Analysis (EDA) on a given pandas DataFrame.

    Parameters:
    df (pd.DataFrame): The input dataframe to analyze.
    """
    print("=" * 60)
    print("🚀 BASIC EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 60)

    # 1. Dataset Shape
    print("\n[1] DATASET SHAPE:")
    print(
        f"Rows: {df.shape[0]:,}\nColumns: {df.shape[1]:,}"
    )  # Formatted with commas

    # 2. Column Names and Data Types
    print("\n[2] COLUMNS & DATA TYPES:")
    dtypes_df = pd.DataFrame(
        {
            "Column Name": df.columns,
            "Data Type": df.dtypes.values,
            "Non-Null Count": df.notnull().sum().values,
        }
    )
    print(dtypes_df.to_string(index=False))

    # 3. Missing Values Summary
    print("\n[3] MISSING VALUES SUMMARY:")
    missing_count = df.isnull().sum()
    missing_pct = (df.isnull().mean() * 100).round(2)

    missing_df = pd.DataFrame(
        {"Missing Values": missing_count, "Percentage (%)": missing_pct}
    )
    # Filter to show only columns with missing values, or state if none
    missing_df = missing_df[missing_df["Missing Values"] > 0]

    if missing_df.empty:
        print("🎉 Great news! There are no missing values in this dataset.")
    else:
        print(missing_df.to_string())

    # 4. Duplicate Rows
    print("\n[4] DUPLICATE ROWS:")
    duplicates = df.duplicated().sum()
    print(
        f"Number of duplicate rows: {duplicates:,} ({(duplicates / len(df)) * 100:.2f}% of total data)"
    )

    # 5. Statistical Summary for Numerical Columns
    print("\n[5] STATISTICAL SUMMARY (Numerical Columns):")
    num_cols = df.select_dtypes(include=["number"])
    if not num_cols.empty:
        print(df.describe().T.to_string())
    else:
        print("No numerical columns found.")

    # 6. Statistical Summary for Categorical/Object Columns
    print("\n[6] STATISTICAL SUMMARY (Categorical Columns):")
    cat_cols = df.select_dtypes(include=["object", "category"])
    if not cat_cols.empty:
        print(df.describe(include=["object", "category"]).T.to_string())
    else:
        print("No categorical columns found.")

    print("\n" + "=" * 60)
    print("✨ EDA COMPLETED SUCCESSFULLY")
    print("=" * 60)
