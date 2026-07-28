
import os
import pandas as pd


def read_uploaded_file(uploaded_file):
    """Reads a file into a pandas DataFrame based on its file extension.

    Supports: CSV, Excel (.xlsx, .xls), JSON, Parquet, and Pickle.
    """
    # Get the file name and extension
    # If 'uploaded_file' is a file-like object (e.g., Streamlit UploadedFile),
    # use file_name. If it's a string path, use the path directly.
    file_name = (
        uploaded_file.name
        if hasattr(uploaded_file, "name")
        else str(uploaded_file)
    )
    ext = os.path.splitext(file_name)[1].lower()

    try:
        if ext == ".csv":
            # You can add common parameters like encoding='utf-8' if needed
            return pd.read_csv(uploaded_file)

        elif ext in [".xls", ".xlsx"]:
            return pd.read_excel(uploaded_file)

        elif ext == ".json":
            return pd.read_json(uploaded_file)

        elif ext == ".parquet":
            return pd.read_parquet(uploaded_file)

        elif ext in [".pkl", ".pickle"]:
            return pd.read_pickle(uploaded_file)

        elif ext == ".txt":
            # Assuming tab-separated or comma-separated; adjust sep as needed
            return pd.read_csv(uploaded_file, sep="\t")

        else:
            raise ValueError(f"Unsupported file extension: {ext}")

    except Exception as e:
        print(f"Error reading file {file_name}: {e}")
        raise e


# ==========================================
# EXAMPLE 1: Reading from a local file path
# ==========================================
# df = read_uploaded_file("data.csv")

# ==========================================
# EXAMPLE 2: Using with Streamlit (Web App)
# ==========================================
# import streamlit as st
# uploaded_file = st.file_uploader("Choose a file")
# if uploaded_file is not None:
#     df = read_uploaded_file(uploaded_file)
#     st.write(df.head())
