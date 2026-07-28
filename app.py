import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain.agents import create_agent

st.set_page_config(page_title="AI Powered Data Analyst Agent", layout="wide")

st.title("🤖 AI-Powered Data Analyst Agent")
st.write("Upload your dataset (CSV, XLSX) and let the AI agent automatically analyze it, generate charts for univariate, bivariate, and multivariate analysis, and chat with your data!")

GOOGLE_API_KEY = st.sidebar.text_input("Enter Google API Key", type="password")
GROQ_API_KEY = st.sidebar.text_input("Enter Groq API Key", type="password")

if GOOGLE_API_KEY and GROQ_API_KEY:
    gemini_llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=GOOGLE_API_KEY
    )

    groq_llm = ChatGroq(
        model="qwen-2.5-coder-32b-instruct",
        api_key=GROQ_API_KEY
    )

    def temp_tool():
        """This is just a dummy tool"""
        return "Hello world"

    agent = create_agent(
        model=gemini_llm,
        tools=[temp_tool]
    )

    uploaded_file = st.file_uploader("Upload your dataset file (CSV/Excel)", type=["csv", "xlsx", "xls"])

    if uploaded_file is not None:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        st.subheader("Automated EDA & Visualizations")
        if st.button("Run AI Data Analysis"):
            with st.spinner("Agent is analyzing the dataset and generating insights..."):
                try:
                    st.write("### Basic EDA Summary")
                    st.write(f"**Shape of Dataset:** {df.shape[0]} rows and {df.shape[1]} columns")
                    st.write("**Missing Values:**")
                    st.write(df.isnull().sum())
                    st.write("**Statistical Description:**")
                    st.write(df.describe())

                    st.subheader("Univariate Analysis")
                    num_cols = df.select_dtypes(include=np.number).columns
                    if len(num_cols) > 0:
                        fig, ax = plt.subplots(figsize=(8, 4))
                        sns.histplot(df[num_cols[0]], kde=True, ax=ax)
                        st.pyplot(fig)

                    st.subheader("Bivariate Analysis")
                    if len(num_cols) >= 2:
                        fig, ax = plt.subplots(figsize=(8, 4))
                        sns.scatterplot(data=df, x=num_cols[0], y=num_cols[1], ax=ax)
                        st.pyplot(fig)

                    st.subheader("Multivariate Analysis (Correlation Heatmap)")
                    if len(num_cols) > 1:
                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
                        st.pyplot(fig)

                    st.success("Analysis Completed Successfully!")
                except Exception as e:
                    st.error(f"Error during analysis: {e}")

        st.subheader("💬 Chat with your Data")
        user_query = st.text_input("Ask anything about your dataset:")
        if user_query:
            with st.spinner("AI is generating response..."):
                chat_prompt = f"Given this dataframe with columns {list(df.columns)} and sample {df.head(2).to_dict()}, answer this question: {user_query}"
                response = agent.invoke({'messages': [{'role': 'user', 'content': chat_prompt}]})
                
                try:
                    answer = response["messages"][-1].content
                    st.write(answer)
                except Exception as e:
                    st.write(str(response))
    else:
        st.info("Please upload a dataset to begin.")
else:
    st.warning("Please enter your Google API Key and Groq API Key in the sidebar to proceed.")
