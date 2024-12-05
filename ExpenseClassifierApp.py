import os
import streamlit as st
import pandas as pd
import vertexai
from vertexai.generative_models import GenerativeModel
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class ExpenseClassifierApp:
    def __init__(self):
        self.model = None

    def set_env(self):
        env_vars = ["GC_CRED"]
        for var in env_vars:
            os.environ[var] = str(st.secrets[var])
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.environ["GC_CRED"]
        # Initialize Vertex AI with the provided project ID
        vertexai.init(project=os.environ["PROJECT_ID"], location=os.environ["LOCATION"])
        self.model = GenerativeModel(os.environ["MODEL"])

    def run(self):
        self.set_env()
        self.display_title()
        self.get_budget_input()
        self.display_file_uploader()
        self.display_table()

    @staticmethod
    def display_title():
        st.title("Automated Expense Categorization")
        st.write(
            "Enter your finance data into the table below or upload a CSV file, and we will provide a breakdown of your finances!")

    def get_budget_input(self):
        st.write("### Enter Your Monthly Budget")
        self.budget = st.number_input("Enter your budget (in USD):", min_value=0.0, step=100.0)

    def display_file_uploader(self):
        # Allow users to upload a CSV file
        uploaded_file = st.file_uploader("Upload your CSV file with financial data", type=["csv"])
        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file)
                # Standardize column names to lowercase for consistency
                data.columns = data.columns.str.strip().str.lower()
                column_mapping = {"date": "Date", "description": "Description", "amount": "Amount"}
                data.rename(columns=lambda col: column_mapping.get(col, col), inplace=True)
                st.session_state["uploaded_data"] = data
            except Exception as e:
                st.error(f"An error occurred while reading the file: {e}")

    def display_table(self):
        # Use uploaded data if available, otherwise create an empty dataframe for user input
        if "uploaded_data" in st.session_state:
            data = st.session_state["uploaded_data"]
        else:
            st.write("Please enter your spending data below:")
            columns = ["Date", "Description", "Amount"]
            data = pd.DataFrame(columns=columns)

        edited_data = st.data_editor(data, use_container_width=True, num_rows="dynamic")

        if st.button("Categorize Expenses"):
            # Verify if the user entered any data into the table
            if not edited_data.empty and not edited_data.isnull().all().all():
                # Categorize the expenses automatically using Vertex AI
                try:
                    categorized_data = self.categorize_expenses(edited_data)
                    # Show the categorized data
                    st.write("### Categorized Data:")
                    st.write(categorized_data)

                    # Display graphs after financial advice
                    self.display_graphs(categorized_data)

                    # Provide financial advice
                    self.display_financial_advice(categorized_data)
                except Exception as e:
                    st.error(f"An error occurred: {e}")
            else:
                st.warning("Please enter valid financial data to categorize.")

    def categorize_expenses(self, data):
        # Ensure compatibility by using standardized column names
        data.columns = data.columns.str.strip().str.lower()

        # Iterate over the data and use Vertex AI to categorize each expense
        categories = []
        for _, row in data.iterrows():
            if pd.notnull(row["amount"]):
                prompt = f"Categorize the following expense: Description: '{row.get('description', '')}', Amount: '{row['amount']}'"
                try:
                    response = self.model.generate_content(prompt)
                    category = response.text.strip()
                except Exception as e:
                    category = f"Error: {e}"
            else:
                category = "Uncategorized"

            categories.append(category)

        data["category"] = categories
        return data

    def display_graphs(self, data):
        # Convert the Amount column to numeric
        data["amount"] = pd.to_numeric(data["amount"], errors='coerce')
        categorized_data = data.groupby("category").sum().reset_index()

        # Set color palette with lighter teal shades
        sns.set_palette("Blues")

        # Bar Chart
        st.write("### Bar Chart of Expenses by Category")
        plt.figure(figsize=(3.5, 2))
        sns.barplot(x="category", y="amount", data=categorized_data,
                    palette=sns.color_palette("Blues", len(categorized_data)))
        plt.xticks(rotation=45)
        plt.xlabel("Category")
        plt.ylabel("Total Amount")
        plt.title("Total Expenses by Category")
        st.pyplot(plt)
        plt.clf()

        # Pie Chart
        st.write("### Pie Chart of Expenses by Category")
        plt.figure(figsize=(3, 2.5))
        plt.pie(categorized_data["amount"], colors=sns.color_palette("Blues", len(categorized_data)),
                wedgeprops=dict(edgecolor='w'))
        plt.legend(categorized_data["category"], loc="best")
        plt.title("Expenses Breakdown by Category")
        st.pyplot(plt)
        plt.clf()

    def display_financial_advice(self, data):
        # Ensure compatibility by using standardized column names
        data.columns = data.columns.str.strip().str.lower()

        # Generate a detailed and personalized financial advice based on data patterns
        budget = float(self.budget)
        total_expenditure = pd.to_numeric(data['amount'], errors='coerce').sum()
        expense_breakdown = data.groupby('category')['amount'].sum().to_dict()

        prompt = (
            f"You are a financial advisor. Based on the following financial data, provide highly specific financial advice:\n"
            f"Budget: ${budget:.2f}\n"
            f"Total Expenditure: ${total_expenditure:.2f}\n"
            f"Expense Breakdown: {expense_breakdown}\n"
            f"Provide unique advice that takes into account the user's spending patterns, and provide actionable steps that are tailored to reducing spending where necessary and optimizing their budget."
        )

        try:
            response = self.model.generate_content(prompt)
            detailed_advice = response.text.strip()
            st.write("### Personalized Financial Advice")
            st.write(detailed_advice)
        except Exception as e:
            st.error(f"An error occurred while generating financial advice: {e}")


if __name__ == "__main__":
    app = ExpenseClassifierApp()
    app.run()
