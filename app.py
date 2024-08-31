import streamlit as st
import pandas as pd
from pipelines.testing_pipeline import testing_pipeline
import os
import matplotlib.pyplot as plt

# Define the folder to save uploaded datasets
UPLOAD_FOLDER = 'uploaded_datasets'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Streamlit app title
st.title('Sales and Item Quantity Forecasting App')

# File uploader
uploaded_file = st.file_uploader("Upload your dataset (CSV file)", type=['csv'])

# If a file is uploaded
if uploaded_file is not None:
    # Save the uploaded file
    file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
    with open(file_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"File saved to: {file_path}")

    # Date input widgets
    start_date = st.date_input("Start Date")
    end_date = st.date_input("End Date")

    # When 'Run Forecast' button is clicked
    if st.button('Run Forecast'):
        # Hardcoded paths to preprocessor and model
        preprocessor_path = 'src/models/preprocessor.pkl'  
        model_path = 'src/models/final_model.pkl'

        # Run the forecasting pipeline
        try:
            forecasted_df, historical_df = testing_pipeline(
                data_path=file_path,
                preprocessor_path=preprocessor_path,
                model_path=model_path,
                start_date=start_date.strftime('%Y-%m-%d'),  # Convert to string
                end_date=end_date.strftime('%Y-%m-%d')   # Convert to string
            )

            # Display the forecasted dataframe
            st.subheader('Forecasted Data:')
            st.dataframe(forecasted_df)

            # Plotting net_sales
            fig, ax = plt.subplots(figsize=(12, 6))  # Set figure size
            ax.plot(historical_df['date_id'], historical_df['net_sales'], label='Historical Net Sales', color='blue', linestyle='-')
            ax.plot(forecasted_df['date_id'], forecasted_df['net_sales'], label='Forecasted Net Sales', color='orange', linestyle='-')
            ax.set_xlabel('Date')
            ax.set_ylabel('Net Sales')
            ax.legend()
            st.pyplot(fig)

            # Plotting item_qty
            fig, ax = plt.subplots(figsize=(12, 6))  # Set figure size
            ax.plot(historical_df['date_id'], historical_df['item_qty'], label='Historical Item Quantity', color='blue', linestyle='-')
            ax.plot(forecasted_df['date_id'], forecasted_df['item_qty'], label='Forecasted Item Quantity', color='orange', linestyle='-')
            ax.set_xlabel('Date')
            ax.set_ylabel('Item Quantity')
            ax.legend()
            st.pyplot(fig)

        except Exception as e:
            st.error(f"An error occurred: {e}")
