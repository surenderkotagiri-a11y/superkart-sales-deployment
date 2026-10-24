import streamlit as st
import pandas as pd
import requests
import io

# Configure page layouts and browser titles
st.set_page_config(page_title="SuperKart Sales Predictor", layout="wide", initial_sidebar_state="expanded")

# Set the backend URL. Because we link our backend container to the network,
# the hostname matches our backend service name inside docker-compose ('superkart_backend')
BACKEND_API_URL = "http://superkart_backend:7860"

st.title("🛒 SuperKart Retail Outlet Sales Forecaster")
st.markdown("""
Welcome to the official SuperKart Sales Forecasting portal. Use this web-application to predict target store sales volume
using advanced Machine Learning pipeline parameters.
""")

tab1, tab2 = st.tabs(["🔮 Real-Time Forecast (Single Product)", "📂 Bulk Batch Operations"])

with tab1:
    st.subheader("Enter Product and Store Configuration Specs")

    # Split inputs into two organized columns
    col1, col2 = st.columns(2)

    with col1:
        product_id_char = st.selectbox("Product ID Classification Group (Prefix)", ["FD", "NC", "DR"], help="FD: Food, DR: Drink, NC: Non-Consumable")
        product_weight = st.slider("Product Weight (kg)", min_value=4.0, max_value=22.0, value=12.0, step=0.1)
        product_sugar = st.selectbox("Sugar Content Level", ["Low Sugar", "Regular", "No Sugar"])
        product_mrp = st.number_input("Product Maximum Retail Price (MRP in USD)", min_value=30.0, max_value=270.0, value=140.0, step=1.0)
        product_allocated_area = st.slider("Product Allocated Display Area Ratio", min_value=0.0, max_value=0.33, value=0.06, step=0.01)

    with col2:
        product_type_category = st.selectbox("Consolidated Product Class Type", ["Perishables", "Non Perishables", "Others"])
        store_size = st.selectbox("Store Size Category", ["Medium", "Small", "High"])
        store_location = st.selectbox("Store Location (City Tier Rank)", ["Tier 1", "Tier 2", "Tier 3"])
        store_type = st.selectbox("Outlet Operational Form Factor (Store Type)", ["Supermarket Type1", "Supermarket Type2", "Departmental Store", "Food Mart"])
        store_age = st.slider("Store Operational Lifetime Age (Years)", min_value=15, max_value=37, value=25)

    # Trigger single prediction request
    if st.button("Run Predictive Forecast Pipeline", type="primary"):
        # Construct JSON payload matching preprocessing structure
        payload = {
            "Product_Weight": float(product_weight),
            "Product_Sugar_Content": product_sugar,
            "Product_Allocated_Area": float(product_allocated_area),
            "Product_MRP": float(product_mrp),
            "Store_Size": store_size,
            "Store_Location_City_Type": store_location,
            "Store_Type": store_type,
            "Product_Id_char": product_id_char,
            "Store_Age_Years": int(store_age),
            "Product_Type_Category": product_type_category
        }

        try:
            with st.spinner("Communicating with backend inference engine..."):
                response = requests.post(f"{BACKEND_API_URL}/v1/predict", json=payload, timeout=10)

            if response.status_code == 200:
                prediction_val = response.json().get("predicted_sales")
                st.balloons()
                st.success(f"### Predicted Sales: **${prediction_val:,.2f}**")
            else:
                st.error(f"Backend API returned an error ({response.status_code}): {response.text}")

        except requests.exceptions.RequestException as req_err:
            st.error(f"Network connectivity error! Unable to reach backend container: {str(req_err)}")

with tab2:
    st.subheader("Bulk Predictive CSV Operations")
    st.markdown("Upload a standard SuperKart CSV dataset containing feature columns to compute batch predictions.")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        # Verify file structure
        try:
            input_df = pd.read_csv(uploaded_file)
            st.write("**Preview of Uploaded Data:**", input_df.head(5))

            if st.button("Execute Batch Model Forecasting", type="secondary"):
                # Reset file cursor and convert file buffer to bytes array
                uploaded_file.seek(0)
                files_payload = {'file': ('batch_upload.csv', uploaded_file.read(), 'text/csv')}

                with st.spinner("Processing batch records..."):
                    response = requests.post(f"{BACKEND_API_URL}/v1/predictbatch", files=files_payload, timeout=30)

                if response.status_code == 200:
                    predictions_dict = response.json()
                    # Convert mapping to series
                    pred_series = pd.Series(predictions_dict).sort_index()

                    # Append predictions back to the viewable DataFrame
                    output_df = input_df.copy()
                    output_df['Predicted_Sales_Total'] = pred_series.values

                    st.success("🎉 Batch processing successfully completed!")
                    st.write("**Output Dataset Preview:**", output_df.head(10))

                    # Provide download link
                    csv_buffer = io.StringIO()
                    output_df.to_csv(csv_buffer, index=False)
                    st.download_button(
                        label="📥 Download Results CSV File",
                        data=csv_buffer.getvalue(),
                        file_name="SuperKart_Batch_Sales_Predictions.csv",
                        mime="text/csv"
                    )
                else:
                    st.error(f"Batch API failed ({response.status_code}): {response.text}")

        except Exception as file_err:
            st.error(f"Error processing dataset input: {str(file_err)}")
