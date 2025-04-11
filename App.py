import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
import pickle



customer_df = pd.read_csv("data/amazon_customers_data.csv")
sales_df = pd.read_csv("data/amazon_sales_data.csv")


def extract_features(customer_id):
    # Today's date for calculations
    today = pd.Timestamp(datetime.today().date())

    # Get customer info
    cust_row = customer_df[customer_df['CustomerID'] == customer_id]
    if cust_row.empty:
        raise ValueError("Customer ID not found in customer data.")

    age = cust_row['Age'].values[0]
    gender = cust_row['Gender'].values[0]
    reg_date = pd.to_datetime(cust_row['RegistrationDate'].values[0])
    customer_lifetime = (today - reg_date).days

    # Filter sales for this customer
    cust_sales = sales_df[sales_df['CustomerID'] == customer_id]
    total_spent = cust_sales['TotalPrice'].sum()
    total_orders = cust_sales.shape[0]

    # Preferred payment method
    if not cust_sales.empty:
        preferred_payment = cust_sales['PaymentMethod'].mode()[0]
        last_order_date = pd.to_datetime(cust_sales['OrderDate']).max()
        days_since_last = (today - last_order_date).days
    else:
        preferred_payment = "Amazon Pay"  # Default fallback
        days_since_last = customer_lifetime  # Assume never ordered

    # Derived metrics
    avg_order_value = total_spent / (total_orders + 1)
    spending_rate = total_spent / (customer_lifetime + 1)

    # Final feature dictionary
    return {
        "Age": age,
        "Gender": gender,
        "TotalSpent": total_spent,
        "TotalOrders": total_orders,
        "PreferredPaymentMethod": preferred_payment,
        "DaysSinceLastOrder": days_since_last,
        "CustomerLifetime": customer_lifetime,
        "AvgOrderValue": avg_order_value,
        "SpendingRate": spending_rate
    } # Assume we saved the feature extraction code here

with open("subscription_model.pkl", "rb") as model_file:
    clf = pickle.load(model_file)

with open("subscription_preprocessor.pkl", "rb") as preprocessor_file:
    preprocessor = pickle.load(preprocessor_file)

top_recommendation = pd.read_csv("top_recommendation.csv")  # Should contain CustomerID, ProductID, ProductName, Category

st.set_page_config(page_title="Amazon Subscription Predictor", layout="centered")

st.title("🛒 Amazon Subscription Predictor + Product Recommender")
st.markdown("Predict if a customer is likely to subscribe and recommend a product based on their behavior.")

# User Input
customer_id = st.text_input("Enter Customer ID")

if st.button("🔍 Predict Subscription Probability"):
    if not customer_id:
        st.warning("Please enter a valid Customer ID.")
    else:
        try:
            # Extract features from customer and sales data
            user_input = extract_features(customer_id)
            user_df = pd.DataFrame([user_input])

            # Display extracted customer details
            st.subheader("📋 Extracted Customer Details")
            st.dataframe(user_df.T.rename(columns={0: "Value"}))  # Transpose for readability

            # Preprocess and predict
            processed = preprocessor.transform(user_df)
            probability = clf.predict_proba(processed)[0][1]
            prediction = clf.predict(processed)[0]

            # Product recommendation
            rec_row = top_recommendation[top_recommendation["CustomerID"] == customer_id]
            if not rec_row.empty:
                recommended_product = rec_row["ProductName"].values[0]
                recommended_category = rec_row["Category"].values[0]
            else:
                recommended_product = "Not found"
                recommended_category = "N/A"

            # Show results
            st.success(f"💡 Probability of subscribing: **{probability:.2%}**")

            if prediction == 1:
                st.markdown(f"✅ **Likely to Subscribe** – Recommend **{recommended_product}** from category **{recommended_category}**.")
            else:
                st.markdown(f"❌ **Unlikely to Subscribe** – Try offering **{recommended_product}** with better incentives.")

        except Exception as e:
            st.error(f"Error processing customer ID: {str(e)}")
