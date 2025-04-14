import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from datetime import datetime
from streamlit_lottie import st_lottie
import json
import requests

# Function to load Lottie animation from a URL with error handling
def load_lottie_url(url: str):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Check for errors
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error loading animation: {e}")
        return None

# Load model and supporting files
model = joblib.load('traffic_model.pkl')
label_encoders = joblib.load('label_encoders.pkl')
feature_columns = joblib.load('feature_columns.pkl')
df = pd.read_csv('mobility_with_new_features.csv')

# Load the Lottie animation
lottie_animation = load_lottie_url("https://assets9.lottiefiles.com/packages/lf20_Q7WY7CfUco.json")

# App config
st.set_page_config(page_title="Traffic Predictor Dashboard", layout="wide", page_icon="🚦", initial_sidebar_state="expanded")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Traffic Prediction", "EDA Dashboard"])

# Home
if page == "Home":
    st.title("🚦 Smart Traffic Prediction System")
    if lottie_animation:
        st_lottie(lottie_animation, speed=1, loop=True, quality="high", height=300)
    st.markdown("""
        Welcome to the interactive dashboard for traffic prediction using smart mobility data. 
        Navigate using the sidebar to explore insights or predict traffic conditions.
    """)

# Traffic Prediction
elif page == "Traffic Prediction":
   with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        vehicle_count = st.number_input("Vehicle Count", min_value=0)
        traffic_speed = st.slider("Traffic Speed (km/h)", 0, 150, 50)
        road_occupancy = st.slider("Road Occupancy (%)", 0, 100, 30)
    with col2:
        traffic_light = st.selectbox("Traffic Light State", label_encoders['Traffic_Light_State'].classes_)
        weather = st.selectbox("Weather Condition", label_encoders['Weather_Condition'].classes_)
        # Ensure that 'Accident_Report' exists in your label_encoders
        if 'Accident_Report' in label_encoders:
            accident = st.selectbox("Accident Report", label_encoders['Accident_Report'].classes_)
        else:
            st.error("Error: 'Accident_Report' label encoder is missing.")
            accident = None
    hour = st.slider("Hour of Day", 0, 23, datetime.now().hour)
    day_of_week = st.selectbox("Day of Week", label_encoders['DayOfWeek'].classes_)
    submit = st.form_submit_button("Predict")

if submit:
    input_dict = {
        'Vehicle_Count': vehicle_count,
        'Traffic_Speed_kmh': traffic_speed,
        'Road_Occupancy_%': road_occupancy,
        'Traffic_Light_State': traffic_light,
        'Weather_Condition': weather,
        'Accident_Report': accident,  # If accident is None, it should not be added
        'Hour': hour,
        'DayOfWeek': day_of_week
    }
    for col, le in label_encoders.items():
        if col in input_dict and input_dict[col] is not None:
            input_dict[col] = le.transform([input_dict[col]])[0]
    input_df = pd.DataFrame([input_dict])[feature_columns]
    prediction = model.predict(input_df)[0]
    target_le = label_encoders['Traffic_Condition']
    prediction_label = target_le.inverse_transform([prediction])[0]
    st.success(f"Predicted Traffic Condition: **{prediction_label}**")


# EDA Dashboard
elif page == "EDA Dashboard":
    st.title("📊 Exploratory Data Analysis Dashboard")
    with st.sidebar:
        st.subheader("Filter Data")
        date_range = st.date_input("Select Date Range", [df['Date'].min(), df['Date'].max()])
        hour_range = st.slider("Select Hour Range", 0, 23, (0, 23))
        weather_filter = st.multiselect("Weather", options=df['Weather_Condition'].unique(), default=df['Weather_Condition'].unique())
        traffic_filter = st.multiselect("Traffic Condition", options=df['Traffic_Condition'].unique(), default=df['Traffic_Condition'].unique())

    filtered_df = df[ 
        (pd.to_datetime(df['Date']).between(pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1]))) &
        (df['Hour'].between(hour_range[0], hour_range[1])) &
        (df['Weather_Condition'].isin(weather_filter)) &
        (df['Traffic_Condition'].isin(traffic_filter))
    ]

    col1, col2, col3 = st.columns(3)
    col1.metric("Average Speed", f"{filtered_df['Traffic_Speed_kmh'].mean():.2f} km/h")
    col2.metric("Avg. Road Occupancy", f"{filtered_df['Road_Occupancy_%'].mean():.2f}%")
    col3.metric("Total Vehicles", f"{filtered_df['Vehicle_Count'].sum()}")

    st.subheader("Traffic Conditions by Hour")
    fig = px.histogram(filtered_df, x="Hour", color="Traffic_Condition", barmode="group")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Speed vs Vehicle Count")
    fig2 = px.scatter(filtered_df, x="Traffic_Speed_kmh", y="Vehicle_Count", color="Traffic_Condition")
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Road Occupancy Over Time")
    fig3 = px.line(filtered_df, x="Timestamp", y="Road_Occupancy_%", color="Traffic_Condition")
    st.plotly_chart(fig3, use_container_width=True)