import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from sklearn.metrics import confusion_matrix, classification_report
import sqlite3
from datetime import datetime
import time
from streamlit_lottie import st_lottie

class TrafficPredictorApp:
    def __init__(self):
        self.model = joblib.load('traffic_model.pkl')
        self.label_encoders = joblib.load('label_encoders.pkl')
        self.feature_columns = joblib.load('feature_columns.pkl')

        # Initialize SQLite connection
        self.db_connection = sqlite3.connect('traffic_predictions.db')
        self.db_cursor = self.db_connection.cursor()
        self._create_table()

        # Set dark theme
        st.set_page_config(page_title="Traffic Condition Predictor", page_icon="🚦", layout="wide", initial_sidebar_state="expanded")
        st.markdown('<style>body {background-color: #212121; color: white;}</style>', unsafe_allow_html=True)

        if 'history' not in st.session_state:
            st.session_state['history'] = pd.DataFrame(columns=[ 
                'Vehicle_Count', 'Traffic_Speed_kmh', 'Road_Occupancy_%', 
                'Traffic_Light_State', 'Weather_Condition', 'Accident_Report', 
                'Hour', 'DayOfWeek', 'Prediction'
            ])

    def _create_table(self):
        """Create table in SQLite database."""
        self.db_cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            Timestamp TEXT,
            Vehicle_Count INTEGER,
            Traffic_Speed_kmh REAL,
            Road_Occupancy REAL,
            Traffic_Light_State TEXT,
            Weather_Condition TEXT,
            Accident_Report INTEGER,
            Hour INTEGER,
            DayOfWeek TEXT,
            Prediction TEXT
        )
        ''')
        self.db_connection.commit()

    def log_to_sqlite(self, input_data, prediction):
        """Log input and prediction to SQLite database."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.db_cursor.execute('''
        INSERT INTO predictions (Timestamp, Vehicle_Count, Traffic_Speed_kmh, Road_Occupancy, 
        Traffic_Light_State, Weather_Condition, Accident_Report, Hour, DayOfWeek, Prediction)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (timestamp, input_data['Vehicle_Count'], input_data['Traffic_Speed_kmh'], 
              input_data['Road_Occupancy_%'], input_data['Traffic_Light_State'], 
              input_data['Weather_Condition'], input_data['Accident_Report'], 
              input_data['Hour'], input_data['DayOfWeek'], prediction))
        self.db_connection.commit()

    def show_title(self):
        st.title("🚦 Traffic Condition Predictor")
        st.markdown("Make predictions and see how traffic condition patterns evolve over time.")

    def get_user_input(self):
        with st.form("prediction_form"):
            input_data = {}
            input_data['Vehicle_Count'] = st.number_input("Vehicle Count", min_value=0)
            input_data['Traffic_Speed_kmh'] = st.number_input("Traffic Speed (km/h)", min_value=0.0)
            input_data['Road_Occupancy_%'] = st.slider("Road Occupancy (%)", 0.0, 100.0)
            input_data['Traffic_Light_State'] = st.selectbox(
                "Traffic Light State", 
                self.label_encoders['Traffic_Light_State'].classes_
            )
            input_data['Weather_Condition'] = st.selectbox(
                "Weather Condition", 
                self.label_encoders['Weather_Condition'].classes_
            )
            accident = st.radio("Accident Reported?", ['No', 'Yes'])
            input_data['Accident_Report'] = 1 if accident == 'Yes' else 0
            input_data['Hour'] = st.slider("Hour of Day (0-23)", 0, 23)
            input_data['DayOfWeek'] = st.selectbox(
                "Day of Week", 
                self.label_encoders['DayOfWeek'].classes_
            )
            submitted = st.form_submit_button("Predict")
        return submitted, input_data

    def preprocess_input(self, input_data):
        df = pd.DataFrame([input_data])
        for col in ['Traffic_Light_State', 'Weather_Condition', 'DayOfWeek']:
            df[col] = self.label_encoders[col].transform(df[col])
        return df[self.feature_columns]

    def make_prediction(self, input_df):
        prediction = self.model.predict(input_df)
        proba = self.model.predict_proba(input_df)[0]
        classes = self.label_encoders['Traffic_Condition'].inverse_transform(
            list(range(len(proba)))
        )
        return prediction[0], classes, proba

    def store_history(self, raw_input, pred_class):
        entry = raw_input.copy()
        entry['Prediction'] = self.label_encoders['Traffic_Condition'].inverse_transform([pred_class])[0]
        st.session_state['history'] = pd.concat([st.session_state['history'], pd.DataFrame([entry])], ignore_index=True)

    def show_prediction(self, pred_class, classes, proba):
        decoded = self.label_encoders['Traffic_Condition'].inverse_transform([pred_class])[0]
        st.success(f"🚗 Predicted Traffic Condition: **{decoded}**")

        prob_df = pd.DataFrame({
            'Traffic Condition': classes,
            'Probability': proba
        }).sort_values(by='Probability', ascending=False)

        fig = px.bar(
            prob_df,
            x='Traffic Condition',
            y='Probability',
            color='Traffic Condition',
            title="Model Confidence per Traffic Condition",
            text_auto='.2f'
        )
        st.plotly_chart(fig, use_container_width=True)

    def show_history_chart(self):
        if not st.session_state['history'].empty:
            st.subheader("📈 Prediction History")
            hist_df = st.session_state['history']
            fig = px.histogram(
                hist_df,
                x='Prediction',
                color='Prediction',
                title='Traffic Condition Prediction Frequency',
                barmode='group'
            )
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(hist_df[::-1])  # show most recent on top

    def show_model_details(self):
        st.subheader("📊 Model Evaluation")
        y_true = st.session_state['history']['Prediction']
        y_pred = st.session_state['history']['Prediction']  # Or use model for predictions
        cm = confusion_matrix(y_true, y_pred)
        st.write("Confusion Matrix:")
        st.write(cm)
        st.write("Classification Report:")
        st.text(classification_report(y_true, y_pred))

    def show_about(self):
        st.subheader("🔍 About")
        st.markdown("""
            This Traffic Condition Prediction app leverages machine learning to predict traffic conditions 
            based on various input parameters, including vehicle count, traffic speed, road occupancy, 
            traffic light state, and weather conditions.
            **Features:**
            - Make predictions for traffic conditions.
            - View model evaluation metrics.
            - Track prediction history with charts.
        """)

    def show_progress_bar(self):
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.05)
            progress_bar.progress(i + 1)

    def show_lottie_animation(self):
        url = "https://assets9.lottiefiles.com/packages/lf20_Na7AM3.json"
        st_lottie(url, speed=1, width=500, height=500)

    def run(self):
        self.show_lottie_animation()
        self.show_title()

        # Create dashboard tabs
        tab_selection = st.sidebar.radio(
            "Select a Section:",
            ("Prediction", "Prediction History", "Model Details", "About")
        )

        if tab_selection == "Prediction":
            submitted, user_input = self.get_user_input()
            if submitted:
                self.show_progress_bar()  # Show progress bar during prediction
                input_df = self.preprocess_input(user_input)
                pred_class, classes, proba = self.make_prediction(input_df)
                self.store_history(user_input, pred_class)
                self.log_to_sqlite(user_input, pred_class)  # Log to database
                self.show_prediction(pred_class, classes, proba)

        elif tab_selection == "Prediction History":
            self.show_history_chart()

        elif tab_selection == "Model Details":
            self.show_model_details()

        elif tab_selection == "About":
            self.show_about()


if __name__ == "__main__":
    app = TrafficPredictorApp()
    app.run()
