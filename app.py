import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

# Constants and Data Input
DATA_PATH_GENERATION = "Plant_1_Generation_Data.csv"
DATA_PATH_WEATHER = "Plant_1_Weather_Sensor_Data.csv"
ANOMALY_THRESHOLD_PERCENT = 0.85  # Flags days producing <85% of expected power


@st.cache_data
def load_and_prep_data():
    #Load CSVs and preps data by date
    try:
        gen_df = pd.read_csv(DATA_PATH_GENERATION)
        weather_df = pd.read_csv(DATA_PATH_WEATHER)
    except FileNotFoundError:
        st.error(f"Missing data files. Please ensure {DATA_PATH_GENERATION} and {DATA_PATH_WEATHER} are in the root directory.")
        st.stop()
    
    # convert raw text to datetime object
    gen_df['DATE_TIME'] = pd.to_datetime(gen_df['DATE_TIME'])
    weather_df['DATE_TIME'] = pd.to_datetime(weather_df['DATE_TIME'])
    
    # report daily metrics
    daily_gen = gen_df.groupby(gen_df['DATE_TIME'].dt.date)['DAILY_YIELD'].max().reset_index()
    daily_weather = weather_df.groupby(weather_df['DATE_TIME'].dt.date).agg({
        'IRRADIATION': 'sum',
        'AMBIENT_TEMPERATURE': 'mean'
    }).reset_index()
    
    # combine the two CSVs
    merged_df = pd.merge(daily_gen, daily_weather, on='DATE_TIME')
    merged_df.rename(columns={
        'DATE_TIME': 'Date',
        'IRRADIATION': 'Irradiation',
        'AMBIENT_TEMPERATURE': 'Temperature',
        'DAILY_YIELD': 'Actual_Yield'
    }, inplace=True)
    
    return merged_df


def detect_anomalies(df):
    X = df[['Irradiation', 'Temperature']]
    y = df['Actual_Yield']

    model = LinearRegression()
    model.fit(X, y)

    df['Expected_Yield'] = model.predict(X)
    df['Is_Anomaly'] = df['Actual_Yield'] < (df['Expected_Yield'] * ANOMALY_THRESHOLD_PERCENT)
    
    return df


def main():
    st.set_page_config(page_title="Solar Analytics", layout="wide")
    st.title("Solarlytics: Performance Diagnostics")
    st.markdown("ML-based anomaly detection isolating equipment degradation from weather variations.")

    # Execute data pipeline
    raw_data = load_and_prep_data()
    df = detect_anomalies(raw_data)

    # Main metrics
    anomalies_count = df['Is_Anomaly'].sum()
    cols = st.columns(3)
    
    cols[0].metric("Monitored Days", len(df))
    
    cols[1].metric(
        label="Anomalies Detected", 
        value=anomalies_count, 
        delta="Inspection Required" if anomalies_count > 0 else "No anomalies", 
        delta_color="red" if anomalies_count > 0 else "green",
        delta_arrow="up" if anomalies_count > 0 else "down"
    )
    cols[2].metric("Avg Daily Yield", f"{df['Actual_Yield'].mean():.2f} kW")

    # time series graph
    st.subheader("Yield Trends vs ML Model")
    fig_timeseries = go.Figure()
    
    fig_timeseries.add_trace(go.Scatter(
        x=df['Date'], y=df['Expected_Yield'], 
        mode='lines', name='Expected (Baseline)', 
        line=dict(dash='dash', color='gray')
    ))
    
    fig_timeseries.add_trace(go.Scatter(
        x=df['Date'], y=df['Actual_Yield'], 
        mode='lines+markers', name='Actual Yield', 
        marker=dict(color=np.where(df['Is_Anomaly'], '#d62728', '#1f77b4')) 
    ))
    
    fig_timeseries.update_layout(xaxis_title="Date", yaxis_title="Power Yield (kW)", margin=dict(t=30))
    st.plotly_chart(fig_timeseries, use_container_width=True)

    # extra diagnostics
    col_chart, col_table = st.columns(2)

    with col_chart:
        st.markdown("**Yield vs Irradiation**")
        fig_scatter = px.scatter(
            df, x='Irradiation', y='Actual_Yield', 
            color='Is_Anomaly', 
            color_discrete_map={True: '#d62728', False: '#1f77b4'},
            hover_data=['Temperature', 'Date']
        )
        fig_scatter.update_layout(margin=dict(t=10))
        st.plotly_chart(fig_scatter, use_container_width=True)

    with col_table:
        st.markdown("**Incident Log**")
        anomalies_df = df[df['Is_Anomaly']].copy()
        
        if not anomalies_df.empty:
            anomalies_df['Deficit'] = ((1 - (anomalies_df['Actual_Yield'] / anomalies_df['Expected_Yield'])) * 100).round(1).astype(str) + "%"
            st.dataframe(anomalies_df[['Date', 'Actual_Yield', 'Expected_Yield', 'Deficit']], hide_index=True)
        else:
            st.success("No anomalies detected. System operating within expected parameters.")


if __name__ == "__main__":
    main()
