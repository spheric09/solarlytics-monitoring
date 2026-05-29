# Solarlytics: Solar Plant Performance Diagnostics & Anomaly Detection

A proof-of-concept Technical Asset Management dashboard built to simulate daily real-time monitoring, data validation, and pre-diagnostics for solar power plants. 

Live App Link: [streamlit cloud URL]

## Key Features & Asset Intelligence Workflow
* **Data Integration & Validation:** Merges and processes split-source CSV infrastructure data (Generation and Weather sensor telemetry) using Pandas, simulating data logger troubleshooting.
* **ML-Based Performance Baseline:** Utilizes a `Scikit-Learn` Linear Regression model to dynamically calculate expected power yield based on real-time solar irradiation and ambient temperature variations.
* **Automated Pre-Diagnostics:** Flags underperforming assets automatically when actual power yield drops below 85% of the machine learning model's expected baseline.
* **Interactive Dashboarding:** Built with `Streamlit` and `Plotly` to display interactive trend analysis, incident logs, and yield-vs-irradiation profiles for technical teams.

## Tech Stack
* **Language:** Python
* **Data Processing:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn (Linear Regression)
* **Visualization & UI:** Streamlit, Plotly

## How to Run Locally
1. Clone this repository:
   git clone [https://github.com/spheric09/solarlytics-monitoring.git](https://github.com/spheric09/solarlytics-monitoring.git)
2. Install Dependencies:
   pip install -r requirements.txt
3. Run the dashboard:
   streamlit run app.py
