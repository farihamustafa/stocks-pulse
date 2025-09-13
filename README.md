StocksPulse

StocksPulse is a stock market analytics and anomaly detection software powered by Streamlit. It supports real-time data ingestion, anomaly detection, forecasting, and portfolio analysis, as well as automatic reporting and alarms.

Features

Interactive dashboards include infographics, KPIs, and portfolio analytics.

Real-time stock data from the Yahoo Finance API

Anomaly detection with Isolation Forest, One-Class SVM, and LOF

Forecasting with Prophet for trends and seasonality analysis

Financial news integration with Google News RSS

Automated email and SMS reports utilizing the Gmail API and Twilio API.
  System Architecture

Frontend: A streamlined interface with charts, tabs, forms, and alerts

Backend modules include data ingestion, feature engineering, anomaly detection, and forecasting.

External APIs include Yahoo Finance for stock data, Google News RSS for insights, Gmail, and Twilio for notifications.

Deployment: Streamlit instance in a browser context.

Installation Prerequisites:

Python 3.8+

Pip Package Manager

API keys for Yahoo Finance, Gmail, and Twilio API.
  teps

Clone the repository.

Clone https://github.com/farihamustafa/stocks-pulse and change directory to stocks-pulse.


Create and activate a virtual environment.

python -m venv venv source venv/bin/activate # Linux/Mac venv\Scripts\activate #Windows


Installing dependencies:

pip install -r requirements.txt.


Set up environment variables in a secrets.toml file in .streamlit folder.

sender_email = "---------------------" # Enter your gmail
sender_pass = "-----------------------" # Enter your generated smtp password


Run the application:

streamlit run app.py

Usage
  Once Streamlit is operating, access the app through your browser.

Choose data sources and perform anomaly detection or forecasting.

Explore portfolio dashboards and news insights.

Export reports or set up automated email/SMS notifications.

Testing

Run the unit and integration tests:

Pytest tests/

Contribution

Fork the repository and make a separate branch for every feature.

Follow the PEP8 code standards and include tests with contributions.

Submit a pull request that includes comprehensive documentation of modifications.

License

This project is licensed under the MIT license. See the LICENSE file for more information.  

