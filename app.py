import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,accuracy_score

# Load historical stock data (assuming you have a CSV file)
data = pd.read_csv("/Dataset/ADANIPORTS.csv")

# Preprocess the data
data['Diff'] = data['Close'] - data['Open']

# Drop rows with missing values in any column
data.dropna(inplace=True)

# Split data into features and target variable
X = data[[ 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]  # Use relevant features
y = [1 if diff > 0 else 0 for diff in data['Diff']]  # Create binary labels based on 'Diff'

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a logistic regression model
logreg_classifier = LogisticRegression(random_state=42)  # You can adjust hyperparameters
logreg_classifier.fit(X_train, y_train)

# Make predictions
def predict_price(features):
    return logreg_classifier.predict([features])

# Streamlit web app
st.title('Stock Price Prediction')

# Introduction and Description
st.markdown('''
            Welcome to our Stock Price Predictor App! This app uses machine learning to forecast stock prices based on historical data. 
            Simply input the stock features on the sidebar and click "Predict" to see the predicted stock price.
            ''')

# Sidebar with input fields
st.sidebar.header('Input Features')
feature1 = st.sidebar.number_input('Open')
feature2 = st.sidebar.number_input('High')
feature3 = st.sidebar.number_input('Low')
feature4 = st.sidebar.number_input('Close')
feature5 = st.sidebar.number_input('Adj Close')
feature6 = st.sidebar.number_input('Volume')
# Add more input fields as needed

# Make prediction and display result
if st.sidebar.button('Predict'):
    features = [feature1, feature2, feature3, feature4, feature5, feature6]  # Add more features here
    prediction = predict_price(features)
    if prediction[0] == 1:
        result_text = "Predicted Stock Price: **Up**"
    else:
        result_text = "Predicted Stock Price: **Down**"
    st.markdown(f'<div style="background-color:#f0f0f0; border-radius:5px; padding:10px;">{result_text}</div>', unsafe_allow_html=True)

# Display an example graph
st.header('Stock Price Trends')
st.write('Here is an example graph showing stock price trends over time:')
df_plot = data[['Close']]  # No need for data.index() here
st.line_chart(df_plot)


# Optionally, display model performance metrics
# For example, you can calculate and display RMSE
y_pred = logreg_classifier.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
score = accuracy_score(y_test, y_pred)


st.write(f'Root Mean Squared Error (RMSE): {rmse}')

