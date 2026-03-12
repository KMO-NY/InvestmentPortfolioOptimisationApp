import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from yfinance import Ticker
from datetime import datetime, timedelta
import quantstats as qs

# Section: Predictions
st.header("Stock Price Predictions")
st.write("Still Working On It!")
# prediction_choice = st.radio(
#     "Select Prediction Horizon:",
#     ["Make Your Choice","1 Day", "1 Week", "1 Month"],
#     index=0
# )
# if prediction_choice == "Make Your Choice":
#     st.write(":red[Kindly Select An Option].")
# else:
#     st.write(f":green[Displaying Price Predictions For {prediction_choice} horizon.]")
