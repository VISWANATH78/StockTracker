import numpy as np
import pandas as pd
import pandas_datareader as data
import matplotlib.pyplot as plt
import yfinance as yf
import keras
import plotly.express as px
from keras.models import load_model
import streamlit as st

start = '1999-01-01'
end='2022-01-01'

st.title('stock trend prediction')
user_input =st.text_input('enter stock name','AAPL')
df =data.DataReader(user_input, 'yahoo',start, end)


st.subheader('data from 1999 -2022')
st.write(df.describe())


st.subheader('closing Price vs time chart')
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('closing Price vs time chart with 100 moving average')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('closing Price vs time chart with 100MA 200MA')
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(ma200)
plt.plot(df.Close)
st.pyplot(fig)

data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.80)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.20):int(len(df))])

print(data_training.shape)
print(data_testing.shape)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_training_array=scaler.fit_transform(data_training)





#LOAD MODEL

model = load_model('keras_model.h5')

past_100_days =data_training.tail(100)
final_df = past_100_days.append(data_testing,ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test =[]
y_test=[]
for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])

x_test, y_test = np.array(x_test),np.array(y_test)
y_predicted = model.predict(x_test)
scaler=scaler.scale_

scale_factor =1/scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

#final graph
st.subheader('predicted vs original') 
fig2=plt.figure(figsize=(12,6))
plt.plot(y_test,'b',label='PREDICTED')
plt.plot(y_predicted,'r',label=' ORIGINAL PRICE ')
plt.xlabel('time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)

