import math

from fbprophet import Prophet
from keras import optimizers
from numpy import concatenate, sqrt
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import datetime
from sklearn.decomposition import PCA
from holidays_code import generate_holiday_dates
from sklearn import preprocessing
pd.set_option('display.max_columns',100)
pd.set_option('display.width',600)

df = pd.read_csv('Final_dataframe.csv')
start_location = 80
df = df.iloc[start_location:, 0:df.shape[1]]
date = df.iloc[start_location:, 0]
print(date)
df = df.reset_index(drop=True)
#df['COUNT']=df.index
#print(df)

df['HOLIDAY']=generate_holiday_dates(df['DATENEW'])     #holiday name gneration according to the date
df['DATENEW']=pd.to_datetime(df['DATENEW'], errors='coerce')
df['MONTH']= pd.DatetimeIndex(df['DATENEW']).month
df['DAY']=pd.DatetimeIndex(df['DATENEW']).day
df['DAY_NAME'] = df['DATENEW'].dt.day_name()             #dayofweek
#print(df)


new_df=df[['DATENEW','COUNT','HOLIDAY','MONTH','DAY','DAY_NAME','TOTAL']]  #filterdataframe by removing the item codes
new_df=pd.get_dummies(new_df, columns=["DAY_NAME","HOLIDAY"])               #Hot encode the day_name and holiday colums
new_df_encoded=new_df[['DATENEW', 'MONTH', 'DAY', 'DAY_NAME_Friday', 'DAY_NAME_Monday', 'DAY_NAME_Saturday', 'DAY_NAME_Sunday', 'DAY_NAME_Thursday', 'DAY_NAME_Tuesday', 'DAY_NAME_Wednesday', 'HOLIDAY_Anzac Day', 'HOLIDAY_Australia Day', 'HOLIDAY_Australia Day (Observed)', 'HOLIDAY_Christmas Day', 'HOLIDAY_Easter Monday', 'HOLIDAY_Friday before the AFL Grand Final', 'HOLIDAY_Labour Day', 'HOLIDAY_Melbourne Cup', 'HOLIDAY_NaN', 'HOLIDAY_New Years Day', 'HOLIDAY_Queens Birthday', 'HOLIDAY_Saturday before Easter Sunday', 'TOTAL']] #totalcolumlocation change
print(new_df)

prophet_date=new_df_encoded.iloc[:,0]
prophet_total_revenue=new_df_encoded.iloc[:,new_df_encoded.shape[1]-1]
prophet_total_revenue=prophet_total_revenue.values.reshape(-1,1)

scaler = MinMaxScaler(feature_range=(0, 1))
prophet_total_revenue_scaled = scaler.fit_transform(prophet_total_revenue)


df_=pd.DataFrame()
df_['ds']=prophet_date
df_['y']=prophet_total_revenue_scaled

m = Prophet(weekly_seasonality=True,yearly_seasonality=True)
m.fit(df_)
future = m.make_future_dataframe(periods=0)



forecast = m.predict(future)
print(forecast)
prophet_revenue_forecast=forecast['yhat']
print(prophet_revenue_forecast)

prophet_revenue_forcaset_reshaped=prophet_revenue_forecast.values.reshape(-1,1)
forecast_revenue_output=scaler.inverse_transform(prophet_revenue_forcaset_reshaped)

forecast_revenue_output=forecast_revenue_output.reshape(-1)
print(forecast_revenue_output)



print(prophet_total_revenue)


error_score = math.sqrt(mean_squared_error(forecast_revenue_output, prophet_total_revenue.reshape(-1)))
print('Error score: %.2f RMSE' % (error_score))
new_df_encoded.insert(new_df_encoded.shape[1]-1,'PROPHET_FORECAST', np.round(forecast_revenue_output))
print(new_df_encoded)


fig = go.Figure()
fig.add_trace(go.Scatter(
    y=forecast_revenue_output,
    x=forecast['ds'],
    mode='lines',
))

fig.add_trace(go.Scatter(
    y=prophet_total_revenue.reshape(-1),
    x=forecast['ds'],
    mode='lines',
))


fig.show()

fig = m.plot_components(forecast)
pyplot.show()



dates=new_df_encoded.values[:,0]

new_df_encoded=new_df_encoded.values[:,1:]



train_size = int(new_df_encoded.shape[0] * 0.90)
test_size = new_df_encoded.shape[0] - train_size


scaler = MinMaxScaler(feature_range=(0, 2))#feature_range=(0, 1)
new_df_encoded = scaler.fit_transform(new_df_encoded)



x_train = new_df_encoded[0:train_size, 0:new_df_encoded.shape[1]-1]
x_test = new_df_encoded[train_size:len(new_df_encoded), 0:new_df_encoded.shape[1]-1]

y_train= new_df_encoded[0:train_size,new_df_encoded.shape[1]-1]
y_test = new_df_encoded[train_size:len(new_df_encoded),new_df_encoded.shape[1]-1]


print(x_train)

print(y_train)
#print(ll)

#x_train = PCA(n_components=16).fit_transform(x_train)
#x_test = PCA(n_components=16).fit_transform(x_test)

x_train=x_train[:,:,np.newaxis]
x_test=x_test[:,:,np.newaxis]
print(x_train.shape)



model = Sequential()

model.add(LSTM(100,input_shape=(22,1),return_sequences=True))
model.add(Activation('relu'))
model.add(Dense(64))
model.add(Dropout(0.1))
model.add(Activation('relu'))
model.add(LSTM(32))
model.add(Activation('relu'))
model.add(Dense(1))

#model.add(ELU(alpha=1.5))
# model.add()
opt=optimizers.Nadam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999)
model.compile(loss='mean_squared_error', optimizer=opt,metrics=['mae', 'acc'])
history=model.fit(x_train, y_train, epochs=1000, batch_size=256, verbose=2,validation_data=(x_test,y_test))
#model.save("trained_revenue_model.h5")
# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

testPredict = model.predict(x_test)
trainPredict = model.predict(x_train)

print(testPredict)
testScore = math.sqrt(mean_squared_error(y_test, testPredict))
print('Test Score: %.2f RMSE' % (testScore))

trainScore = math.sqrt(mean_squared_error(y_train, trainPredict))
print('Train Score: %.2f RMSE' % (trainScore))


predicted_out=trainPredict.reshape(-1).tolist()+testPredict.reshape(-1).tolist()
actual_out=y_train.tolist()+y_test.tolist()

fig = go.Figure()

fig.add_trace(go.Scatter(
    y=actual_out,
    x=dates.tolist(),
    mode='lines',
))

fig.add_trace(go.Scatter(
    y=predicted_out,
    x=dates.tolist(),
    mode='lines',
))

fig.show()










