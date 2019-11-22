
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

from sklearn import preprocessing

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

pd.set_option('display.max_columns',100)
pd.set_option('display.width',600)


Austrlian_holidays_2019 = {"2019-01-01": "New Years Day", "2019-01-26": 'Australia Day',
                           "2019-01-28": 'Australia Day (Observed)', "2019-03-11": 'Labour Day',
                           "2019-04-19": 'Good Friday', "2019-04-20": 'Saturday before Easter Sunday',
                           "2019-04-21": 'Easter Sunday',
                           "2019-04-22": 'Easter Monday', "2019-04-25": 'Anzac Day', "2019-05-10": 'Queens Birthday',
                           "2019-09-27": 'Friday before the AFL Grand Final', "2019-11-05": 'Melbourne Cup',
                           "2019-12-25": 'Christmas Day',
                           "2019-12-26": 'Boxing Day'
                           }


Austrlian_holidays_2018 = {"2018-01-01": "New Years Day", "2018-01-26": 'Australia Day',
                           "2018-01-28": 'Australia Day (Observed)', "2018-03-12": 'Labour Day',
                           "2018-03-30": 'Good Friday', "2018-03-31": 'Saturday before Easter Sunday',
                           "2018-04-01": 'Easter Sunday',
                           "2018-04-02": 'Easter Monday', "2018-04-25": 'Anzac Day', "2018-05-11": 'Queens Birthday',
                           "2018-09-28": 'Friday before the AFL Grand Final', "2018-11-06": 'Melbourne Cup',
                           "2018-12-25": 'Christmas Day',
                           "2018-12-26": 'Boxing Day'
                           }

Austrlian_holidays_2017 = {"2017-01-01": "New Years Day", "2017-01-02": 'New Years Day',
                           "2017-01-26": 'Australia Day', "2017-03-13": 'Labour Day',
                           "2017-04-14": 'Good Friday', "2017-04-15": 'Saturday before Easter Sunday',
                           "2017-04-16": 'Easter Sunday',
                           "2017-04-17": 'Easter Monday', "2017-04-25": 'Anzac Day', "2017-05-12": 'Queens Birthday',
                           "2017-09-29": 'Friday before the AFL Grand Final', "2017-11-07": 'Melbourne Cup',
                           "2017-12-25": 'Christmas Day',
                           "2017-12-26": 'Boxing Day'
                           }

Austrlian_holidays_2016 = {"2016-01-01": "New Years Day",
                           "2016-01-26": 'Australia Day', "2016-03-14": 'Labour Day',
                           "2016-03-25": 'Good Friday', "2016-03-26": 'Saturday before Easter Sunday',
                           "2016-03-27": 'Easter Sunday',
                           "2016-03-28": 'Easter Monday', "2016-04-25": 'Anzac Day', "2016-05-13": 'Queens Birthday',
                           "2016-09-30": 'Friday before the AFL Grand Final', "2016-11-01": 'Melbourne Cup',
                           "2016-12-25": 'Christmas Day',
                           "2016-12-26": 'Boxing Day'
                           }


def generate_holiday_dates(df_):
    holidays=[]
    for i in range(len(df_)):
        date_ = str(df_[i]).split(" ")
        try:
            out = str(Austrlian_holidays_2016[date_[0]])
            holidays.append(out)
        except:

            try:
                out =  str(Austrlian_holidays_2017[date_[0]])

                holidays.append(out)

            except:
                try:
                    out =  str(Austrlian_holidays_2018[date_[0]])

                    holidays.append(out)

                except:
                    try:
                        out =  str(Austrlian_holidays_2019[date_[0]])

                        holidays.append(out)

                    except:
                        out = 0
                        holidays.append("NaN")
                        continue

    return holidays

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


df = pd.read_csv('/kaggle/input/reatils-sales-daily-filtered/Final_dataframe.csv')
start_location = 280
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
revenue_save=new_df['TOTAL']


new_df=new_df.drop(columns=['TOTAL'])

print(new_df.columns)
new_df.insert(new_df.shape[1],'TOTAL', np.round(revenue_save))

new_df_encoded=new_df #totalcolumlocation change
print(new_df)

#print(ll)


prophet_date=new_df_encoded.iloc[:,0]
prophet_total_revenue=new_df_encoded.iloc[:,new_df_encoded.shape[1]-1]
prophet_total_revenue=prophet_total_revenue.values.reshape(-1,1)

scaler = MinMaxScaler(feature_range=(0, 1))
prophet_total_revenue_scaled = scaler.fit_transform(prophet_total_revenue)
print(prophet_total_revenue)



df_=pd.DataFrame()
df_['ds']=prophet_date
df_['y']=prophet_total_revenue_scaled

m = Prophet(changepoint_prior_scale=60, yearly_seasonality=40,weekly_seasonality=40, interval_width=0.9)
m.fit(df_)
future = m.make_future_dataframe(periods=0,freq='D')



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

dates=new_df_encoded.values[:,0]
print(new_df_encoded)
new_df_encoded=new_df_encoded.values[:,2:]
print(new_df_encoded)
#print(ll)

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
model.add(Activation('selu'))
model.add(LSTM(64))
model.add(Dropout(0.3))
model.add(Activation('selu'))
model.add(Dense(32))
model.add(Dropout(0.2))
model.add(Activation('selu'))
model.add(Dense(1))

#model.add(ELU(alpha=1.5))
# model.add()
opt=optimizers.Nadam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999)
model.compile(loss='mean_squared_error', optimizer=opt,metrics=['mae', 'acc'])
history=model.fit(x_train, y_train, epochs=2000, batch_size=128, verbose=2,validation_data=(x_test,y_test))
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


















