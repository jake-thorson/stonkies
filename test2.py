import yfinance as yf
import pandas as pd
from datetime import datetime
import streamlit as st
import xgboost as xgb
from xgboost import XGBClassifier
import numpy as np
from dateutil.relativedelta import relativedelta
import altair as alt
from sklearn import linear_model
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestRegressor




st.set_page_config(page_title='stonks', layout='wide')

def get_ticky(ticky):
    try:
        df = yf.download(tickers=ticky, period="5d", interval="1m")

        return df
    except:
        st.warning('delete ' + ticky)
        return pd.DataFrame()

def get_new_hour_minute(df):
    new_minute = int(df.at[len(df)-1, 'minute']) + 1
    new_hour = int(df.at[len(df)-1, 'hour'])
    if new_minute == 60:
        new_hour = new_hour + 1
        new_minute = 0
    return new_minute, new_hour

def get_time_as_string(time_val):
    new_time = str(time_val).replace('.0','')
    if len(str(new_time)) < 2:
        return ('0' + new_time).replace('.0','')
    return new_time.replace('.0','')

def scrub_it(ticky, train_cols):
    df = get_ticky(ticky)

    df.columns = [' '.join(col).strip() for col in df.columns.values]
    all_times = df.index # index is datetime :(

    # going to merge on time in case the call takes over 5 minutes
    df['month'] = [0] * len(df)
    df['day'] = [0] * len(df)
    df['time'] = ['0'] * len(df)
    df.index = np.arange(0, len(df))
    count = 0
    for i in all_times: # scrub time into cols
        temp = (str(i))

        df.at[count, 'month'] = float(temp[5:7])
        df.at[count, 'day'] = float(temp[8:10])
        df.at[count, 'hour'] = float(temp[11:13].replace(':',''))
        df.at[count, 'minute'] = float(temp[14:16].replace(':',''))

        df.at[count, 'time'] = str(temp[:4]) + get_time_as_string(str(df.at[count, 'month'])) + \
            get_time_as_string(str(df.at[count, 'day'])) + get_time_as_string(str(df.at[count, 'hour'])) + \
            get_time_as_string(str(df.at[count, 'minute']))

        count+=1

    for col in df:
        if 'Volume' in col or 'Low' in col or 'High' in col or 'Close' in col:
            df = df.drop(col, axis='columns')

    return df

def add_local_min_max(df, ticky):
    df['local_min_' + ticky] = np.where(
        (df['Open ' + ticky] < df['Open ' + ticky].shift(1)) &
        (df['Open ' + ticky] < df['Open ' + ticky].shift(-1)),
        df['Open ' + ticky],
        np.nan
    )

    df['local_max_' + ticky] = np.where(
        (df['Open ' + ticky] > df['Open ' + ticky].shift(1)) &
        (df['Open ' + ticky] > df['Open ' + ticky].shift(-1)),
        df['Open ' + ticky],
        np.nan
    )

    df['local_min_' + ticky] = df['local_min_' + ticky].ffill()
    df['local_max_' + ticky] = df['local_max_' + ticky].ffill()

    for i in range(1, train_cols):
        df['next_val_' + ticky + '_' + str(i)] = df['Open ' + ticky].shift(i)
    df = df.iloc[train_cols:]
    df.index = np.arange(0, len(df))

    return df

def fill_nones(df1, df2):
    for col in df1:
        if 'Open' in col:
            temp = df1.drop(col, axis='columns')
    df2 = df2.merge(temp, on = ['time', 'month', 'day', 'hour', 'minute'], how = 'outer')
    for col in df2:
        if 'Open' in col:
            df2[col] = df2[col].ffill()
    return df2

def create_prediction(l_rate, levels, training_data, train_cols, model, next_iters):

    btc = scrub_it('BTC-USD', train_cols)
    xrp = scrub_it('XRP-USD', train_cols)

    xrp = fill_nones(btc, xrp)
    btc  = fill_nones(xrp, btc)

    btc = add_local_min_max(btc, 'BTC-USD')
    xrp = add_local_min_max(xrp, 'XRP-USD')

    stonks = xrp.merge(btc, on = ['time', 'month', 'day', 'hour', 'minute'], how = 'outer')

    stonks = stonks.iloc[len(stonks)-training_data:]
    stonks.index = np.arange(0, len(stonks))


    def create_new_line(df, train_cols):
        new_line = pd.DataFrame({'Open XRP-USD': [0.0],
                                'Open BTC-USD': [0.0],
                                'minute': [df.at[len(df)-1, 'minute']+1],
                                'day': [df.at[len(df)-1, 'day']],
                                'month': [df.at[len(df)-1, 'month']]})
        for i in range(1, train_cols):
            new_line['next_val_XRP-USD_' + str(i)] = [df.at[len(df)-(i), 'Open XRP-USD']]
            new_line['next_val_BTC-USD_' + str(i)] = [df.at[len(df)-(i), 'Open BTC-USD']]

        if new_line.at[0, 'minute'] == 60:
            new_line.at[0, 'minute'] = 0
            new_line['hour'] = [df.at[len(df)-1, 'hour']+1]
        else:
            new_line['hour'] = [df.at[len(df)-1, 'hour']]

        new_line['time'] = str(df.at[len(df)-1, 'time'])[:8] + get_time_as_string(str(new_line.at[0, 'hour'])) + get_time_as_string(str(new_line.at[0, 'minute']))

        for col in df:
            if 'local' in col:
                new_line[col] = [0.0]

        return new_line


    def stonk_predict(stonks, l_rate, levels, train_cols, model_type, next_iters):
        for i in range(0, next_iters):

            new_line = create_new_line(stonks, train_cols)

            stonks = pd.concat([stonks, new_line], axis = 0)

            for col in stonks:
                if 'local' in col:
                    stonks[col] = stonks[col].shift(1)
            stonks = stonks.iloc[1:]
            stonks.index = np.arange(0, len(stonks))

            for col in stonks:
                if col != 'Open XRP-USD' and col != 'Open BTC-USD':
                    stonks[col] = stonks[col].astype(float)

            train = stonks.iloc[:len(stonks)-1]
            test = stonks.iloc[len(stonks)-1:]
            train.index = np.arange(0, len(train))
            test.index = np.arange(0, len(test))

            for each_target in ['Open XRP-USD', 'Open BTC-USD']:

                features = []
                for col in stonks:
                    if col != 'Open XRP-USD' and col != 'Open BTC-USD':
                        stonks[col] = stonks[col].astype(float)
                        features.append(col)

                if model_type == 'forest' or model_type == 'combo':
                    xtrain = train[features]
                    ytrain = train[each_target]
                    xtest = test[features]
                    ytest = test[each_target]

                    X = stonks[features]
                    y = stonks[each_target]

                    scaler = StandardScaler()

                    X_train_scaled = scaler.fit_transform(xtrain)
                    X_test_scaled = scaler.transform(xtest)

                    # Train Decision Tree Classifier
                    model = RandomForestRegressor(max_depth=levels, random_state=70)
                    model.fit(X_train_scaled, ytrain)

                    # Predict and evaluate
                    y_pred = model.predict(X_test_scaled)
                    result1 = y_pred.tolist()[0]

                features = []
                for col in stonks:
                    if col != 'Open XRP-USD' and col != 'Open BTC-USD':
                        features.append(col)

                xtrain = train[features]
                ytrain = train[each_target]
                xtest = test[features]
                ytest = test[each_target]

                if model_type == 'regression' or model_type == 'combo':

                    reg = xgb.XGBRegressor(n_estimators = 500, early_stopping_rounds = 15,
                                        learning_rate = l_rate, nthread = 1, verbosity = 0)
                    reg.fit(xtrain, ytrain, eval_set = [(xtrain, ytrain), (xtest, ytest)], verbose = True)

                    result = reg.predict(xtest)
                    result = result.tolist()[0]

                #acutal_result = actuals.loc[actuals['time'].astype(str) == str(stonks.at[len(stonks)-1, 'time']).replace('.0','')][each_target].tolist()[0]
                #all_results = pd.concat([all_results, pd.DataFrame({'random_forrest': [result1],
                #                                                    'levels': [levels],
                #                                                    'regression': [result],
                #                                                    'learning_rate': [l_rate],
                #                                                    'time': [stonks.at[len(stonks)-1, 'time']],
                #                                                    'actual': [acutal_result]})], axis = 0)
                #all_results.index = np.arange(0, len(all_results))

                #if 'XRP' in each_target:
                #    st.write(stonks.at[len(stonks)-1, 'time'])
                #    st.write(result)
                #    st.write(result1)

                if model_type == 'combo':
                    result = (result + result1) / 2
                if model_type == 'forest':
                    result = result1
                if model_type == 'regression':
                    result = result

            stonks.at[len(stonks)-1, each_target] = result

            for ticky in ['XRP-USD', 'BTC-USD']:
                stonks['local_min_' + ticky] = np.where(
                    (stonks['Open ' + ticky] < stonks['Open ' + ticky].shift(1)) &
                    (stonks['Open ' + ticky] < stonks['Open ' + ticky].shift(-1)),
                    stonks['Open ' + ticky],
                    np.nan)

                stonks['local_max_' + ticky] = np.where(
                    (stonks['Open ' + ticky] > stonks['Open ' + ticky].shift(1)) &
                    (stonks['Open ' + ticky] > stonks['Open ' + ticky].shift(-1)),
                    stonks['Open ' + ticky],
                    np.nan)

                stonks['local_min_' + ticky] = stonks['local_min_' + ticky].ffill()
                stonks['local_max_' + ticky] = stonks['local_max_' + ticky].ffill()
        return stonks

    new_stonkie = stonk_predict(stonks, l_rate, levels, train_cols, model_type, next_iters)

    new_stonkie=new_stonkie.iloc[len(new_stonkie)-15:]

    pred = pd.DataFrame({'xrp_pred': new_stonkie['Open XRP-USD'],
                        'btc_pred': new_stonkie['Open BTC-USD'],
                        'time': new_stonkie['time'].astype(str).str.replace('.0','')})
    pred = pred.iloc[len(pred)-next_iters-2:]

    y_min = pred['xrp_pred'].min()
    y_max = pred['xrp_pred'].max()

    chart = alt.Chart(pred).transform_fold(
        ['xrp_pred'],
        as_=['Variable', 'Value']).mark_line().encode(
        x='time',
        y=alt.Y('Value:Q',
                scale=alt.Scale(domain=[y_min, y_max], zero=False)),
        color=alt.Color('Variable:N'))

    st.altair_chart(chart, use_container_width=True)


next_iters = 5

l_rate = 0.03
levels = 20
training_data = 400
train_cols = 6
model_type = 'combo'

create_prediction(l_rate, levels, training_data, train_cols, model_type, next_iters)

l_rate = 0.5
levels = 10
training_data = 2500
train_cols = 4
model_type = 'combo'

create_prediction(l_rate, levels, training_data, train_cols, model_type, next_iters)

l_rate = 0.5
levels = 10
training_data = 800
train_cols = 8
model_type = 'combo'

create_prediction(l_rate, levels, training_data, train_cols, model_type, next_iters)

l_rate = 0.7
levels = 10
training_data = 200
train_cols = 12
model_type = 'combo'

create_prediction(l_rate, levels, training_data, train_cols, model_type, next_iters)

l_rate = 0.7
levels = 10
training_data = 200
train_cols = 12
model_type = 'forest'

create_prediction(l_rate, levels, training_data, train_cols, model_type, next_iters)

l_rate = 0.7
levels = 10
training_data = 400
train_cols = 4
model_type = 'regression'

create_prediction(l_rate, levels, training_data, train_cols, model_type, next_iters)
