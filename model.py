import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import pickle

# 1. Get Data Ready
bike_data = pd.read_csv("data/train.csv")
bike_test_data = pd.read_csv("data/test.csv")

# Create X (features matrix)
X = bike_data.drop(["count"], axis=1)

# Create y (labels)
y = bike_data["count"]

train_df = bike_data.copy()
test_df = bike_test_data.copy()
bike_data_dt = bike_data.copy()

wrangled_df_train = bike_data.drop(
    ["datetime", "season", "holiday", "atemp", "windspeed", "casual", "registered"], axis=1)

Z = bike_data.drop(["datetime", "season", "holiday", "atemp", "windspeed", "casual", "registered", "count"], axis=1)
A = bike_data.drop(["datetime", "season", "holiday", "atemp", "windspeed", "casual", "registered"], axis=1)

weather = pd.get_dummies(Z['weather'], prefix='weather')
wdf = pd.concat([A, weather], axis=1)

x_train, x_test, y_train, y_test = train_test_split(wdf.drop('count', axis=1),
                                                    wdf['count'], test_size=0.2, random_state=42)

np.random.seed(42)

X = wdf.drop(["count"], axis=1)
y = wdf["count"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = RandomForestRegressor(n_estimators=100)
clf.fit(X_train, y_train)

# Make Predictions
from sklearn.datasets import make_regression
X, y = make_regression(n_features=7, n_informative=2, random_state=0, shuffle=False)
regr = RandomForestRegressor(max_depth=8, random_state=10, n_estimators=100)
regr.fit(X, y)
# workingday, temp, humidity, weather_1, weather_2, weather_3, weather_4

y_preds = regr.predict(X_test)
mae = mean_absolute_error(y_test, y_preds)

# dump the model regr to the disk
pickle.dump(regr, open('pred_model.pkl', 'wb'))

# load the model from the disk

pred_model = pickle.load(open('pred_model.pkl', 'rb'))

