#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =============================================================================
# Imports #%%
# =============================================================================
#part| #%%
from datetime import datetime

import numpy as np
import pandas as pd
from datetime import timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import VotingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.decomposition import FastICA
from sklearn.ensemble import StackingRegressor
import xgboost as xgb
from tpot import TPOTRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
import tensorflow as tf 
from tensorflow.keras import callbacks
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator
from sklearn.preprocessing import FunctionTransformer
from copy import copy
import joblib as joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler

def max_normalize(x):
    return (x - x.min()) / (x.max() - x.min())

def mse(ground_truth, predictions):
    diff = (ground_truth - predictions)**2
    return diff.mean()

def mae(ground_truth, predictions):
    diff = abs(ground_truth - predictions)
    return diff.mean()

df = pd.read_csv("../data/train/all_data_merged/fr/Enriched_Covid_history_data.csv")
df = df.dropna()
df["all_day_bing_tiles_visited_relative_change"]=df["all_day_bing_tiles_visited_relative_change"].astype(float)
df["all_day_ratio_single_tile_users"]=df["all_day_ratio_single_tile_users"].astype(float)
print(df)
columnstonormalize = ['pm25', 'no2','o3','pm10','co','pm257davg','no27davg','o37davg','co7davg', 'pm107davg','1MMaxpm25','1MMaxpm10','1MMaxo3','1MMaxno2','1MMaxco']
# for c in columnstonormalize:
#     df[c]=max_normalize(df[c])

featuresandtarget = ['idx', 'pm25', 'no2',
'o3','pm10','co',\
    'pm257davg','no27davg',\
    'o37davg','co7davg', 'pm107davg',\
        'hospiprevday','covidpostestprevday',\
            'all_day_bing_tiles_visited_relative_change','all_day_ratio_single_tile_users','vac1nb', 'vac2nb',\
                 'Insuffisance respiratoire chronique grave (ALD14)', \
                     'Insuffisance cardiaque grave, troubles du rythme graves, cardiopathies valvulaires graves, cardiopathies congénitales graves (ALD5)',\
                         'Smokers',\
                             "minority",\
                                  "Nb_susp_501Y_V1","Nb_susp_501Y_V2_3",\
                                      '1MMaxpm25','1MMaxpm10','1MMaxo3','1MMaxno2','1MMaxco',\
                                          'pm251Mavg','no21Mavg','o31Mavg','pm101Mavg','co1Mavg',\
                                           'newhospi'\

                            ]
                        
features = ['idx', 'pm25', 'no2',
'o3','pm10','co',\
    'pm257davg','no27davg',\
    'o37davg','co7davg', 'pm107davg',\
        'hospiprevday','covidpostestprevday',\
            'all_day_bing_tiles_visited_relative_change','all_day_ratio_single_tile_users','vac1nb', 'vac2nb',\
                 'Insuffisance respiratoire chronique grave (ALD14)', \
                     'Insuffisance cardiaque grave, troubles du rythme graves, cardiopathies valvulaires graves, cardiopathies congénitales graves (ALD5)',\
                         'Smokers',\
                         "minority",\
                             "Nb_susp_501Y_V1","Nb_susp_501Y_V2_3",
                                 '1MMaxpm25','1MMaxpm10','1MMaxo3','1MMaxno2','1MMaxco',\
                                 'pm251Mavg','no21Mavg','o31Mavg','pm101Mavg','co1Mavg'\
                                     \
                            ]

featurespm25 = ['idx', 'pm25',\
    'pm257davg',\
        #"normpm25",\
        #'hospiprevday','covidpostestprevday','prevdaytotalcovidcasescumulated',\
            'all_day_bing_tiles_visited_relative_change','all_day_ratio_single_tile_users','vac1nb', 'vac2nb',\
                 'Insuffisance respiratoire chronique grave (ALD14)', \
                     'Insuffisance cardiaque grave, troubles du rythme graves, cardiopathies valvulaires graves, cardiopathies congénitales graves (ALD5)',\
                         'Smokers',\
                         "minority",\
                             "Nb_susp_501Y_V1","Nb_susp_501Y_V2_3",
                                 '1MMaxpm25',\
                                 'pm251Mavg'\
                            ]

df["time"]=pd.to_datetime(df["time"])
#df=df[df["time"]>pd.to_datetime("2020-10-20")]
X1=df[['idx', 'pm25', 'no2']]
X2=df[featurespm25]

y= df['newhospi']

# Hold-out
X_train, X_test, y_train, y_test = train_test_split(X1, y, test_size=0.33)
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y, test_size=0.33)
print("\n")

# print(" Scikit Learn ExtratreesRegressor")
# ETregr = ExtraTreesRegressor()
# ETregr.fit(X_train2, y_train2)
# predET = ETregr.predict(X_test2).round(0)
# predETdf = pd.DataFrame(predET)
# predETdf.columns = ["prednewhospi"]
# featuresandtargetdf = X_test2.merge(y_test2, left_on = X_test2.index, right_on = y_test2.index)
# featuresandtargetdf["prednewhospi"]=predETdf["prednewhospi"].round(0)
# featuresandtargetdf.to_csv("../predictions/fr/new_hospi_predictions.csv", index = False)
# ETMSE = mse(y_test2, predET)
# ETMAE = mae(y_test2, predET)

# print("MSE")
# print(ETMSE)
# print("MAE")
# print(ETMAE)
# print("Cross Validation")
# scores = cross_validate(ETregr, X2, y, cv=10,
#                         scoring=('neg_mean_squared_error',"neg_mean_absolute_error"),
#                          return_train_score=True)
# print("MSE")
# print(scores["test_neg_mean_squared_error"].mean())
# print("MAE")
# print(scores["test_neg_mean_absolute_error"].mean())
# print(scores)
# print("\n")
# print("T-Pot exported current best pipeline")
# # Average CV score on the training set was: -94.5319545151712

# GBR1 =  GradientBoostingRegressor(alpha=0.75, learning_rate=0.1, loss="ls", max_depth=8, max_features=0.33, min_samples_leaf=14, min_samples_split=18, n_estimators=100, subsample=0.9500000000000001)

# exported_pipeline = make_pipeline(
#     StandardScaler(),
#     RobustScaler(),
#     GBR1)
#Fix random state for all the steps in exported pipeline
#set_param_recursive(exported_pipeline.steps, 'random_state', 42)
# exported_pipeline.fit(X_train2, y_train2)
# predictions = exported_pipeline.predict(X_test2)
# TPOTMSE = mse(y_test2, predictions)
# TPOTMAE = mae (y_test2, predictions)
# print("MSE:")
# print(TPOTMSE)
# print("MAE:")
# print(TPOTMAE)
# print("Cross Validation")
# scores = cross_validate(exported_pipeline, X2, y, cv=10,
#                         scoring=('neg_mean_squared_error',"neg_mean_absolute_error"),
#                          return_train_score=True)
# print("MSE")
# print(scores["test_neg_mean_squared_error"].mean())
# print("MAE")
# print(scores["test_neg_mean_absolute_error"].mean())
# print(scores)
# print("\n")

# print("Scikit Learn RandomForestRegressor without feature engineering")
# regr = RandomForestRegressor()
# regr.fit(X_train, y_train)
# pred = regr.predict(X_test).round(0)
# RFRMSE = mse(y_test, pred)
# print(RFRMSE)
# print("Average error on new number of hospitalizations per day:", round(RFRMSE ** 0.5,0))
#print("\n")

# print(" Scikit Learn RandomForestRegressor")
# regr2 = RandomForestRegressor()
# regr2.fit(X_train2, y_train2)
# pred2 = regr2.predict(X_test2).round(0)
# RFRMSE2 = mse(y_test2, pred2)
# RFMAE2 = mae(y_test2, pred2)
# print("MSE:")
# print(RFRMSE2)
# print("MAE")
# print(RFMAE2)
# print("Cross Validation")
# scores = cross_validate(regr2, X2, y, cv=10,
#                         scoring=('neg_mean_squared_error',"neg_mean_absolute_error"),
#                          return_train_score=True)
# print("MSE")
# print(scores["test_neg_mean_squared_error"].mean())
# print("MAE")
# print(scores["test_neg_mean_absolute_error"].mean())
# print(scores)
# print("\n")

# print("GradientBoostingRegressor Model")
# model = GradientBoostingRegressor(
#     n_estimators=100, 
#     learning_rate=0.1
# )
# model.fit(X_train2,y_train2)
# pred4 = model.predict(X_test2).round(0)
# MSE4 = mse(y_test2, pred4)
# MAE4 = mae(y_test2, pred4)
# print("MSE:")
# print(MSE4)
# print("MAE:")
# print(MAE4)
# print("Cross Validation")
# scores = cross_validate(regr2, X2, y, cv=10,
#                         scoring=('neg_mean_squared_error',"neg_mean_absolute_error"),
#                          return_train_score=True)
# print("MSE")
# print(scores["test_neg_mean_squared_error"].mean())
# print("MAE")
# print(scores["test_neg_mean_absolute_error"].mean())
# print(scores)
# print("\n")


# print("\n")
# print("DecisionTreeRegressor Model")
# regr2 = DecisionTreeRegressor()
# regr2.fit(X_train2, y_train2)
# pred2 = regr2.predict(X_test2).round(0)
# RFRMSE2 = mse(y_test2, pred2)
# print(RFRMSE2)
# print("Average error on new number of hospitalizations per day:", round(RFRMSE2 ** 0.5,0))

# print("\n")
# print("XGBoost Regressor Model")
# xgb_model = xgb.XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
#              colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
#              importance_type='gain', interaction_constraints='',
#              learning_rate=0.25, max_delta_step=0, max_depth=6,
#              min_child_weight=1, monotone_constraints='()',
#              n_estimators=100, n_jobs=1, num_parallel_tree=1, random_state=0,
#              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
#              tree_method='exact', validate_parameters=1, verbosity=None)

# xgb_model.fit(X_train2, y_train2)
# pred3 = xgb_model.predict(X_test2).round(0)
# RFRMSE3 = mse(y_test2, pred3)
# XGBMAE = mae(y_test2, pred3)
# print("MSE:")
# print(RFRMSE3)
# print("MAE:")
# print(XGBMAE)
# print("Cross Validation")
# scores = cross_validate(xgb_model, X2, y, cv=5,
#                         scoring=('neg_mean_squared_error',"neg_mean_absolute_error"),
#                          return_train_score=True)
# print("MSE")
# print(scores["test_neg_mean_squared_error"].mean())
# print("MAE")
# print(scores["test_neg_mean_absolute_error"].mean())
# print(scores)
# print("\n")

print("VotingRegressor: The Votes of an XGBoost (Extreme Gradient) Regressor VS the votes of a Gradient Boosting Regressor")

GBR1 =  GradientBoostingRegressor(alpha=0.75, learning_rate=0.1, loss="ls", max_depth=8, max_features=0.33, min_samples_leaf=14, min_samples_split=18, n_estimators=100, subsample=0.9500000000000001)
exported_pipeline = make_pipeline(
    StandardScaler(),
    RobustScaler(),
    GBR1)

xgb_model = xgb.XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
             importance_type='gain', interaction_constraints='',
             learning_rate=0.25, max_delta_step=0, max_depth=6,
             min_child_weight=1, monotone_constraints='()',
             n_estimators=100, n_jobs=1, num_parallel_tree=1, random_state=0,
             reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
             tree_method='exact', validate_parameters=1, verbosity=None)

ensemble = VotingRegressor(
    estimators = [("TPET",exported_pipeline),("xgbr",xgb_model)],
   )

ensemble.fit(X_train2, y_train2)
predvot = ensemble.predict(X_test2).round(0)
MSE5 = mse(y_test2,predvot)
MAE5 = mae(y_test2,predvot)
print("MSE:")
print(MSE5)
print("MAE:")
print(MAE5)
print("Cross Validation")
scores = cross_validate(ensemble, X2, y, cv=5,
                        scoring=('neg_mean_squared_error',"neg_mean_absolute_error"),
                         return_train_score=True)
print("MSE:")
print(scores["test_neg_mean_squared_error"].mean())
print("MAE:")
print(scores["test_neg_mean_absolute_error"].mean())
print("Cross-val Scores")
print(scores)
print("\n")

xgb_model.fit(X_train2, y_train2)
GBR1.fit(X_train2, y_train2)
# plot
plt.bar(range(len(xgb_model.feature_importances_)), xgb_model.feature_importances_)
plt.show(block=True)
print("XGBoost Feature importance report:")
FIlist = xgb_model.feature_importances_.tolist()
FIlistdf = pd.DataFrame(FIlist)
FIlistdf = FIlistdf.T
FIlistdf.columns = featurespm25
FIlistdf = FIlistdf.T
FIlistdf.columns = ["feature_importance"]
FIlistdf.sort_values(by = ["feature_importance"], inplace = True, ascending = False)
print(FIlistdf)
FIlistdf.to_csv("../feature_importance/XGBoost Regressor Feature importance report.csv", sep = ";")
print("\n")

print("\n")
# plot
plt.bar(range(len(GBR1.feature_importances_)), GBR1.feature_importances_)
plt.show(block=True)
print("SCIKIT Learn's Gradient Boosting Regressor Feature Importance report:")
FIlist = GBR1.feature_importances_.tolist()
FIlistdf = pd.DataFrame(FIlist)
FIlistdf = FIlistdf.T
FIlistdf.columns = featurespm25
FIlistdf = FIlistdf.T
FIlistdf.columns = ["feature_importance"]
FIlistdf.sort_values(by = ["feature_importance"], inplace = True, ascending = False)
print(FIlistdf)
print("\n")
FIlistdf.to_csv("../feature_importance/SCIKIT Learn's Gradient Boosting Regressor Feature Importance report.csv", sep = ';')
#Save model to .joblib file
# save the model to disk
filename = '../model/model.joblib'
joblib.dump(ensemble, filename)
 
# # some time later...
 
# # load the model from disk
# loaded_model = joblib.load(filename)
# result = loaded_model.score(X_test, Y_test)
# print(result)



print("\n")
print("TPOTRegressor")
tpot = TPOTRegressor(generations=1, population_size=50, verbosity=2, random_state=42)
tpot.fit(X2, y)
#print(tpot.score(X, y_test2))
tpot.export('tpot_covid_pipeline.py')

print("\n")
print("Neural Network")
X_trainNN = X_train2.values.reshape(X_train2.shape[0], X_train2.shape[1], 1)
y_trainNN = y_train2.values
X_testNN = X_test2.values.reshape(X_test2.shape[0],X_test2.shape[1],1)
y_testNN = y_test2.values
NNmodel = Sequential()
#NNmodel.add(layers.Dense(215, input_shape=(X_trainNN.shape[0], X_trainNN.shape[1])))
NNmodel.add(layers.LSTM(units=22, activation='tanh',return_sequences=True, input_shape=X_trainNN.shape[1:]))
NNmodel.add(layers.LSTM(units=10, activation='tanh', return_sequences=False))
NNmodel.add(layers.Dense(1, activation="linear"))

# The compilation
NNmodel.compile(loss='mse', 
              optimizer='rmsprop')

es = callbacks.EarlyStopping(patience=30, restore_best_weights=True)

# The fit
NNmodel.fit(X_trainNN, y_trainNN,
         batch_size=16, validation_split = 0.3,
         epochs=100, verbose=1,callbacks=[es])

# The prediction
print("MSE:")
print(NNmodel.evaluate(X_testNN, y_testNN, verbose=0))

#print('validation loss (MSE):', val_loss, '\n validation MAE:', val_mae)
#print("Average error on new number of hospitalizations per day:", round(val_mae ** 0.5,0))