#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =============================================================================
# Imports #%%
# =============================================================================
#part| #%%
import numpy as np
import pandas as pd
from datetime import timedelta
from datetime import datetime
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import VotingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.svm import LinearSVR
from sklearn.preprocessing import MinMaxScaler
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

class maintrain():

    def __init__(self, skipcrossval, skipimpute, skiptpot):

        self.df = None
        self.modelday0features = None
        self.modelday1features = None
        self.modelday2features = None
        self.modelday3features = None
        self.modelday4features = None
        self.featuresandtarget = None
        self.X = None
        self.y = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.predictiondata = None
        self.featuresandtargetforanalysis = None
        self.skipcrossval=skipcrossval
        self.skipimpute = skipimpute
        self.skiptpot = skiptpot

    def max_normalize(self, x):
        return (x - x.min()) / (x.max() - x.min())

    def mse(self, ground_truth, predictions):
        diff = (ground_truth - predictions)**2
        return diff.mean()

    def mae(self, ground_truth, predictions):
        diff = abs(ground_truth - predictions)
        return diff.mean()

    def initdata(self):
        self.df = pd.read_csv("/home/ludo915/code/covsco/data/train/all_data_merged/fr/traindfknnimputed.csv")
        self.df["date"]=pd.to_datetime(self.df["date"])
        self.df = self.df[self.df["date"]> pd.to_datetime("2020-05-13")]
        self.df["all_day_bing_tiles_visited_relative_change"]=self.df["all_day_bing_tiles_visited_relative_change"].astype(float)
        self.df["all_day_ratio_single_tile_users"]=self.df["all_day_ratio_single_tile_users"].astype(float)
        print(self.df)
        self.featuresandtargetforanalysis = ['idx', 'pm25', 'no2',\
        'o3','pm10','co','so2',\
            'pm257davg','no27davg',\
            'o37davg','co7davg', 'pm107davg','so27davg',\
                "normpm25","normno2","normo3","normpm10","normco","normso2",\
                'hospiprevday','covidpostestprevday','prevdaytotalcovidcasescumulated','CovidPosTest',\
                "newhospi7davg","newreanim7davg","newhospi3davg","newreanim3davg",\
                    'all_day_bing_tiles_visited_relative_change','all_day_ratio_single_tile_users','vac1nb', 'vac2nb',\
                        'Insuffisance respiratoire chronique grave (ALD14)', \
                            'Insuffisance cardiaque grave, troubles du rythme graves, cardiopathies valvulaires graves, cardiopathies congénitales graves (ALD5)',\
                                'Smokers',\
                                    "minority",\
                                        "Nb_susp_501Y_V1","Nb_susp_501Y_V2_3",\
                                            '1MMaxpm25','1MMaxpm10','1MMaxo3','1MMaxno2','1MMaxco','1MMaxso2',\
                                                'pm251Mavg','no21Mavg','o31Mavg','pm101Mavg','co1Mavg','so21Mavg',\
                                                    '1MMaxnormpm25','1MMaxnormno2','1MMaxnormo3','1MMaxnormpm10','1MMaxnormco','1MMaxnormso2',\
                                                        "normpm257davg","normno27davg","normo37davg","normpm107davg","normco7davg","normso27davg",\
                                                                "normpm251Mavg","normno21Mavg","normo31Mavg","normpm101Mavg","normco1Mavg","normso21Mavg",\
                                                                'newhospi','newreanim',"newhospinextday"]

        # self.featuresandtarget = ['idx', 'pm25', 'no2','o3','pm10','co','so2',\
        #     'pm257davg','no27davg','o37davg','co7davg', 'pm107davg','so27davg',\
        #         'pm251Mavg','no21Mavg','o31Mavg','pm101Mavg','co1Mavg','so21Mavg',\
        #             '1MMaxpm25','1MMaxpm10','1MMaxo3','1MMaxno2','1MMaxco','1MMaxso2',\
        #                 'hospi','newhospi','CovidPosTest',\
        #                     'all_day_bing_tiles_visited_relative_change','all_day_ratio_single_tile_users',\
        #                         'vac1nb', 'vac2nb',\
        #                             'Insuffisance respiratoire chronique grave (ALD14)', \
        #                                 'Insuffisance cardiaque grave, troubles du rythme graves, cardiopathies valvulaires graves, cardiopathies congénitales graves (ALD5)',\
        #                                     'Smokers',\
        #                                         "minority","pauvrete","rsa","ouvriers",\
        #                                             "Nb_susp_501Y_V1","Nb_susp_501Y_V2_3",\
        #                                                 "newhospinextday"]
        
        self.featuresandtarget = ['idx', \
        # 'pm25', 'no2','o3','pm10','co','so2',\
        #     'pm257davg','no27davg','o37davg','co7davg', 'pm107davg','so27davg',\
        #         'pm251Mavg','no21Mavg','o31Mavg','pm101Mavg','co1Mavg','so21Mavg',\
        #             '1MMaxpm25','1MMaxpm10','1MMaxo3','1MMaxno2','1MMaxco','1MMaxso2',\
                        'hospi','newhospi','CovidPosTest',\
                            'all_day_bing_tiles_visited_relative_change','all_day_ratio_single_tile_users',\
                                'vac1nb', 'vac2nb',\
                                    'Insuffisance respiratoire chronique grave (ALD14)', \
                                        'Insuffisance cardiaque grave, troubles du rythme graves, cardiopathies valvulaires graves, cardiopathies congénitales graves (ALD5)',\
                                            'Smokers',\
                                                "minority","pauvrete","rsa","ouvriers",\
                                                    "Nb_susp_501Y_V1","Nb_susp_501Y_V2_3",\
                                                        "newhospinextday"]
        

        
                                
        self.modelday0features = ['idx',\
        'pm25', \
              'no2',
            'o3',\
            'pm10',
            'co',\
            'so2',\
             'pm257davg', \
                 'no27davg',\
            'o37davg',\
               'co7davg', \
            'pm107davg','so27davg',\
                 'pm251Mavg',
                 'no21Mavg',
                 'o31Mavg',
                'pm101Mavg',
                 'co1Mavg',
                 'so21Mavg',\
                    '1MMaxpm25',\
                    '1MMaxpm10',\
                    '1MMaxo3',
                    '1MMaxno2',
                    '1MMaxco',\
                        '1MMaxso2',\
                       # 'hospi','newhospi','CovidPosTest',\
                            'all_day_bing_tiles_visited_relative_change','all_day_ratio_single_tile_users',\
                                'vac1nb', 'vac2nb',\
                                    'Insuffisance respiratoire chronique grave (ALD14)', \
                                        'Insuffisance cardiaque grave, troubles du rythme graves, cardiopathies valvulaires graves, cardiopathies congénitales graves (ALD5)',\
                                            'Smokers',\
                                                "minority","pauvrete","rsa","ouvriers",\
                                                    "Nb_susp_501Y_V1","Nb_susp_501Y_V2_3",\
                                                        "pm252001","pm252002","pm252003","pm252004","pm252005","pm252006","pm252007","pm252008",\
            "pm252009","pm252010","pm252011","pm252012","pm252013","pm252014","pm252015","pm252016","pm252017","pm252018"\
                ]

        self.modelday1features = ['idx', "dateiminusonepm25dayiforecast","dateiminusoneno2dayiforecast","dateiminusoneo3dayiforecast",\
                                            "dateiminusonepm10dayiforecast","dateiminusonecodayiforecast","dateiminusoneso2dayiforecast",\
                                                "dateiminusonepm25dayiforecast7davg","dateiminusoneno2dayiforecast7davg",\
                                                    "dateiminusoneo3dayiforecast7davg","dateiminusonepm10dayiforecast7davg",\
                                                        "dateiminusonecodayiforecast7davg","dateiminusoneso2dayiforecast7davg",\
                                                        "dateiminusonepm25dayiforecast1Mavg","dateiminusoneno2dayiforecast1Mavg",\
                                                        "dateiminusoneo3dayiforecast1Mavg","dateiminusonepm10dayiforecast1Mavg",\
                                                        "dateiminusonecodayiforecast1Mavg","dateiminusoneso2dayiforecast1Mavg",\
                                                        "dateiminusonepm25dayiforecast1MMax","dateiminusoneno2dayiforecast1MMax",\
                                                        "dateiminusoneo3dayiforecast1MMax","dateiminusonepm10dayiforecast1MMax",\
                                                        "dateiminusonecodayiforecast1MMax","dateiminusoneso2dayiforecast1MMax",\
                                                        'dateiminusonehospi','dateiminusonenewhospi',"dateiminusonecovidpostest",\
                                                            'dateiminusonefbmobility2','dateiminusonefbmobility1',\
                                                                'dateiminusonevac1nb', 'dateiminusonevac2nb',\
                                                                    'Insuffisance respiratoire chronique grave (ALD14)', \
                                                                        'Insuffisance cardiaque grave, troubles du rythme graves, cardiopathies valvulaires graves, cardiopathies congénitales graves (ALD5)',\
                                                                            'Smokers',\
                                                                                "minority","pauvrete","rsa","ouvriers",\
                                                                                    "dateiminusoneNb_susp_501Y_V1","dateiminusoneNb_susp_501Y_V2_3"\
                                                                                        ]
       
        self.modelday2features = ['idx',  "dateiminustwopm25dayiforecast","dateiminustwono2dayiforecast","dateiminustwoo3dayiforecast",\
                                            "dateiminustwopm10dayiforecast","dateiminustwocodayiforecast","dateiminustwoso2dayiforecast",\
        "dateiminustwopm25dayiforecast7davg",\
                "dateiminustwono2dayiforecast7davg",\
                "dateiminustwoo3dayiforecast7davg",\
                "dateiminustwopm10dayiforecast7davg",\
                "dateiminustwocodayiforecast7davg",\
                "dateiminustwoso2dayiforecast7davg",\
                "dateiminustwopm25dayiforecast1Mavg",\
                "dateiminustwono2dayiforecast1Mavg",\
                "dateiminustwoo3dayiforecast1Mavg",\
                "dateiminustwopm10dayiforecast1Mavg",\
                "dateiminustwocodayiforecast1Mavg",\
                "dateiminustwoso2dayiforecast1Mavg",\
                "dateiminustwopm25dayiforecast1MMax",\
                "dateiminustwono2dayiforecast1MMax",\
                "dateiminustwoo3dayiforecast1MMax",\
                "dateiminustwopm10dayiforecast1MMax",\
                "dateiminustwocodayiforecast1MMax",\
                "dateiminustwoso2dayiforecast1MMax",\
                'dateiminustwohospi','dateiminustwonewhospi',"dateiminustwocovidpostest",\
                                                            'dateiminustwofbmobility2','dateiminustwofbmobility1',\
                                                                'dateiminustwovac1nb', 'dateiminustwovac2nb',\
                                                                    'Insuffisance respiratoire chronique grave (ALD14)', \
                                                                        'Insuffisance cardiaque grave, troubles du rythme graves, cardiopathies valvulaires graves, cardiopathies congénitales graves (ALD5)',\
                                                                            'Smokers',\
                                                                                "minority","pauvrete","rsa","ouvriers",\
                                                                                    "dateiminustwoNb_susp_501Y_V1","dateiminustwoNb_susp_501Y_V2_3"\
        ]  
        self.modelday3features = ['idx',"dateiminusthreepm25dayiforecast","dateiminusthreeno2dayiforecast","dateiminusthreeo3dayiforecast",\
                                    "dateiminusthreepm10dayiforecast","dateiminusthreecodayiforecast","dateiminusthreeso2dayiforecast",\
                                        "dateiminusthreepm25dayiforecast7davg","dateiminusthreeno2dayiforecast7davg",\
                                            "dateiminusthreeo3dayiforecast7davg","dateiminusthreepm10dayiforecast7davg",\
                                                "dateiminusthreecodayiforecast7davg","dateiminusthreeso2dayiforecast7davg",\
                                                    "dateiminusthreepm25dayiforecast1Mavg","dateiminusthreeno2dayiforecast1Mavg",\
                                                        "dateiminusthreeo3dayiforecast1Mavg","dateiminusthreepm10dayiforecast1Mavg",\
                                                            "dateiminusthreecodayiforecast1Mavg","dateiminusthreeso2dayiforecast1Mavg",\
                                                                "dateiminusthreepm25dayiforecast1MMax","dateiminusthreeno2dayiforecast1MMax",\
                                                                    "dateiminusthreeo3dayiforecast1MMax","dateiminusthreepm10dayiforecast1MMax",\
                                                                     "dateiminusthreecodayiforecast1MMax","dateiminusthreeso2dayiforecast1MMax",\
                                    'dateiminusthreehospi','dateiminusthreenewhospi',"dateiminusthreecovidpostest",\
                                                            'dateiminusthreefbmobility2','dateiminusthreefbmobility1',\
                                                                'dateiminusthreevac1nb', 'dateiminusthreevac2nb',\
                                                                    'Insuffisance respiratoire chronique grave (ALD14)', \
                                                                        'Insuffisance cardiaque grave, troubles du rythme graves, cardiopathies valvulaires graves, cardiopathies congénitales graves (ALD5)',\
                                                                            'Smokers',\
                                                                                "minority","pauvrete","rsa","ouvriers",\
                                                                                    "dateiminusthreeNb_susp_501Y_V1","dateiminusthreeNb_susp_501Y_V2_3"\
        ]
        self.modelday4features = ['idx',\
                "dateiminusfourpm25dayiforecast",\
                "dateiminusfourno2dayiforecast",\
                "dateiminusfouro3dayiforecast",\
                "dateiminusfourpm10dayiforecast",\
                "dateiminusfourcodayiforecast",\
                "dateiminusfourso2dayiforecast",
                "dateiminusfourpm25dayiforecast7davg",\
                "dateiminusfourno2dayiforecast7davg",\
                "dateiminusfouro3dayiforecast7davg",\
                "dateiminusfourpm10dayiforecast7davg",\
                "dateiminusfourcodayiforecast7davg",\
                "dateiminusfourso2dayiforecast7davg",\
                "dateiminusfourpm25dayiforecast1Mavg",\
                "dateiminusfourno2dayiforecast1Mavg",\
                "dateiminusfouro3dayiforecast1Mavg",\
                "dateiminusfourpm10dayiforecast1Mavg",\
                "dateiminusfourcodayiforecast1Mavg",\
                "dateiminusfourso2dayiforecast1Mavg",\
                "dateiminusfourpm25dayiforecast1MMax",\
                "dateiminusfourno2dayiforecast1MMax",\
                "dateiminusfouro3dayiforecast1MMax",\
                "dateiminusfourpm10dayiforecast1MMax",\
                "dateiminusfourcodayiforecast1MMax",\
                "dateiminusfourso2dayiforecast1MMax",\
                'dateiminusfourhospi','dateiminusfournewhospi',"dateiminusfourcovidpostest",\
                                                            'dateiminusfourfbmobility2','dateiminusfourfbmobility1',\
                                                                'dateiminusfourvac1nb', 'dateiminusfourvac2nb',\
                                                                    'Insuffisance respiratoire chronique grave (ALD14)', \
                                                                        'Insuffisance cardiaque grave, troubles du rythme graves, cardiopathies valvulaires graves, cardiopathies congénitales graves (ALD5)',\
                                                                            'Smokers',\
                                                                                "minority","pauvrete","rsa","ouvriers",\
                                                                                    "dateiminusfourNb_susp_501Y_V1","dateiminusfourNb_susp_501Y_V2_3"\
        ]

        if  self.skipimpute == None:   
            self.df = self.df.replace("Nan", np.nan)
            self.df.fillna( 9999, inplace = True )
            
            print("KNNImputing NaN values for features and target for analysis ...")
            imputer = KNNImputer(missing_values= 9999, n_neighbors=5, weights='uniform', metric='nan_euclidean')
            imputer.fit(self.df[self.featuresandtargetforanalysis])
            self.df[self.featuresandtargetforanalysis] = imputer.transform(self.df[self.featuresandtargetforanalysis])
            
            print("KNNImputing NaN values for model day 1 features ...")
            imputer = KNNImputer(missing_values=9999, n_neighbors=5, weights='uniform', metric='nan_euclidean')
            imputer.fit(self.df[self.modelday1features])
            self.df[self.modelday1features] = imputer.transform(self.df[self.modelday1features])

            print("KNNImputing NaN values for model day 2 features ...")
            imputer = KNNImputer(missing_values=9999, n_neighbors=5, weights='uniform', metric='nan_euclidean')
            imputer.fit(self.df[self.modelday2features])
            self.df[self.modelday2features] = imputer.transform(self.df[self.modelday2features])

            print("KNNImputing NaN values for model day 3 features ...")
            imputer = KNNImputer(missing_values=9999, n_neighbors=5, weights='uniform', metric='nan_euclidean')
            imputer.fit(self.df[self.modelday3features])
            self.df[self.modelday3features] = imputer.transform(self.df[self.modelday3features])
    
            print("KNNImputing NaN values for model day 4 features ...")
            imputer = KNNImputer(missing_values=9999, n_neighbors=5, weights='uniform', metric='nan_euclidean')
            imputer.fit(self.df[self.modelday4features])
            self.df[self.modelday4features] = imputer.transform(self.df[self.modelday4features])

            self.df.to_csv("/home/ludo915/code/covsco/data/train/all_data_merged/fr/traindfknnimputed.csv", index = False)
        
        else:

            self.df = pd.read_csv("/home/ludo915/code/covsco/data/train/all_data_merged/fr/traindfknnimputed.csv")

        return None

    def HoldOut(self, features, model):
        print("Proceding to Hold-Out Method")
        self.df["date"]=pd.to_datetime(self.df["date"])
        self.X=self.df[features]
        if model == 1:
            self.X.rename(columns = {"dateiminusonepm25dayiforecast":"pm25","dateiminusoneno2dayiforecast":"no2","dateiminusoneo3dayiforecast":"o3",\
                                            "dateiminusonepm10dayiforecast":"pm10","dateiminusonecodayiforecast":"co","dateiminusoneso2dayiforecast":"so2",\
                                                "dateiminusonepm25dayiforecast7davg":"pm257davg","dateiminusoneno2dayiforecast7davg":"no27davg",\
                                                    "dateiminusoneo3dayiforecast7davg":"o37davg","dateiminusonepm10dayiforecast7davg":"pm107davg",\
                                                        "dateiminusonecodayiforecast7davg":"co7davg","dateiminusoneso2dayiforecast7davg":"so27davg",\
                                                        "dateiminusonepm25dayiforecast1Mavg":"pm251Mavg","dateiminusoneno2dayiforecast1Mavg":"no21Mavg",\
                                                        "dateiminusoneo3dayiforecast1Mavg":"o31Mavg","dateiminusonepm10dayiforecast1Mavg":"pm101Mavg",\
                                                        "dateiminusonecodayiforecast1Mavg":"co1Mavg","dateiminusoneso2dayiforecast1Mavg":"so21Mavg",\
                                                        "dateiminusonepm25dayiforecast1MMax":"1MMaxpm25","dateiminusoneno2dayiforecast1MMax":"1MMaxno2",\
                                                        "dateiminusoneo3dayiforecast1MMax":"1MMaxo3","dateiminusonepm10dayiforecast1MMax":"1MMaxpm10",\
                                                        "dateiminusonecodayiforecast1MMax":"1MMaxco","dateiminusoneso2dayiforecast1MMax":"1MMaxso2",\
                                                        'dateiminusonehospi':'hospi','dateiminusonenewhospi':'newhospi',"dateiminusonecovidpostest":"CovidPosTest",\
                                                            'dateiminonefbmobility2':'all_day_bing_tiles_visited_relative_change','dateiminonefbmobility1':'all_day_ratio_single_tile_users',\
                                                                'dateiminusonevac1nb':'vac1nb', 'dateiminusonevac2nb':'vac2nb',\
                                                                "dateiminusoneNb_susp_501Y_V1":"Nb_susp_501Y_V1","dateiminusoneNb_susp_501Y_V2_3":"Nb_susp_501Y_V2_3"})
        if model == 2:
            self.X.rename(columns = {"dateiminustwopm25dayiforecast":"pm25","dateiminustwono2dayiforecast":"no2","dateiminustwoo3dayiforecast":"o3",\
                                            "dateiminustwopm10dayiforecast":"pm10","dateiminustwocodayiforecast":"co","dateiminustwoso2dayiforecast":"so2",\
                                                "dateiminustwopm25dayiforecast7davg":"pm257davg","dateiminustwono2dayiforecast7davg":"no27davg",\
                                                    "dateiminustwoo3dayiforecast7davg":"o37davg","dateiminustwopm10dayiforecast7davg":"pm107davg",\
                                                        "dateiminustwocodayiforecast7davg":"co7davg","dateiminustwoso2dayiforecast7davg":"so27davg",\
                                                        "dateiminustwopm25dayiforecast1Mavg":"pm251Mavg","dateiminustwono2dayiforecast1Mavg":"no21Mavg",\
                                                        "dateiminustwoo3dayiforecast1Mavg":"o31Mavg","dateiminustwopm10dayiforecast1Mavg":"pm101Mavg",\
                                                        "dateiminustwocodayiforecast1Mavg":"co1Mavg","dateiminustwoso2dayiforecast1Mavg":"so21Mavg",\
                                                        "dateiminustwopm25dayiforecast1MMax":"1MMaxpm25","dateiminustwono2dayiforecast1MMax":"1MMaxno2",\
                                                        "dateiminustwoo3dayiforecast1MMax":"1MMaxo3","dateiminustwopm10dayiforecast1MMax":"1MMaxpm10",\
                                                        "dateiminustwocodayiforecast1MMax":"1MMaxco","dateiminustwoso2dayiforecast1MMax":"1MMaxso2",\
                                                        'dateiminustwohospi':'hospi','dateiminustwonewhospi':'newhospi',"dateiminustwocovidpostest":"CovidPosTest",\
                                                            'dateimintwofbmobility2':'all_day_bing_tiles_visited_relative_change','dateimintwofbmobility1':'all_day_ratio_single_tile_users',\
                                                                'dateiminustwovac1nb':'vac1nb', 'dateiminustwovac2nb':'vac2nb',\
                                                                                    "dateiminustwoNb_susp_501Y_V1":"Nb_susp_501Y_V1","dateiminustwoNb_susp_501Y_V2_3":"Nb_susp_501Y_V2_3"})

        if model == 3:
            self.X.rename(columns = {"dateiminusthreepm25dayiforecast":"pm25","dateiminusthreeno2dayiforecast":"no2","dateiminusthreeo3dayiforecast":"o3",\
                                            "dateiminusthreepm10dayiforecast":"pm10","dateiminusthreecodayiforecast":"co","dateiminusthreeso2dayiforecast":"so2",\
                                                "dateiminusthreepm25dayiforecast7davg":"pm257davg","dateiminusthreeno2dayiforecast7davg":"no27davg",\
                                                    "dateiminusthreeo3dayiforecast7davg":"o37davg","dateiminusthreepm10dayiforecast7davg":"pm107davg",\
                                                        "dateiminusthreecodayiforecast7davg":"co7davg","dateiminusthreeso2dayiforecast7davg":"so27davg",\
                                                        "dateiminusthreepm25dayiforecast1Mavg":"pm251Mavg","dateiminusthreeno2dayiforecast1Mavg":"no21Mavg",\
                                                        "dateiminusthreeo3dayiforecast1Mavg":"o31Mavg","dateiminusthreepm10dayiforecast1Mavg":"pm101Mavg",\
                                                        "dateiminusthreecodayiforecast1Mavg":"co1Mavg","dateiminusthreeso2dayiforecast1Mavg":"so21Mavg",\
                                                        "dateiminusthreepm25dayiforecast1MMax":"1MMaxpm25","dateiminusthreeno2dayiforecast1MMax":"1MMaxno2",\
                                                        "dateiminusthreeo3dayiforecast1MMax":"1MMaxo3","dateiminusthreepm10dayiforecast1MMax":"1MMaxpm10",\
                                                        "dateiminusthreecodayiforecast1MMax":"1MMaxco","dateiminusthreeso2dayiforecast1MMax":"1MMaxso2",\
                                                        'dateiminusthreehospi':'hospi','dateiminusthreenewhospi':'newhospi',"dateiminusthreecovidpostest":"CovidPosTest",\
                                                            'dateiminthreefbmobility2':'all_day_bing_tiles_visited_relative_change','dateiminthreefbmobility1':'all_day_ratio_single_tile_users',\
                                                                'dateiminusthreevac1nb':'vac1nb', 'dateiminusthreevac2nb':'vac2nb',\
                                                                                    "dateiminusthreeNb_susp_501Y_V1":"Nb_susp_501Y_V1","dateiminusthreeNb_susp_501Y_V2_3":"Nb_susp_501Y_V2_3"})

        if model == 4:
            self.X.rename(columns = {"dateiminusfourpm25dayiforecast":"pm25","dateiminusfourno2dayiforecast":"no2","dateiminusfouro3dayiforecast":"o3",\
                                            "dateiminusfourpm10dayiforecast":"pm10","dateiminusfourcodayiforecast":"co","dateiminusfourso2dayiforecast":"so2",\
                                                "dateiminusfourpm25dayiforecast7davg":"pm257davg","dateiminusfourno2dayiforecast7davg":"no27davg",\
                                                    "dateiminusfouro3dayiforecast7davg":"o37davg","dateiminusfourpm10dayiforecast7davg":"pm107davg",\
                                                        "dateiminusfourcodayiforecast7davg":"co7davg","dateiminusfourso2dayiforecast7davg":"so27davg",\
                                                        "dateiminusfourpm25dayiforecast1Mavg":"pm251Mavg","dateiminusfourno2dayiforecast1Mavg":"no21Mavg",\
                                                        "dateiminusfouro3dayiforecast1Mavg":"o31Mavg","dateiminusfourpm10dayiforecast1Mavg":"pm101Mavg",\
                                                        "dateiminusfourcodayiforecast1Mavg":"co1Mavg","dateiminusfourso2dayiforecast1Mavg":"so21Mavg",\
                                                        "dateiminusfourpm25dayiforecast1MMax":"1MMaxpm25","dateiminusfourno2dayiforecast1MMax":"1MMaxno2",\
                                                        "dateiminusfouro3dayiforecast1MMax":"1MMaxo3","dateiminusfourpm10dayiforecast1MMax":"1MMaxpm10",\
                                                        "dateiminusfourcodayiforecast1MMax":"1MMaxco","dateiminusfourso2dayiforecast1MMax":"1MMaxso2",\
                                                        'dateiminusfourhospi':'hospi','dateiminusfournewhospi':'newhospi',"dateiminusfourcovidpostest":"CovidPosTest",\
                                                            'dateiminfourfbmobility2':'all_day_bing_tiles_visited_relative_change','dateiminfourfbmobility1':'all_day_ratio_single_tile_users',\
                                                                'dateiminusfourvac1nb':'vac1nb', 'dateiminusfourvac2nb':'vac2nb',\
                                                        "dateiminusfourNb_susp_501Y_V1":"Nb_susp_501Y_V1","dateiminusfourNb_susp_501Y_V2_3":"Nb_susp_501Y_V2_3"})
       
        self.y= self.df['newhospinextday']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.33,random_state=84)
        print("\n")
        return None

    def CurrentBestModel(self):
        
        featurelist = [self.modelday0features,self.modelday0features,self.modelday0features,self.modelday0features,self.modelday0features]
        counter = 0
        for features in featurelist:
            self.HoldOut(features, counter)
            print("Model for day ",counter)
            #print("VotingRegressor: The Votes of an XGBoost (Extreme Gradient) Regressor VS the votes of a Gradient Boosting Regressor")
            print("Gradient Boosting Regressor:")
            GBR1 = GradientBoostingRegressor(alpha=0.9, learning_rate=0.1, loss="huber", max_depth=8, max_features=1.0, min_samples_leaf=19, min_samples_split=11, n_estimators=100, subsample=0.9500000000000001)
            exported_pipeline = make_pipeline(
            StackingEstimator(estimator=LinearSVR(C=10.0, dual=False, epsilon=0.1, loss="squared_epsilon_insensitive", tol=0.001)),\
                              MinMaxScaler(),\
                              GBR1)\

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

            ensemble.fit(self.X_train, self.y_train)
            predvot = ensemble.predict(self.X_test).round(0)
            MSE5 = self.mse(self.y_test,predvot)
            MAE5 = self.mae(self.y_test,predvot)
            print("MSE:")
            print(MSE5)
            print("MAE:")
            print(MAE5)

            # GBR1.fit(self.X_train, self.y_train)
            # predvot = GBR1.predict(self.X_test).round(0)
            # MSE5 = self.mse(self.y_test,predvot)
            # MAE5 = self.mae(self.y_test,predvot)
            # print("MSE:")
            # print(MSE5)
            # print("MAE:")
            # print(MAE5)

            if self.skipcrossval == None:
                print("Cross Validation")
                scores = cross_validate(ensemble, self.X, self.y, cv=5,
                                        scoring=('neg_mean_squared_error',"neg_mean_absolute_error"),
                                        return_train_score=True)
                print("MSE:")
                print(scores["test_neg_mean_squared_error"].mean())
                print("MAE:")
                print(scores["test_neg_mean_absolute_error"].mean())
                print("Cross-val Scores")
                print(scores)
                print("\n")

            xgb_model.fit(self.X, self.y)
            GBR1.fit(self.X, self.y)
            # plot
            plt.bar(range(len(xgb_model.feature_importances_)), xgb_model.feature_importances_)
            plt.show(block=True)
            print("XGBoost Feature importance report:")
            FIlist = xgb_model.feature_importances_.tolist()
            FIlistdf = pd.DataFrame(FIlist)
            FIlistdf = FIlistdf.T
            FIlistdf.columns = features
            FIlistdf = FIlistdf.T
            FIlistdf.columns = ["feature_importance"]
            FIlistdf.sort_values(by = ["feature_importance"], inplace = True, ascending = False)
            print(FIlistdf)
            FIlistdf.to_csv("/home/ludo915/code/covsco/feature_importance/XGBoost Regressor Feature importance report: model day "+ str(counter) + ".csv", sep = ";")
            print("\n")

            print("\n")
            # plot
            plt.bar(range(len(GBR1.feature_importances_)), GBR1.feature_importances_)
            plt.show(block=True)
            print("SCIKIT Learn's Gradient Boosting Regressor Feature Importance report:")
            FIlist = GBR1.feature_importances_.tolist()
            FIlistdf = pd.DataFrame(FIlist)
            FIlistdf = FIlistdf.T
            FIlistdf.columns = features
            FIlistdf = FIlistdf.T
            FIlistdf.columns = ["feature_importance"]
            FIlistdf.sort_values(by = ["feature_importance"], inplace = True, ascending = False)
            print(FIlistdf)
            print("\n")
            FIlistdf.to_csv("/home/ludo915/code/covsco/feature_importance/SCIKIT Learn's Gradient Boosting Regressor Feature Importance report: model day "+ str(counter) + ".csv", sep = ';')
            filename = '/home/ludo915/code/covsco/model/model_day_'+ str(counter) + '.joblib'
            joblib.dump(ensemble, filename)
         
            print("\n")
            if self.skiptpot == None:
                self.tpotregressor(counter)
            counter += 1

        return None

    def tpotregressor(self, i):
        print("TPOTRegressor")
        tpot = TPOTRegressor(generations=50, population_size=50, verbosity=2, random_state=42)
        tpot.fit(self.X, self.y)
        #print(tpot.score(X, y_test2))
        tpot.export('tpot_covid_pipeline_day_' + str(i) + '.py')
        print("\n")
        return None

    def neural_network(self):
        print("Neural Network")
        X_trainNN = self.X_train.values.reshape(self.X_train.shape[0], self.X_train.shape[1], 1)
        y_trainNN = self.y_train.values
        X_testNN = self.X_test.values.reshape(self.X_test.shape[0],self.X_test.shape[1],1)
        y_testNN = self.y_test.values
        NNmodel = Sequential()
        NNmodel.add(layers.LSTM(units=22, activation='tanh',return_sequences=True, input_shape=X_trainNN.shape[1:]))
        NNmodel.add(layers.LSTM(units=10, activation='tanh', return_sequences=False))
        NNmodel.add(layers.Dense(1, activation="linear"))
        NNmodel.compile(loss='mse', optimizer='rmsprop')
        es = callbacks.EarlyStopping(patience=30, restore_best_weights=True)
        NNmodel.fit(X_trainNN, y_trainNN,batch_size=16, validation_split = 0.3, epochs=100, verbose=1,callbacks=[es])
        print("MSE:")
        print(NNmodel.evaluate(X_testNN, y_testNN, verbose=0))
        return None

    def predict(self):
        for i in range(5):
            #currentDate = datetime.today().strftime('%Y-%m-%d')
            
            filename = '/home/ludo915/code/covsco/model/model_day_'+ str(i) + '.joblib'
            loaded_model = joblib.load(filename)
            Xpredictdf = pd.read_csv('/home/ludo915/code/covsco/predictions/fr/data/day'+str(i)+'df.csv')
            currentDate = (pd.to_datetime(Xpredictdf["date"].max())+pd.Timedelta("1 Days")).strftime('%Y-%m-%d')
         
            print(Xpredictdf)
            Xpredict = Xpredictdf.drop(columns = ["date","numero"])
            Xpredict = Xpredict[self.modelday0features]
            newhospipredictions = pd.DataFrame(loaded_model.predict(Xpredict))
            newhospipredictions.columns = ["newhospipred"]
            newhospipredictions["date"] = Xpredictdf["date"]
            newhospipredictions["depnum"] = Xpredictdf["numero"]
            newhospipredictions.to_csv("/home/ludo915/code/covsco/predictions/fr/" + currentDate + "_predictions_for_day_"+ str(i) + '.csv', index = False)
        return None
            

        # print(" Scikit Learn ExtratreesRegressor")
        # ETregr = ExtraTreesRegressor()
        # ETregr.fit(X_train2, y_train2)
        # predET = ETregr.predict(X_test2).round(0)
        # predETdf = pd.DataFrame(predET)
        # predETdf.columns = ["prednewhospi"]
        # featuresandtargetdf = X_test2.merge(y_test2, left_on = X_test2.index, right_on = y_test2.index)
        # featuresandtargetdf["prednewhospi"]=predETdf["prednewhospi"].round(0)
        # featuresandtargetdf.to_csv("/home/ludo915/code/covsco/predictions/fr/new_hospi_predictions.csv", index = False)
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

if __name__ == '__main__':

    TrainModel = maintrain(skipcrossval = "Y", skipimpute = "Y", skiptpot = None)
    TrainModel.initdata()
    TrainModel.CurrentBestModel()
    TrainModel.predict()