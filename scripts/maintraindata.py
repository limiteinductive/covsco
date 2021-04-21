#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =============================================================================
# Imports #%%
# =============================================================================
#part| #%%
import sys
import datetime
from datetime import datetime
import numpy as np
import pandas as pd
import xarray as xr
from datetime import timedelta
import re
from tqdm import tqdm
from operator import itemgetter
import itertools
from datetime import date
from os import listdir
from os.path import isfile, join
import os
from hdx.hdx_configuration import Configuration
from hdx.data.dataset import Dataset
import csv
from zipfile import ZipFile
from utilities import download_url
import subprocess
import time
from compute_engineered_features import Compute_Engineered_Features_for_df
from process_population_historical_data import process_population_hist
from download_covid_hist_data import download_covid_hist_data
from process_covid_historical_data import process_covid_historical_data
from download_cams_forecast import download_cams_forecast
from process_cams_forecast_data import process_cams_forecast_data
from process_mobility_historical_data import process_mobility_historical_data
from process_covid_positive_test_historical_data import process_covid_positive_test_historical_data
from process_hist_vaccination_data import process_hist_vaccination_data
from process_comorbidities import process_comorbidities_data
from process_smokers_data import process_smokers_data
from process_variants_hist_data import process_variants_hist_data

itertools.imap = lambda *args, **kwargs: list(map(*args, **kwargs))
# =============================================================================
# Functions #%%
# =============================================================================

def max_normalize(x):
    return (x - x.min()) / (x.max() - x.min())

# =============================================================================
# Merge data #%%
# =============================================================================

# Population
# -----------------------------------------------------------------------------

# =============================================================================
# Data
# =============================================================================
print('Processing Population data ... ', flush=True, end='')
GetPopulationData = process_population_hist()
GetPopulationData.get_data()
# Covid #%%
# -----------------------------------------------------------------------------
print('Processing Covid data ... ', flush=True, end='')
CovidHistData = download_covid_hist_data()
CovidHistData.GetData()
ProcessCovidHistoricalData = process_covid_historical_data()
ProcessCovidHistoricalData.process_covid_hist_data()
# CAMS #%%
# -----------------------------------------------------------------------------
print('Downloading Cams Forecast Data')
CamsHistForecasts = download_cams_forecast()
CamsHistForecasts.download()

print('Processing Cams Forecast Data')
ProcessCams = process_cams_forecast_data()
ProcessCams.process_cams()

# Get Mobility indices historical data and merge it by time & region with the rest of the data
#  and export it to the Enriched_Covid_history_data.csv 
print("Processing Mobility indices data ...")
ProcessMobility = process_mobility_historical_data()
ProcessMobility.process_mobility()

# Get Covid Positive Test historical data and merge it by time & departement with the rest of the data
#  and export it to the Enriched_Covid_history_data.csv 
print("Processing Covid Positive Tests (Previous day) ...")
ProcessCovidPositiveTests = process_covid_positive_test_historical_data()
ProcessCovidPositiveTests.process_covid_positive_test()

print("Processing Vaccination historical data ...")
ProcessVaccination = process_hist_vaccination_data()
ProcessVaccination.process_hist_vaccination()

print("Processing Comorbidities Data...")
ProcessComorbidities = process_comorbidities_data()
ProcessComorbidities.process_comorbidities()

print("Processing Smokers Data...")
ProcessSmokers = process_smokers_data()
ProcessSmokers.process_smokers()

print("Processing Variants data ...")
ProcessVariantsHistoricalData = process_variants_hist_data()
ProcessVariantsHistoricalData.process_variants()

print("Processing minority data...")

data = pd.read_csv('../data/train/minority/fr/minority.csv', sep=';')
data.rename(columns={
    'Corse du sud': 'Corse-du-Sud',
    'Haute Corse': 'Haute-Corse',
    "Côtes d'Armor": "Côtes-d'Armor"
},
            inplace=True)
df = pd.read_csv(
    '../data/train/all_data_merged/fr/Enriched_Covid_history_data.csv')
depts = pd.read_csv('../data/train/pop/fr/departements-francais.csv', sep=';')
depts_list = [element for element in depts['NOM']]
dic = {
    k: ('Unknown' if data[k][0] == 'nd' else
        float(data[k][0].replace("\u202f", '')) if k in depts_list else 'todo')
    for k in data.columns
}


def add_minority(row):
    return dic[row['nom']]


df['minority'] = df.apply(add_minority, axis=1)
print(df)
df.to_csv("../data/train/all_data_merged/fr/Enriched_Covid_history_data.csv",
          index=False)
print(df)

print("\n")
print("Reprocessing population data...")
data = pd.read_csv(
    '../data/train/all_data_merged/fr/Enriched_Covid_history_data.csv')
pop2020 = pd.read_excel('../data/train/pop/fr/popfr.xls', sheet_name='2020')
pop2021 = pd.read_excel('../data/train/pop/fr/popfr.xls', sheet_name='2021')

dic2020 = {k: v for k, v in zip(pop2020['depname'], pop2020['pop'])}
dic2021 = {k: v for k, v in zip(pop2021['depname'], pop2021['pop'])}


def rectify_pop(row):
    if row['time'][:4] == '2020':
        ans = dic2020[row['name']]
    elif row['time'][:4] == '2021':
        ans = dic2021[row['name']]
    return ans


data['total'] = data.apply(rectify_pop, axis=1)
data['idx'] = data.apply(rectify_pop, axis=1)
data.to_csv(
    '../data/train/all_data_merged/fr/Enriched_Covid_history_data.csv',
    index=False)
print(data)

print("Processing low_income data")
df = pd.read_csv('../data/train/low_income/fr/low_income.csv', sep=';')

df.rename(columns={
    'Corse du sud': 'Corse-du-Sud',
    'Haute Corse': 'Haute-Corse',
    "Côtes d'Armor": "Côtes-d'Armor"
},
          inplace=True)

data = pd.read_csv(
    '../data/train/all_data_merged/fr/Enriched_Covid_history_data.csv')
depts = pd.read_csv('../data/train/pop/fr/departements-francais.csv', sep=';')

depts_list = [element for element in depts['NOM']]

pauvrete_dic = {
    k: ('Unknown' if df[k][0] == 'nd' else float(df[k][0].replace(
        "\u202f", '.').replace(',', '.')) if k in depts_list else 'todo')
    for k in df.columns
}
rsa_dic = {
    k: ('Unknown' if df[k][1] == 'nd' else float(df[k][1].replace(
        "\u202f", '.').replace(',', '.')) if k in depts_list else 'todo')
    for k in df.columns
}
ouvriers_dic = {
    k: ('Unknown' if df[k][2] == 'nd' else float(df[k][2].replace(
        "\u202f", '.').replace(',', '.')) if k in depts_list else 'todo')
    for k in df.columns
}


def add_feature(row, feature):
    return feature[row['nom']]


data['pauvrete'] = data.apply(add_feature, axis=1, feature=pauvrete_dic)
data['rsa'] = data.apply(add_feature, axis=1, feature=rsa_dic)
data['ouvriers'] = data.apply(add_feature, axis=1, feature=ouvriers_dic)

print(data)
data.to_csv("../data/train/all_data_merged/fr/Enriched_Covid_history_data.csv",
          index=False)

#Computing the Engineered Features
Engineered_Features = Compute_Engineered_Features_for_df()
Engineered_Features.get_data()
Engineered_Features.max_normalize_data()
Engineered_Features.compute_dictionnaries()
Engineered_Features.compute_Engineered_features_assign_to_df()

print("Computing pm2.5 Pollutions levels")
df = pd.read_csv('../data/train/all_data_merged/fr/Enriched_Covid_history_data.csv')
df["time"]=pd.to_datetime(df["time"])
minpm25 = df["pm25"].min()
print(minpm25)
maxpm25 = df["pm25"].max()
print(maxpm25)

increment = (maxpm25 - minpm25)/4
pm25levelslist =[]
for i in range(5):
    pm25levelslist.append(minpm25 + i * increment)
print(pm25levelslist)

def pm25levels(row):
    pm25 = row['pm25']
    
    if (pm25 <= pm25levelslist[1]):
        level = 0
        levelstring = "Low"
    elif ((pm25levelslist[1] < pm25) & (pm25levelslist[2] >= pm25)):
        level = 1
        levelstring = "Medium"
    elif ((pm25levelslist[2] < pm25) & (pm25levelslist[3] >= pm25)):
        level = 2
        levelstring = "High"
    else:
        level = 3
        levelstring = "Very High"
          
    return (level, levelstring)


def pm25levels_to_df(data):
    data[["pm25level","pm25levelstring"]] \
                = data.apply(pm25levels, axis=1).apply(pd.Series)
    print("\n")
    return data

df =  pm25levels_to_df(df)
print(df)
print(df[(df['time']==df["time"].max()) & ((df["pm25levelstring"]=="High") | (df["pm25levelstring"]=="Very High"))][["nom","pm25levelstring"]])
df.to_csv("../data/train/all_data_merged/fr/Enriched_Covid_history_data.csv", index = False)
print('OK')
