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

itertools.imap = lambda *args, **kwargs: list(map(*args, **kwargs))
# =============================================================================
# Functions #%%
# =============================================================================

def max_normalize(x):
    return (x - x.min()) / (x.max() - x.min())

# =============================================================================
# Merge data #%%
# =============================================================================

def simple_time_tracker(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int(te - ts)
        else:
            print(method.__name__, round(te - ts, 2))
        return result
    return timed

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
# Get Covid Positive Test historical data and merge it by time & departement with the rest of the data
#  and export it to the Enriched_Covid_history_data.csv 
print("Processing Covid Positive Tests (Previous day) ...")

url1= "https://www.data.gouv.fr/en/datasets/r/406c6a23-e283-4300-9484-54e78c8ae675"
download_url(url1, "../data/train/covidpostest/fr/covid_pos_test_hist_data.csv", chunk_size=128)
df = pd.read_csv ("../data/train/covidpostest/fr/covid_pos_test_hist_data.csv", sep =";")
df2 = pd.read_csv("../data/train/all_data_merged/fr/Enriched_Covid_history_data.csv", sep = ",") 
df['dep'] = df['dep'].replace({'2A':'201','2B':'202'}).astype(int)
df= df[df["dep"]<203]
df["jour"]=pd.to_datetime(df["jour"], dayfirst = True)
df2["time"]=pd.to_datetime(df2["time"])
df = df.groupby(["dep","jour"]).sum().sort_values(["dep","jour"]).reset_index()
dftotalcovidcasescumulated = df.groupby(['dep', 'jour']).sum().groupby(level=0).cumsum().sort_values(["dep","jour"]).reset_index()
print(dftotalcovidcasescumulated)
df = df[["dep","jour","P"]]
df["totalcovidcasescumulated"]=dftotalcovidcasescumulated["P"]
df.to_csv("test.csv", sep =';')
week_days=["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
#week_num=datetime.date(2020,7,24).weekday()
#print(week_days[week_num])
covpostesttuple = (df['dep'], df['jour'], df["P"], df["totalcovidcasescumulated"] )
diccovpostest = {(i, j) : (k,l) for (i, j, k, l) in zip(*covpostesttuple)}
maxdate = df2["time"].max()
referencedate = df["jour"].min()
now = datetime.now()
current_time_hour = int(now.strftime("%H"))
def CovidPosTest(row):
    date = row['time']
    date2 = row['time'] - pd.Timedelta("1 days")
    if(((date == maxdate)|
        (date == maxdate - pd.Timedelta("1 days"))| 
        (date == maxdate - pd.Timedelta("2 days"))|
        (date == maxdate - pd.Timedelta("3 days"))) & ((week_days[date.weekday()]=="Saturday") |
                                                      (week_days[date.weekday()]=="Sunday") |
                                                      ((week_days[date.weekday()]=="Monday") & current_time_hour < 12))):

        if week_days[date.weekday()]=="Saturday":
            date = row['time'] - pd.Timedelta("2 days")
            date2 = row['time'] - pd.Timedelta("3 days")
        elif (week_days[date.weekday()]=="Sunday"):
            date = row['time'] - pd.Timedelta("3 days")
            date2 = row['time'] - pd.Timedelta("4 days")
        elif ( (week_days[date.weekday()]=="Monday") & current_time_hour < 12):
            date = row['time'] - pd.Timedelta("4 days")
            date2 = row['time'] - pd.Timedelta("5 days")
          

    if (date < referencedate):
        datatuple = ("NaN","NaN")
    else:
        datatuple = diccovpostest[(row["numero"],date)]

    if (date2 < referencedate):
        prevdaycovidpostest = "NaN"
        prevdaytotalcovidcasescumulated ="Nan"

    else:   
        prevdaycovidpostest = diccovpostest[(row["numero"], date2)][0]
        prevdaytotalcovidcasescumulated = diccovpostest[(row["numero"], date2)][1]
    return (datatuple[0], datatuple[1], prevdaycovidpostest, prevdaytotalcovidcasescumulated)

@simple_time_tracker
def CovidPosTest_to_df(data):
    data[['CovidPosTest','totalcovidcasescumulated','covidpostestprevday',"prevdaytotalcovidcasescumulated"]] \
                = data.apply(CovidPosTest, axis=1).apply(pd.Series)
    print("\n")
    return data

df2 =  CovidPosTest_to_df(df2)
print(df2)
df2.to_csv("../data/train/all_data_merged/fr/Enriched_Covid_history_data.csv", index = False)
print('OK')




print("\n")

print("Processing Vaccination historical data ...")
url2 = "https://www.data.gouv.fr/es/datasets/r/59aeab47-c364-462c-9087-ce233b6acbbc"

download_url(url2, "../data/train/vaccination/fr/vaccination_hist_data.csv", chunk_size=128)

df = pd.read_csv("../data/train/vaccination/fr/vaccination_hist_data.csv")
print(df.columns)
df['departement'] = df['departement'].replace({
    '2A': '201',
    '2B': '202'
}).astype(int)
df = df[df['departement'] < 203]
df["date_debut_semaine"] = pd.to_datetime(df["date_debut_semaine"],
                                          dayfirst=True)

df2 = pd.read_csv(
    "../data/train/all_data_merged/fr/Enriched_Covid_history_data.csv",
    sep=",")
df2['vac1nb']=0
df2['vac2nb']=0
df2["time"] = pd.to_datetime(df2["time"])

dfvac1 = df[df["rang_vaccinal"] == 1].reset_index()
dfvac2 = df[df["rang_vaccinal"] == 2].reset_index()

referencedate1 = dfvac1['date_debut_semaine'].min()
referencedate2 = dfvac2['date_debut_semaine'].min()

cum1 = dfvac1.groupby(['departement', 'date_debut_semaine']).sum().groupby(
    level=0).cumsum().sort_values("date_debut_semaine").reset_index().drop(
        columns="index")
cum2 = dfvac2.groupby(['departement', 'date_debut_semaine']).sum().groupby(
    level=0).cumsum().sort_values("date_debut_semaine").reset_index().drop(
        columns="index")


def create_week(row):
    return pd.date_range(start=row['date_debut_semaine'], periods=7).tolist()


cum1['7_days'] = cum1.apply(create_week, axis=1)
cum2['7_days'] = cum2.apply(create_week, axis=1)


def check_vaccin(v_row, date):
    if date in v_row['7_days']:
        return v_row['nb']


def enriched_vaccin(row):
    date = row['time']
    depnum = row['numero']
    if date < referencedate1:
        (first1, first2) = (0, 0)
    else:
        cum1_dep = cum1[cum1['departement'] == depnum]
        res1 = cum1_dep.apply(check_vaccin, date=date, axis=1)
        first1 = [el for el in res1
                  if el == el][0]  #get the first non null element of res

        cum2_dep = cum2[cum2['departement'] == depnum]
        res2 = cum2_dep.apply(check_vaccin, date=date, axis=1)
        first2 = next((el for el in res2 if el == el),
                      None)  #get the first non null element of res
        if first2 is None:
            first2 = 0

    return ((first1, first2))


df2[['vac1nb','vac2nb' ]] = df2.apply(enriched_vaccin, axis=1).apply(pd.Series)
print(df2)
df2.to_csv("../data/train/all_data_merged/fr/Enriched_Covid_history_data.csv",
           index=False)

print("Processing Comorbidities Data...")
df = pd.read_excel("../data/train/comorbidities/fr/2019_ALD-prevalentes-par-departement_serie-annuelle.xls", skiprows = 1)
df['Code département']=df['Code département'].astype(int)
df= df[df['Code département']<203]
df = df[["Code département", 'Insuffisance respiratoire chronique grave (ALD14)','Insuffisance cardiaque grave, troubles du rythme graves, cardiopathies valvulaires graves, cardiopathies congénitales graves (ALD5)']]
print(df)
print(df.columns)
df2 = pd.read_csv("../data/train/all_data_merged/fr/Enriched_Covid_history_data.csv")
df2 = df2.merge(df, how ="inner", left_on = "numero", right_on = "Code département")
print(df2)
df2.to_csv("../data/train/all_data_merged/fr/Enriched_Covid_history_data.csv", index = False)

print("Processing Smokers Data...")
df = pd.read_csv("../data/train/smoker/fr/smoker_regions_departements.csv", sep =';')
df2 = pd.read_csv("../data/train/all_data_merged/fr/Enriched_Covid_history_data.csv")
print(df)
df["depnum"]=df["depnum"].astype(int)
df["Smokers"]=df["Smokers"].astype(float)
df2 = df2.merge(df, how = "inner", left_on = "numero", right_on = "depnum")
df2.to_csv("../data/train/all_data_merged/fr/Enriched_Covid_history_data.csv", index = False)
print(df2)
print("\n")

print("Processing Variants data ...")

url3 ="https://www.data.gouv.fr/fr/datasets/r/16f4fd03-797f-4616-bca9-78ff212d06e8"
download_url(url3, "../data/train/variants/fr/variants_hist_data.csv", chunk_size=128)

df = pd.read_csv("../data/train/variants/fr/variants_hist_data.csv", sep=';')
df2 = pd.read_csv("../data/train/all_data_merged/fr/Enriched_Covid_history_data.csv")

df2["time"] = pd.to_datetime(df2["time"])
df['dep'] = df['dep'].replace({'2A': '201', '2B': '202'}).astype(int)
df = df[df['dep'] < 203]
df = df.groupby(['dep', 'semaine'
                 ])[["dep", "semaine", "Nb_susp_501Y_V1", "Nb_susp_501Y_V2_3"
                     ]].sum().drop(columns=["dep"]).reset_index()


def to_datalist(row):
    date = pd.to_datetime(row["semaine"][11:21], yearfirst=True)
    return date


df['jour'] = df.apply(to_datalist, axis=1)
df.drop(columns='semaine', inplace=True)
# df.rename(columns={'jour': 'semaine'}, inplace=True)


# def create_possibilities(row):
#     return pd.date_range(start=row['semaine'], periods=7).tolist()


# df['7_days'] = df.apply(create_possibilities, axis=1)

referencedate = df['jour'].min()

variantstuple = (df['dep'], df['jour'], df["Nb_susp_501Y_V1"], df["Nb_susp_501Y_V2_3"] )
dicvariant = {(i, j) : (k,l) for (i, j, k, l) in zip(*variantstuple)}

def enriched_variant(row):
    date = row['time']
    depnum = row['numero']
    week_days=["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    maxdate = df2["time"].max()
    now = datetime.now()
    current_time_hour = int(now.strftime("%H"))
    if(((date == maxdate)|
        (date == maxdate - pd.Timedelta("1 days"))| 
        (date == maxdate - pd.Timedelta("2 days"))|
        (date == maxdate - pd.Timedelta("3 days"))) & ((week_days[date.weekday()]=="Saturday") |
                                                      (week_days[date.weekday()]=="Sunday") |
                                                      ((week_days[date.weekday()]=="Monday") & current_time_hour < 12))):

        if week_days[date.weekday()]=="Saturday":
            date = row['time'] - pd.Timedelta("2 days")
        elif (week_days[date.weekday()]=="Sunday"):
            date = row['time'] - pd.Timedelta("3 days")
        elif ( (week_days[date.weekday()]=="Monday") & current_time_hour < 12):
            date = row['time'] - pd.Timedelta("4 days")
    
    if date < referencedate:
        return (0, 0)
    else:
        return (dicvariant[(depnum,date)])


df2[['Nb_susp_501Y_V1','Nb_susp_501Y_V2_3']] = df2.apply(enriched_variant, axis=1).apply(pd.Series)
df2.sort_values(by = ["numero","time"], inplace = True)
print(df2)
df2.to_csv("../data/train/all_data_merged/fr/Enriched_Covid_history_data.csv",\
           index=False)
print("\n")

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

@simple_time_tracker
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
