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
def parse_dsm(coord):
    deg, min, sec, dir = re.split('[°\'"]', coord)
    dd = float(deg) + (float(min)/60) + (float(sec)/60/60)
    if (dir == 'W') | (dir == 'S'):
        dd *= -1
    return dd
# =============================================================================
# Data
# =============================================================================
print('Processing Population data ... ', flush=True, end='')
population  = pd.read_csv('../data/train/pop/fr/departements-francais.csv', sep=';')
population.columns = ['dep_num', 'name', 'region', 'capital', 'area', 'total', 'density']
#population['dep_num'] = population['dep_num'].replace({'2A':'201','2B':'202'}).astype(int)
population = population.sort_values('dep_num')
population = population[:-5]


dep_centre = pd.read_excel(
    '../data/train/pop/fr/Centre_departement.xlsx',
    engine='openpyxl', header=1, usecols=[0,1,2,3,4])
dep_centre.columns = ['dep_num','name','area', 'lon', 'lat']
dep_centre['dep_num'] = dep_centre['dep_num'].replace({'2A':'201','2B':'202'}).astype(int)
dep_centre = dep_centre.sort_values('dep_num')
dep_centre['lon'] = dep_centre['lon'].apply(lambda x: parse_dsm(x))
dep_centre['lat'] = dep_centre['lat'].apply(lambda x: parse_dsm(x))
dep_centre = dep_centre.merge(population, on=['dep_num'], how='outer')
dep_centre = dep_centre.drop(columns=['name_x', 'area_x', 'region'])
dep_centre.columns = ['dep_num','lon','lat','name','captial','area','total','density']

dep_centre.to_csv('../data/train/pop/fr/population_2020.csv', index=False)

population  = pd.read_csv('../data/train/pop/fr/population_2020.csv')

# Population Index
# Min-Max-normalized values of the log10 transformation
#population['idx'] = max_normalize(np.log10(population['total']))
population['idx'] = population['total']
population.reset_index(inplace = True, drop=True)
print('OK', flush=True)

print("\n")
# Covid #%%
# -----------------------------------------------------------------------------
print('Processing Covid data ... ', flush=True, end='')
filePath = '../data/train/covid/fr/'
fileName = 'Covid_data_history.csv'
os.system("""python download_covid_hist_data.py""")
covid = pd.read_csv(filePath + fileName, sep=',').dropna()
covid['date'] = pd.to_datetime(covid['date'])
# rename departments of la Corse to assure integer
#covid['numero'] = covid['numero'].replace({'2A':'201','2B':'202'}).astype(int)

# remove oversea departments
covid = covid[covid['numero']<203]

# take 1-week moving average and take today's values
# covid = covid.groupby('dep').rolling(window=7).mean()
# covid = covid.groupby(level=0).tail(1).reset_index(drop=True)

# add lon/lat + population index to covid dataframe
covid = covid.merge(population, how='inner', left_on='numero', right_on='dep_num')
print('OK', flush=True)
print("\n")
# CAMS #%%
# -----------------------------------------------------------------------------
def findmostancientdateofcamsdata(mypath):
    dates = []
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    for filename in onlyfiles:
        dates.append(pd.to_datetime(filename[14:24]))
    if dates != []:
        return min(dates)
    else:
        return "NaN"

os.system("""python download_cams_reanalysis.py""")
filePath = '../data/train/cams/fr/analysis/'
start_date, end_date = [pd.to_datetime(findmostancientdateofcamsdata(filePath)), date.today() -pd.Timedelta("1 Days")]
dates = pd.date_range(start_date, end_date, freq='D')[:-1]

print('Processing CAMS data ... ', flush=True, end='')

cams = xr.open_mfdataset(
    '../data/train/cams/fr/analysis/*.nc',
    combine='nested', concat_dim='time',
    parallel=True)
# n_time_steps = cams.coords['time'].size
# dates = dates[:-24]
#cams = cams.drop('level').squeeze()
cams = cams.assign_coords(time=dates)
cams = cams.assign_coords(longitude=(((cams['longitude'] + 180) % 360) - 180))
cams = cams.sel(longitude=slice(-10,10),latitude=slice(55,40))
cams = cams.sortby('longitude')

# CAMS is hourly ==> take daily means
cams = cams.resample({'time':'D'}).mean()
print(cams)
# there seems to be a pretty annoying issue with dask.array
# somehow I cannot manage to convert the dask.array to
# a standard xarray.DataArray; unfortunately, xarray.interp()
# seem not yet to work with dask.array; Therefore, as a workaround, I recreate
# a DataArray from scratch to assure that is a standard DataArray and no
# dask.array
# another minor issue here is that this workaround is only possible for each
# variable individually; really annoying....
pm25 = xr.DataArray(
    cams.pm2p5_conc.values,
    dims=['time','latitude','longitude'],
    coords = {
        'time':dates.to_period('d').unique(),
        'latitude':cams.coords['latitude'].values,
        'longitude':cams.coords['longitude'].values
    }
)
no2 = xr.DataArray(
    cams.no2_conc.values,
    dims=['time','latitude','longitude'],
    coords = {
        'time':dates.to_period('d').unique(),
        'latitude':cams.coords['latitude'].values,
        'longitude':cams.coords['longitude'].values
    }
)
co = xr.DataArray(
    cams.co_conc.values,
    dims=['time','latitude','longitude'],
    coords = {
        'time':dates.to_period('d').unique(),
        'latitude':cams.coords['latitude'].values,
        'longitude':cams.coords['longitude'].values
    }
)
o3 = xr.DataArray(
    cams.o3_conc.values,
    dims=['time','latitude','longitude'],
    coords = {
        'time':dates.to_period('d').unique(),
        'latitude':cams.coords['latitude'].values,
        'longitude':cams.coords['longitude'].values
    }
)
pm10 = xr.DataArray(
    cams.pm10_conc.values,
    dims=['time','latitude','longitude'],
    coords = {
        'time':dates.to_period('d').unique(),
        'latitude':cams.coords['latitude'].values,
        'longitude':cams.coords['longitude'].values
    }
)
# recreate Dataset (without dask)
cams = xr.Dataset({'pm25': pm25, 'no2': no2, 'o3': o3, 'co':co, 'pm10':pm10})
# interpolate CAMS data to lon/lat of departments
print("\n")
print("Interpolate CAMS data to lon/lat of departments ...")
lons = xr.DataArray(
    population['lon'],
    dims='dep_num',
    coords={'dep_num':population['dep_num']},
    name='lon')
lats = xr.DataArray(
    population['lat'],
    dims='dep_num',
    coords={'dep_num':population['dep_num']},
    name='lat')

cams = cams.interp(longitude=lons, latitude=lats)
cams = cams.to_dataframe().reset_index('dep_num')
cams.index = cams.index.to_timestamp()
cams = cams.reset_index()
cams.columns
covid = covid.rename(columns = {'date':'time'})
covid = covid.merge(cams, how='inner', on=['time','dep_num'])
covid.to_csv("../data/train/all_data_merged/fr/Enriched_Covid_history_data.csv", index = False)
print(covid)
print("\n")


print ("\n")
# Get Mobility indices historical data and merge it by time & region with the rest of the data
#  and export it to the Enriched_Covid_history_data.csv 
print("Processing Mobility indices data ...")
Configuration.create(hdx_site='prod',
                     user_agent='A_Quick_Example',
                     hdx_read_only=True)
dataset = Dataset.read_from_hdx('movement-range-maps')
resources = dataset.get_resources()
dic = resources[1]
url_mobility = dic['download_url']

file_mobility = "../data/train/mobility/fr/mvt_range.zip"
download_url(url_mobility, file_mobility)

with ZipFile(file_mobility, 'r',) as zipf:
    # printing all the contents of the zip file
    zipf.printdir()

    # extracting all the files
    print('Extracting mv_range file now...')
    mvt_range = zipf.namelist()[-1]
    zipf.extract(mvt_range,"../data/train/mobility/fr/")
    print('Done!')

os.chdir("../data/train/mobility/fr/")
os.system("""grep "FRA" """+ mvt_range + """ > mouvement-range-FRA.txt""")
os.system("""head -n 1 """+ mvt_range + """ > header.txt""")
os.system("""cat header.txt mouvement-range-FRA.txt > mouvement-range-FRA-final.csv""")
os.chdir("../../../../scripts")
df = pd.read_csv("../data/train/mobility/fr/mouvement-range-FRA-final.csv", sep = '\t')
print(df)
df["ds"]=pd.to_datetime(df["ds"], dayfirst = True)
df['polygon_name'] = df['polygon_name'].replace(
    {'Ile-de-France': 'Île-de-France',\
    '-le-de-France': 'Île-de-France',\
    "Auvergne-Rh-ne-Alpes":"Auvergne-Rhône-Alpes",\
    "Bourgogne-Franche-Comt-":"Bourgogne-Franche-Comté",\
    "Provence-Alpes-C-te d'Azur":"Provence-Alpes-Côte d'Azur"})

df2 = pd.read_csv('../data/train/all_data_merged/fr/Enriched_Covid_history_data.csv')
df2["time"]=pd.to_datetime(df2["time"])
df3 = pd.read_csv("../data/train/pop/fr/regions_departements.csv", sep = ";")

mdlist = []

df.reset_index(inplace=  True)
df2.reset_index(inplace = True)
df3.reset_index(inplace = True)
df.drop(columns = ["index"],inplace = True)
df2.drop(columns = ["index"],inplace = True)
df3.drop(columns = ["index"],inplace = True)



#df3['depnum'] = df3['depnum'].replace({'2A':'201','2B':'202'}).astype(int)
df2 = df2.merge(df3, how='inner', left_on = "numero", right_on = "depnum",suffixes=("","_y"))
#df2 = df2.merge(df, on = ["time, numero"])
df2 = df2.merge(df, how ="outer", left_on = ["Region","time"], right_on = ["polygon_name","ds"],suffixes=("","_y")).dropna()
print("TRUTH")
print(df["polygon_name"].unique().sort()==df3["Region"].unique().sort()==df2["Region"].unique().sort())
print(df["polygon_name"].unique())
print(df2["Region"].unique())
print(df2)

df2.to_csv("../data/train/all_data_merged/fr/Enriched_Covid_history_data.csv", index = False)
print('OK')

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

print("Computing the engineered features: 1M Trailing Maximal Pollution Concentrations (1M-TMPCs),1M Trailing Average Pollution Concentrations (1M-TAPCs), 7D Trailing Average Pollution Concentrations (7D-TAPCs) and the previous day's total hospitalizations...")

data = pd.read_csv('../data/train/all_data_merged/fr/Enriched_Covid_history_data.csv')

data["normpm25"]=max_normalize(data["pm25"])
data["normno2"]=max_normalize(data["no2"])
data["normo3"]=max_normalize(data["o3"])
data["normpm10"]=max_normalize(data["pm10"])
data["normco"]=max_normalize(data["co"])

data["time"]=pd.to_datetime(data["time"])

pm25tuple = (data['numero'], data['time'], data["pm25"], data["normpm25"] )
no2tuple = (data['numero'], data['time'], data["no2"], data["normno2"])
o3tuple = (data['numero'], data['time'], data["o3"], data["normo3"])
pm10tuple = (data['numero'], data['time'],data["pm10"], data["normpm10"])
cotuple = (data['numero'], data['time'], data["co"], data["normco"])
tothospituple = (data['numero'], data['time'], data["hospi"])

dicpm25 = {(i, j) : (k,l) for (i, j, k, l) in zip(*pm25tuple)}
dicno2 = {(i, j) : (k,l)  for (i, j, k, l) in zip(*no2tuple)}
dico3 = {(i, j) : (k,l)  for (i, j, k, l) in zip(*o3tuple)}
dicpm10 = {(i, j) : (k,l)  for (i, j, k, l) in zip(*pm10tuple)}
dicco = {(i, j) : (k,l) for (i, j, k, l) in zip(*cotuple)}
dictothospi = {(i, j) : k for (i, j, k) in zip(*tothospituple)}


referencedate = data["time"].min()
def compute_Engineered_Features(row):
    datalist = []
    datalist2 = []
    date = row['time'] - pd.Timedelta("30 days")
    date2 = row['time'] - pd.Timedelta("6 days")
    dateprevday = row['time'] - pd.Timedelta("1 days")

    dates = pd.date_range(start = date, periods=31).tolist()
    dates2 = pd.date_range(start = date2, periods=7).tolist()

    if (dateprevday < referencedate):
        prevdaytothospi = "NaN"
    else:
        prevdaytothospi = dictothospi[(row['numero'], dateprevday)] 

    for valuedate in dates:
        if(valuedate < referencedate):
            datalist.append((('NaN','Nan'),('NaN','Nan'),('NaN','Nan'),('NaN','Nan'),('NaN','Nan')))
        
        else:
            datalist.append((dicpm25[(row['numero'], pd.to_datetime(str(valuedate)))], \
                            dicno2[(row['numero'], pd.to_datetime(str(valuedate)))], \
                            dico3[(row['numero'], pd.to_datetime(str(valuedate)))], \
                            dicpm10[(row['numero'], pd.to_datetime(str(valuedate)))],\
                            dicco[(row['numero'], pd.to_datetime(str(valuedate)))]))

    if (dateprevday < referencedate):
        prevdaytothospi = "NaN"
    else:
        prevdaytothospi = dictothospi[(row['numero'], dateprevday)]

    for valuedate in dates2:
        if(valuedate < referencedate):
            datalist2.append((('NaN','Nan'),('NaN','Nan'),('NaN','Nan'),('NaN','Nan'),('NaN','Nan')))
        
        else:
            datalist2.append((dicpm25[(row['numero'], pd.to_datetime(str(valuedate)))], \
                            dicno2[(row['numero'], pd.to_datetime(str(valuedate)))], \
                            dico3[(row['numero'], pd.to_datetime(str(valuedate)))], \
                            dicpm10[(row['numero'], pd.to_datetime(str(valuedate)))],\
                            dicco[(row['numero'], pd.to_datetime(str(valuedate)))]))
    
    cleanedList = [((float(x),float(a)),(float(y),float(b)),(float(z),float(c)),(float(w),float(d)),(float(v),float(e))) \
        for ((x,a),(y,b),(z,c),(w,d),(v,e)) in datalist \
            if ((str(x),str(a)),(str(y),str(b)),(str(z),str(c)),(str(w),str(d)),(str(v),str(e))) \
                 != (('NaN','Nan'),('NaN','Nan'),('NaN','Nan'),('NaN','Nan'),('NaN','Nan'))]

    cleanedList2 = [((float(x),float(a)),(float(y),float(b)),(float(z),float(c)),(float(w),float(d)),(float(v),float(e))) \
        for ((x,a),(y,b),(z,c),(w,d), (v,e)) in datalist2 \
            if ((str(x),str(a)),(str(y),str(b)),(str(z),str(c)),(str(w),str(d)),(str(v),str(e))) \
                 != (('NaN','Nan'),('NaN','Nan'),('NaN','Nan'),('NaN','Nan'),('NaN','Nan'))]

    avg = [tuple(sum(j)/len(cleanedList) for j in zip(*i)) for i in zip(*cleanedList)]
    avg2 = [tuple(sum(j)/len(cleanedList2) for j in zip(*i)) for i in zip(*cleanedList2)]
    
    return (max(cleanedList,key=itemgetter(0))[0][0],\
            max(cleanedList,key=itemgetter(1))[1][0],\
            max(cleanedList,key=itemgetter(2))[2][0],\
            max(cleanedList,key=itemgetter(3))[3][0],\
            max(cleanedList,key=itemgetter(4))[4][0],\
            max(cleanedList,key=itemgetter(0))[0][1],\
            max(cleanedList,key=itemgetter(1))[1][1],\
            max(cleanedList,key=itemgetter(2))[2][1],\
            max(cleanedList,key=itemgetter(3))[3][1],\
            max(cleanedList,key=itemgetter(4))[4][1],\
            prevdaytothospi,\
            avg[0][0],\
            avg[1][0],\
            avg[2][0],\
            avg[3][0],\
            avg[4][0],\
            avg2[0][0],\
            avg2[1][0],\
            avg2[2][0],\
            avg2[3][0],\
            avg2[4][0],
            avg[0][1],\
            avg[1][1],\
            avg[2][1],\
            avg[3][1],\
            avg[4][1],\
            avg2[0][1],\
            avg2[1][1],\
            avg2[2][1],\
            avg2[3][1],\
            avg2[4][1])
            
@simple_time_tracker
def compute_Engineered_features_assign_to_df(data):
    data[['1MMaxpm25','1MMaxno2','1MMaxo3','1MMaxpm10','1MMaxco',\
            '1MMaxnormpm25','1MMaxnormno2','1MMaxnormo3','1MMaxnormpm10','1MMaxnormco', 
            'hospiprevday',
            'pm257davg','no27davg','o37davg', 'pm107davg','co7davg',\
            'pm251Mavg','no21Mavg','o31Mavg','pm101Mavg','co1Mavg',\
            "normpm257davg","normno27davg","normo37davg","normpm107davg","normco7davg",\
            "normpm251Mavg","normno21Mavg","normo31Mavg","normpm101Mavg","normco1Mavg"]] \
                = data.apply(compute_Engineered_Features, axis=1).apply(pd.Series)
    print("\n")
    return data

data =  compute_Engineered_features_assign_to_df(data)
print(data)

data.to_csv('../data/train/all_data_merged/fr/Enriched_Covid_history_data.csv', index = False)

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
