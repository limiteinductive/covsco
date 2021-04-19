from hdx.hdx_configuration import Configuration
from hdx.data.dataset import Dataset
import pandas as pd
import numpy as np
import csv
from zipfile import ZipFile
import os
from utilities import download_url
import subprocess
import time

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
    "Bourgogne-Franche-Comt-":"Bourgogne-Franche-ComtÃ©",\
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
print(df2)

df2.to_csv("../data/train/all_data_merged/fr/Enriched_Covid_history_data.csv", index = False)
print('OK')


