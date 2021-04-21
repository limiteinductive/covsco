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

class process_mobility_historical_data:
    
    def __init__(self):
        self.file_mobility = None
        self.df = None
        self.df2 = None
        self.df3 = None

    def process_mobility(self):

        print("Processing Mobility indices data ...")
        Configuration.create(hdx_site='prod',
                            user_agent='A_Quick_Example',
                            hdx_read_only=True)
        dataset = Dataset.read_from_hdx('movement-range-maps')
        resources = dataset.get_resources()
        dic = resources[1]
        url_mobility = dic['download_url']

        self.file_mobility = "../data/train/mobility/fr/mvt_range.zip"
        download_url(url_mobility, self.file_mobility)

        with ZipFile(self.file_mobility, 'r',) as zipf:
            zipf.printdir()
            print('Extracting mv_range file now...')
            mvt_range = zipf.namelist()[-1]
            zipf.extract(mvt_range,"../data/train/mobility/fr/")
            print('Done!')

        os.chdir("../data/train/mobility/fr/")
        os.system("""grep "FRA" """+ mvt_range + """ > mouvement-range-FRA.txt""")
        os.system("""head -n 1 """+ mvt_range + """ > header.txt""")
        os.system("""cat header.txt mouvement-range-FRA.txt > mouvement-range-FRA-final.csv""")
        os.chdir("../../../../scripts")
        self.df = pd.read_csv("../data/train/mobility/fr/mouvement-range-FRA-final.csv", sep = '\t')
        print(self.df)
        self.df["ds"]=pd.to_datetime(self.df["ds"], dayfirst = True)
        self.df['polygon_name'] = self.df['polygon_name'].replace(
            {'Ile-de-France': 'Île-de-France',\
            '-le-de-France': 'Île-de-France',\
            "Auvergne-Rh-ne-Alpes":"Auvergne-Rhône-Alpes",\
            "Bourgogne-Franche-Comt-":"Bourgogne-Franche-Comté",\
            "Provence-Alpes-C-te d'Azur":"Provence-Alpes-Côte d'Azur"})

        self.df2 = pd.read_csv('../data/train/all_data_merged/fr/Enriched_Covid_history_data.csv')
        self.df2["date"]=pd.to_datetime(self.df2["date"])
        self.df3 = pd.read_csv("../data/train/pop/fr/regions_departements.csv", sep = ";")

        self.df.reset_index(inplace=  True)
        self.df2.reset_index(inplace = True)
        self.df3.reset_index(inplace = True)
        self.df.drop(columns = ["index"],inplace = True)
        self.df2.drop(columns = ["index"],inplace = True)
        self.df3.drop(columns = ["index"],inplace = True)

        self.df2 = self.df2.merge(self.df3, how='inner', left_on = "numero", right_on = "depnum",suffixes=("","_y"))
        self.df2 = self.df2.merge(self.df, how ="outer", left_on = ["Region","date"], right_on = ["polygon_name","ds"],suffixes=("","_y")).dropna()
        print(self.df2)
        self.df2.to_csv("../data/train/all_data_merged/fr/Enriched_Covid_history_data.csv", index = False)
        print('OK')

        return None

if __name__ == '__main__':

    ProcessMobility = process_mobility_historical_data()
    ProcessMobility.process_mobility()

