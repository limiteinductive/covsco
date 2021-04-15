from hdx.hdx_configuration import Configuration
from hdx.data.dataset import Dataset
import pandas as pd
import numpy as np
import csv
from zipfile import ZipFile
import os

from scripts.utilities import download_url

Configuration.create(hdx_site='prod',
                     user_agent='A_Quick_Example',
                     hdx_read_only=True)
dataset = Dataset.read_from_hdx('movement-range-maps')
resources = dataset.get_resources()
dic = resources[1]
url_mobility = dic['download_url']

file_mobility = "mvt_range.zip"
download_url(url_mobility, file_mobility)

with ZipFile(file_mobility, 'r') as zip:
    # printing all the contents of the zip file
    zip.printdir()

    # extracting all the files
    print('Extracting mv_range file now...')
    mvt_range = zip.namelist()[-1]
    zip.extract(mvt_range)
    print('Done!')

with open(mvt_range) as f:
    reader = csv.reader(f, delimiter="\t")
    d = list(reader)

data_mob = pd.DataFrame(d[1:], columns=d[0])
data_mob = data_mob[data_mob['country'] == 'FRA']

data_mob = data_mob.iloc[[-1]]  #get latest data, last line

#to do : add row to a df with all live_data
