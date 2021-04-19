### IMPORTS

from hdx.hdx_configuration import Configuration
from hdx.data.dataset import Dataset
import requests
import pandas as pd
import numpy as np
import csv
from zipfile import ZipFile
import os

# ----------------

### INITIATE PREDICTION FILE
depts = pd.read_csv('../data/train/pop/fr/departements-francais.csv', sep=';')
depts.columns = [
    'dep_num', 'name', 'region', 'capital', 'area', 'total', 'density'
]
depts = depts.sort_values('dep_num')
depts = depts[:-5]
depts['region'] = depts['region'].replace({'Ile-de-France': 'Île-de-France'})

depts.to_csv('../data/predictiondata.csv')
