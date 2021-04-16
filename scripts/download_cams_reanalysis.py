import os
from datetime import date
import datetime as dt
import cdsapi
import yaml
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm
from os import listdir
from os.path import isfile, join
import urllib3

urllib3.disable_warnings()


work_dir = os.path.dirname(os.path.abspath(__file__))
save_to = os.path.join(work_dir, '../data/train/cams/fr/analysis')
if not os.path.exists(save_to):
    os.makedirs(save_to)

# get personal directory of cdsapi
try:
    with open('.cdsapirc_cams', 'r') as file:
        cams_api = file.readline().rstrip()
except FileNotFoundError:
    raise FileNotFoundError("""cdsapirc file cannot be found. Write the
        directory of your personal .cdsapirc file in a local file called
        `.cdsapirc_cams` and place it in the directory where this script lies.""")

# Download CAMS
# -----------------------------------------------------------------------------
print('Download data from CAMS ...', flush=True)

with open(cams_api, 'r') as f:
    credentials = yaml.safe_load(f)

mypath = "../data/train/cams/fr/analysis/"

def findlatestdateofcamsdata(mypath):
    dates = []
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    for filename in onlyfiles:
        dates.append(pd.to_datetime(filename[14:24]))
    
    if dates != []:
        return (dates, max(dates))
    else:
        return (dates, date.today() - pd.Timedelta("3 years"))

prevday = date.today() - pd.Timedelta("1 days")
startdate =findlatestdateofcamsdata(mypath)[1]
#startdate = pd.to_datetime("2020-01-01")
datesnotclean = pd.date_range(
    start=startdate,
    end= prevday
    ).strftime("%Y-%m-%d").tolist()
dates = []

for date in datesnotclean:
    if date not in findlatestdateofcamsdata(mypath)[0]:
        dates.append(date)

print(dates)
times 		= [dt.time(i).strftime('%H:00') for i in range(24)]

variables = [
    'carbon_monoxide',
    'nitrogen_dioxide',
    'ozone',
    'particulate_matter_2.5um',
    'particulate_matter_10um',
]

area = [51.75, -5.83, 41.67,11.03]


for date in tqdm(dates):
    file_name = 'cams-forecast-{:}.nc'.format(date)
    output_file = os.path.join(save_to,file_name)
    if not os.path.exists(output_file):
        c = cdsapi.Client(url=credentials['url'], key=credentials['key'])
        c.retrieve(
            'cams-europe-air-quality-forecasts',
            {
                'model': 'ensemble',
                'date': date,
                'format': 'netcdf',
                'variable': variables,
                'level': '0',
                'type': 'analysis',
                'time': times,
                'leadtime_hour': '0',
                'area'          : area
            },
            output_file
        )

        ds = xr.open_dataset(output_file)
        ds.close()
        ds= ds.mean('time').drop('level').squeeze()
        xdates = xr.DataArray(dates, dims=['time'], coords={'time': dates}, name='time')
        ds = ds.expand_dims(time=xdates).sel(time=date)
        ds.to_netcdf(output_file)

print('Download finished.', flush=True)
