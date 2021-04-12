import os
import datetime as dt
import cdsapi
import yaml
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm
import urllib3
urllib3.disable_warnings()


work_dir = os.path.dirname(os.path.abspath(__file__))
save_to = os.path.join(work_dir, '../data/train/cams/analysis')
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

dates = pd.date_range(
    start="2018-04-12",
    end='2021-04-11'
    ).strftime("%Y-%m-%d").tolist()

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
