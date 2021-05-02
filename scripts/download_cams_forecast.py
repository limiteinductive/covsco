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

class download_cams_forecast:
    
    def __init__(self):
        self.work_dir = None
        self.save_to = None

    def download(self):
        self.work_dir = os.path.dirname(os.path.abspath(__file__))
        self.save_to = os.path.join(self.work_dir, '/home/ludo915/code/covsco/data/train/cams/fr/forecast')

        if not os.path.exists(self.save_to):
            os.makedirs(self.save_to)

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

        mypath = "/home/ludo915/code/covsco/data/train/cams/fr/forecast/"

        def findlatestdateofcamsdata(mypath):
            dates = []
            onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
            for filename in onlyfiles:
                dates.append(pd.to_datetime(filename[14:24]))
            
            if dates != []:
                return (dates, max(dates))
            else:
                return (dates, dt.date.today() - pd.Timedelta("3 Y"))

        prevday = dt.date.today() - pd.Timedelta("1 days")
        startdate =findlatestdateofcamsdata(mypath)[1]
        datesnotclean = pd.date_range(start=startdate,end= prevday).strftime("%Y-%m-%d").tolist()
        
        dates = []

        for date in datesnotclean:
            if date not in pd.to_datetime(findlatestdateofcamsdata(mypath)[0]):
                dates.append(date)

        print(dates)

        area = [51.75, -5.83, 41.67,11.03]

        for date in tqdm(dates):
            print(date)
            file_name = 'cams-forecast-{:}.nc'.format(date)
            output_file = os.path.join(self.save_to,file_name)
            if not os.path.exists(output_file):
                c = cdsapi.Client(url=credentials['url'], key=credentials['key'])
                c.retrieve(
                                'cams-europe-air-quality-forecasts',
                                {
                                    'variable': [
                                        'carbon_monoxide', 'nitrogen_dioxide', 'ozone',
                                        'particulate_matter_10um', 'particulate_matter_2.5um', 'sulphur_dioxide',
                                    ],
                                    'model': 'ensemble',
                                    'level': '0',
                                    'date': date,
                                    'type': 'forecast',
                                    'time': '00:00',
                                    'leadtime_hour': [
                                        '0', '24', '48',
                                        '72', '96'
                                    ],
                                    'area': area,
                                    'format': 'netcdf',
                                },
                                output_file)

        print('Download finished.', flush=True)

if __name__ == '__main__':
    CamsHistForecasts = download_cams_forecast()
    CamsHistForecasts.download()
