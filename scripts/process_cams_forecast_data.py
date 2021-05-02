import xarray as xr
import pandas as pd
from datetime import date
from os import listdir
from os.path import isfile, join
from datetime import datetime

class process_cams_forecast_data:
    
    def __init__(self):

        self.filePath = None
        self.file_name = None
        self.cams = None
        self.pm25 = None
        self.no2 = None
        self.o3 = None
        self.so2 = None
        self.population = None
        self.covid = None

    def findmostancientdateofcamsdata(self, mypath):
        dates = []
        onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
        for filename in onlyfiles:
            dates.append(pd.to_datetime(filename[14:24]))
        if dates != []:
            return min(dates)
        else:
            return "NaN"

    def findmostrecentdateofcamsdata(self, mypath):
        dates = []
        onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
        for filename in onlyfiles:
            dates.append(pd.to_datetime(filename[14:24]))
        if dates != []:
            return max(dates)
        else:
            return "NaN"

    def process_cams(self):

        self.filePath = '/home/ludo915/code/covsco/data/train/cams/fr/forecast/'

        start_date, end_date = [pd.to_datetime(self.findmostancientdateofcamsdata(self.filePath)), self.findmostrecentdateofcamsdata(self.filePath)]
        dates = pd.date_range(start_date, end_date, freq='D')
        leadtime_hours = ['0', '24', '48','72', '96']
        print('Processing CAMS data ... ', flush=True, end='')
    
        self.cams = xr.open_mfdataset(self.filePath + "*.nc", combine='nested', concat_dim='date', parallel=True)
        # n_time_steps = cams.coords['time'].size
        # dates = dates[:-24]
        self.cams = self.cams.drop('level').squeeze()
        self.cams = self.cams.drop('time').squeeze()
        self.cams = self.cams.assign_coords(date=dates)
        self.cams = self.cams.assign_coords(leadtime_hour=leadtime_hours)
        self.cams = self.cams.assign_coords(longitude=(((self.cams['longitude'] + 180) % 360) - 180))
        self.cams = self.cams.sel(longitude=slice(-10,10),latitude=slice(55,40))
        self.cams = self.cams.sortby('longitude')


   
        self.pm25 = xr.DataArray(
            self.cams.pm2p5_conc.values,
            dims=['date','leadtime_hour','latitude','longitude'],
            coords = {
                'date': self.cams.coords['date'].values,
                'leadtime_hour': self.cams.coords['leadtime_hour'].values,
                'latitude': self.cams.coords['latitude'].values,
                'longitude': self.cams.coords['longitude'].values
            }
        )
        self.no2 = xr.DataArray(
            self.cams.no2_conc.values,
            dims=['date','leadtime_hour','latitude','longitude'],
            coords = {
                'date': self.cams.coords['date'].values,
                'leadtime_hour': self.cams.coords['leadtime_hour'].values,
                'latitude': self.cams.coords['latitude'].values,
                'longitude': self.cams.coords['longitude'].values
            }
        )
        self.co = xr.DataArray(
            self.cams.co_conc.values,
            dims=['date','leadtime_hour','latitude','longitude'],
            coords = {
                'date': self.cams.coords['date'].values,
                'leadtime_hour': self.cams.coords['leadtime_hour'].values,
                'latitude': self.cams.coords['latitude'].values,
                'longitude': self.cams.coords['longitude'].values
            }
        )
        self.o3 = xr.DataArray(
            self.cams.o3_conc.values,
            dims=['date','leadtime_hour','latitude','longitude'],
            coords = {
                'date': self.cams.coords['date'].values,
                'leadtime_hour': self.cams.coords['leadtime_hour'].values,
                'latitude': self.cams.coords['latitude'].values,
                'longitude': self.cams.coords['longitude'].values
            }
        )
        self.pm10 = xr.DataArray(
            self.cams.pm10_conc.values,
            dims=['date','leadtime_hour','latitude','longitude'],
            coords = {
                'date': self.cams.coords['date'].values,
                'leadtime_hour': self.cams.coords['leadtime_hour'].values,
                'latitude': self.cams.coords['latitude'].values,
                'longitude': self.cams.coords['longitude'].values
            }
        )
        self.so2 = xr.DataArray(
            self.cams.so2_conc.values,
            dims=['date','leadtime_hour','latitude','longitude'],
            coords = {
                'date': self.cams.coords['date'].values,
                'leadtime_hour': self.cams.coords['leadtime_hour'].values,
                'latitude': self.cams.coords['latitude'].values,
                'longitude': self.cams.coords['longitude'].values
            }
        )
        self.cams = xr.Dataset({'pm25': self.pm25, 'no2': self.no2, 'o3': self.o3, 'co':self.co, 'pm10':self.pm10, 'so2':self.so2})
        # interpolate CAMS data to lon/lat of departments
        print("\n")
        print("Interpolate CAMS data to lon/lat of departments ...")
        self.population  = pd.read_csv('/home/ludo915/code/covsco/data/train/pop/fr/population_2020.csv')
        lons = xr.DataArray(
            self.population['lon'],
            dims='numero',
            coords={'numero':self.population['dep_num']},
            name='lon')
        lats = xr.DataArray(
            self.population['lat'],
            dims='numero',
            coords={'numero':self.population['dep_num']},
            name='lat')

        self.cams = self.cams.interp(longitude=lons, latitude=lats)
        self.cams = self.cams.to_dataframe().reset_index()
        print(self.cams)
        #self.cams.index = self.cams.index.to_timestamp()
        print(self.cams.columns)
        self.file_name = '/home/ludo915/code/covsco/data/train/covid/fr/Covid_data_history.csv'
        self.covid = pd.read_csv(self.file_name, sep=',')
        self.covid = self.covid.reset_index()
        print(self.covid)
        self.covid['date']=pd.to_datetime(self.covid['date'])
        self.cams['date']=pd.to_datetime(self.cams['date'])
        self.covid = self.covid.merge(self.cams, how= 'outer', on = ['date','numero'])
        #self.covid.columns = self.covid.columns.droplevel()
        self.covid.to_csv("/home/ludo915/code/covsco/data/train/all_data_merged/fr/Enriched_Covid_history_data.csv", index = False)
        print(self.covid)
        print(self.covid.columns)

        print("\n")

        return None

if __name__ == '__main__':

    ProcessCams = process_cams_forecast_data()
    ProcessCams.process_cams()