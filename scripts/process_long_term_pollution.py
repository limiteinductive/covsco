import xarray as xr
import pandas as pd
from datetime import date
from os import listdir
from os.path import isfile, join
from datetime import datetime

class process_long_term_pollution:
    
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

    def process_ltp(self):

        self.filePath = '/home/ludo915/code/covsco/data/train/longterm_pollution/fr/'

        #start_date, end_date = [pd.to_datetime(self.findmostancientdateofcamsdata(self.filePath)), self.findmostrecentdateofcamsdata(self.filePath)]
        dates = ['2001','2002','2003','2004','2005','2006','2007','2008','2009','2010','2011','2012','2013','2014','2015','2016','2017','2018']

        print('Processing long-term pollution data ... ', flush=True, end='')
    
        self.cams = xr.open_mfdataset(self.filePath + "*.nc", combine='nested', concat_dim='date', parallel=True)
        print(self.cams)
        # n_time_steps = cams.coords['time'].size
        # dates = dates[:-24]
        # self.cams = self.cams.drop('level').squeeze()
        # self.cams = self.cams.drop('time').squeeze()
        self.cams = self.cams.assign_coords(date=dates)
        #self.cams = self.cams.assign_coords(leadtime_hour=leadtime_hours)
        self.cams = self.cams.assign_coords(LON=(((self.cams['LON'] + 180) % 360) - 180))
        self.cams = self.cams.sel(LON=slice(-10,10),LAT=slice(55,40))
        self.cams = self.cams.sortby('LON')


   
        self.pm25 = xr.DataArray(
            self.cams.PM25.values,
            dims=['date','LAT','LON'],
            coords = {
                'date': self.cams.coords['date'].values,
                'LAT': self.cams.coords['LAT'].values,
                'LON': self.cams.coords['LON'].values
            }
        )

        self.cams = xr.Dataset({'pm25': self.pm25})
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

        print(self.cams)
        self.cams = self.cams.interp(LON=lons, LAT=lats)
        self.cams = self.cams.to_dataframe().reset_index()
        print(self.cams)
        # #self.cams.index = self.cams.index.to_timestamp()
        # print(self.cams.columns)
        # self.file_name = '/home/ludo915/code/covsco/data/train/covid/fr/Covid_data_history.csv'
        # self.covid = pd.read_csv(self.file_name, sep=',')
        # self.covid = self.covid.reset_index()
        # print(self.covid)
        # self.covid['date']=pd.to_datetime(self.covid['date'])
        # self.cams['date']=pd.to_datetime(self.cams['date'])
        # self.covid = self.covid.merge(self.cams, how= 'outer', on = ['date','numero'])
        # #self.covid.columns = self.covid.columns.droplevel()
        # self.covid.to_csv("/home/ludo915/code/covsco/data/train/all_data_merged/fr/Enriched_Covid_history_data.csv", index = False)
        # print(self.covid)
        # print(self.covid.columns)

        print("\n")

        return None

if __name__ == '__main__':

    ProcessLTP = process_long_term_pollution()
    ProcessLTP.process_ltp()