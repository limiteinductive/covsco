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
        self.pm25lttuple = None
        self.dicpm25lt = None
        self.ltpollutiondf = None

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
        print("Interpolate Long Term Pollution data to lon/lat of departments ...")
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
        self.compute_dictionnary()

        self.file_name = "/home/ludo915/code/covsco/data/train/all_data_merged/fr/Enriched_Covid_history_data.csv"
        self.ltpollutiondf = pd.read_csv("/home/ludo915/code/covsco/data/train/pop/fr/departements-francais.csv",sep =";")
        self.ltpollutiondf = self.ltpollutiondf[self.ltpollutiondf["numero"]<203]
        self.ltpollutiondf[["pm252001","pm252002","pm252003","pm252004","pm252005","pm252006","pm252007","pm252008",\
            "pm252009","pm252010","pm252011","pm252012","pm252013","pm252014","pm252015","pm252016","pm252017","pm252018"]]=\
                self.ltpollutiondf.apply(\
                self.compute_lt_pollution, axis = 1).apply(pd.Series)
        self.covid = pd.read_csv(self.file_name, sep=',')
        self.covid = self.covid.reset_index()
        print(self.covid)
        # self.covid['date']=pd.to_datetime(self.covid['date'])
        # self.cams['date']=pd.to_datetime(self.cams['date'])
        self.covid = self.covid.merge(self.ltpollutiondf, how= 'outer', on = ['numero'])
        # #self.covid.columns = self.covid.columns.droplevel()
        self.covid.to_csv("/home/ludo915/code/covsco/data/train/all_data_merged/fr/Enriched_Covid_history_data.csv", index = False)
        print(self.covid)
        # print(self.covid.columns)

        print("\n")

        return None
    
    def compute_dictionnary(self):
        self.pm25lttuple = (self.cams['numero'], self.cams["date"].astype(int), self.cams["pm25"])
        self.dicpm25lt = {(i, j) : k for (i, j, k) in zip(*self.pm25lttuple)}

        return None
    
    def compute_lt_pollution(self, row):
        depnum = row["numero"]
        data2001 = self.dicpm25lt[(depnum, 2001)]
        data2002 = self.dicpm25lt[(depnum, 2002)]
        data2003 = self.dicpm25lt[(depnum, 2003)]
        data2004 = self.dicpm25lt[(depnum, 2004)]
        data2005 = self.dicpm25lt[(depnum, 2005)]
        data2006 = self.dicpm25lt[(depnum, 2006)]
        data2007 = self.dicpm25lt[(depnum, 2007)]
        data2008 = self.dicpm25lt[(depnum, 2008)]
        data2009 = self.dicpm25lt[(depnum, 2009)]
        data2010 = self.dicpm25lt[(depnum, 2010)]
        data2011 = self.dicpm25lt[(depnum, 2011)]
        data2012 = self.dicpm25lt[(depnum, 2012)]
        data2013 = self.dicpm25lt[(depnum, 2013)]
        data2014 = self.dicpm25lt[(depnum, 2014)]
        data2015 = self.dicpm25lt[(depnum, 2015)]
        data2016 = self.dicpm25lt[(depnum, 2016)]
        data2017 = self.dicpm25lt[(depnum, 2017)]
        data2018 = self.dicpm25lt[(depnum, 2018)]

        return (
                data2001,\
                data2002,\
                data2003,\
                data2004,\
                data2005,\
                data2006,\
                data2007,\
                data2008,\
                data2009,\
                data2010,\
                data2011,\
                data2012,\
                data2013,\
                data2014,\
                data2015,\
                data2016,\
                data2017,\
                data2018\
                    )

if __name__ == '__main__':

    ProcessLTP = process_long_term_pollution()
    ProcessLTP.process_ltp()