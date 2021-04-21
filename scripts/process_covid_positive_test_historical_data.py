import pandas as pd
from tqdm import tqdm
import time
from utilities import download_url
from datetime import datetime

# Get Covid Positive Test historical data and merge it by time & departement with the rest of the data
#  and export it to the Enriched_Covid_history_data.csv 

class process_covid_positive_test_historical_data:
   
    def __init__(self):

        self.url1 = None
        self.df = None
        self.df2 = None
        self.covpostesttuple = None
        self.diccovpostest = None 

    def CovidPosTest(self, row):
        date = row['date']
        date2 = row['date'] - pd.Timedelta("1 days")
        maxdate = self.df2["date"].max()
        referencedate = self.df["jour"].min()
        now = datetime.now()
        current_time_hour = int(now.strftime("%H"))
        week_days=["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        if(((date == maxdate)|
            (date == maxdate - pd.Timedelta("1 days"))| 
            (date == maxdate - pd.Timedelta("2 days"))|
            (date == maxdate - pd.Timedelta("3 days"))) & ((week_days[date.weekday()]=="Saturday") |
                                                        (week_days[date.weekday()]=="Sunday") |
                                                        ((week_days[date.weekday()]=="Monday") & current_time_hour < 12))):

            if week_days[date.weekday()]=="Saturday":
                date = row['date'] - pd.Timedelta("2 days")
                date2 = row['date'] - pd.Timedelta("3 days")
            elif (week_days[date.weekday()]=="Sunday"):
                date = row['date'] - pd.Timedelta("3 days")
                date2 = row['date'] - pd.Timedelta("4 days")
            elif ( (week_days[date.weekday()]=="Monday") & current_time_hour < 12):
                date = row['date'] - pd.Timedelta("4 days")
                date2 = row['date'] - pd.Timedelta("5 days")
            

        if (date < referencedate):
            datatuple = ("NaN","NaN")
        else:
            datatuple = self.diccovpostest[(row["numero"],date)]

        if (date2 < referencedate):
            prevdaycovidpostest = "NaN"
            prevdaytotalcovidcasescumulated ="Nan"

        else:   
            prevdaycovidpostest = self.diccovpostest[(row["numero"], date2)][0]
            prevdaytotalcovidcasescumulated = self.diccovpostest[(row["numero"], date2)][1]
        return (datatuple[0], datatuple[1], prevdaycovidpostest, prevdaytotalcovidcasescumulated)

    def CovidPosTest_to_df(self, data):
        data[['CovidPosTest','totalcovidcasescumulated','covidpostestprevday',"prevdaytotalcovidcasescumulated"]] \
                    = data.apply(self.CovidPosTest, axis=1).apply(pd.Series)
        print("\n")
        return data

    def process_covid_positive_test(self):

        print("Processing Covid Positive Tests (Previous day) ...")
        self.url1= "https://www.data.gouv.fr/en/datasets/r/406c6a23-e283-4300-9484-54e78c8ae675"
        download_url(self.url1, "../data/train/covidpostest/fr/covid_pos_test_hist_data.csv", chunk_size=128)
        self.df = pd.read_csv ("../data/train/covidpostest/fr/covid_pos_test_hist_data.csv", sep =";")
        self.df2 = pd.read_csv("../data/train/all_data_merged/fr/Enriched_Covid_history_data.csv", sep = ",") 
        self.df['dep'] = self.df['dep'].replace({'2A':'201','2B':'202'}).astype(int)
        self.df= self.df[self.df["dep"]<203]
        self.df["jour"]=pd.to_datetime(self.df["jour"], dayfirst = True)
        self.df2["date"]=pd.to_datetime(self.df2["date"])
        self.df = self.df.groupby(["dep","jour"]).sum().sort_values(["dep","jour"]).reset_index()
        dftotalcovidcasescumulated = self.df.groupby(['dep', 'jour']).sum().groupby(level=0).cumsum().sort_values(["dep","jour"]).reset_index()
        print(dftotalcovidcasescumulated)
        self.df = self.df[["dep","jour","P"]]
        self.df["totalcovidcasescumulated"]=dftotalcovidcasescumulated["P"]
        self.covpostesttuple = (self.df['dep'], self.df['jour'], self.df["P"], self.df["totalcovidcasescumulated"] )
        self.diccovpostest = {(i, j) : (k,l) for (i, j, k, l) in zip(*self.covpostesttuple)}
        self.df2 =  self.CovidPosTest_to_df(self.df2)
        print(self.df2)
        self.df2.to_csv("../data/train/all_data_merged/fr/Enriched_Covid_history_data.csv", index = False)
        print('OK')

if __name__ == '__main__':

    ProcessCovidPositiveTests = process_covid_positive_test_historical_data()
    ProcessCovidPositiveTests.process_covid_positive_test()