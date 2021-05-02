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
        referencedate = self.df["jour"].min()
        maxdatecovidpostest = self.df["jour"].max()

        if (date >= maxdatecovidpostest):
            date = maxdatecovidpostest   

        if (date2 >= maxdatecovidpostest):
            date2 = maxdatecovidpostest   
            
        if (date < referencedate):
            datatuple = ("NaN","NaN")
        else:
            datatuple = self.diccovpostest[(row["numero"],date)]

        if (date2 < referencedate):
            prevdaycovidpostest = "NaN"
            prevdaytotalcovidcasescumulated ="NaN"

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
        download_url(self.url1, "/home/ludo915/code/covsco/data/train/covidpostest/fr/covid_pos_test_hist_data.csv", chunk_size=128)
        self.df = pd.read_csv ("/home/ludo915/code/covsco/data/train/covidpostest/fr/covid_pos_test_hist_data.csv", sep =";")
        self.df2 = pd.read_csv("/home/ludo915/code/covsco/data/train/all_data_merged/fr/Enriched_Covid_history_data.csv", sep = ",") 
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
        self.df2.to_csv("/home/ludo915/code/covsco/data/train/all_data_merged/fr/Enriched_Covid_history_data.csv", index = False)
        print('OK')

if __name__ == '__main__':

    ProcessCovidPositiveTests = process_covid_positive_test_historical_data()
    ProcessCovidPositiveTests.process_covid_positive_test()