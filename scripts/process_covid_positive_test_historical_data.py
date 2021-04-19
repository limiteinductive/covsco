import pandas as pd
from tqdm import tqdm
import time
from utilities import download_url
from datetime import datetime

def simple_time_tracker(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int(te - ts)
        else:
            print(method.__name__, round(te - ts, 2))
        return result
    return timed
# Get Covid Positive Test historical data and merge it by time & departement with the rest of the data
#  and export it to the Enriched_Covid_history_data.csv 
print("Processing Covid Positive Tests (Previous day) ...")

url1= "https://www.data.gouv.fr/en/datasets/r/406c6a23-e283-4300-9484-54e78c8ae675"
download_url(url1, "../data/train/covidpostest/fr/covid_pos_test_hist_data.csv", chunk_size=128)
df = pd.read_csv ("../data/train/covidpostest/fr/covid_pos_test_hist_data.csv", sep =";")
df2 = pd.read_csv("../data/train/all_data_merged/fr/Enriched_Covid_history_data.csv", sep = ",") 
df['dep'] = df['dep'].replace({'2A':'201','2B':'202'}).astype(int)
df= df[df["dep"]<203]
df["jour"]=pd.to_datetime(df["jour"], dayfirst = True)
df2["time"]=pd.to_datetime(df2["time"])
df = df.groupby(["dep","jour"]).sum().sort_values(["dep","jour"]).reset_index()
dftotalcovidcasescumulated = df.groupby(['dep', 'jour']).sum().groupby(level=0).cumsum().sort_values(["dep","jour"]).reset_index()
print(dftotalcovidcasescumulated)
df = df[["dep","jour","P"]]
df["totalcovidcasescumulated"]=dftotalcovidcasescumulated["P"]
df.to_csv("test.csv", sep =';')
week_days=["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
#week_num=datetime.date(2020,7,24).weekday()
#print(week_days[week_num])
covpostesttuple = (df['dep'], df['jour'], df["P"], df["totalcovidcasescumulated"] )
diccovpostest = {(i, j) : (k,l) for (i, j, k, l) in zip(*covpostesttuple)}
maxdate = df2["time"].max()
referencedate = df["jour"].min()
now = datetime.now()
current_time_hour = int(now.strftime("%H"))
def CovidPosTest(row):
    date = row['time']
    date2 = row['time'] - pd.Timedelta("1 days")
    if(((date == maxdate)|
        (date == maxdate - pd.Timedelta("1 days"))| 
        (date == maxdate - pd.Timedelta("2 days"))|
        (date == maxdate - pd.Timedelta("3 days"))) & ((week_days[date.weekday()]=="Saturday") |
                                                      (week_days[date.weekday()]=="Sunday") |
                                                      ((week_days[date.weekday()]=="Monday") & current_time_hour < 12))):

        if week_days[date.weekday()]=="Saturday":
            date = row['time'] - pd.Timedelta("2 days")
            date2 = row['time'] - pd.Timedelta("3 days")
        elif (week_days[date.weekday()]=="Sunday"):
            date = row['time'] - pd.Timedelta("3 days")
            date2 = row['time'] - pd.Timedelta("4 days")
        elif ( (week_days[date.weekday()]=="Monday") & current_time_hour < 12):
            date = row['time'] - pd.Timedelta("4 days")
            date2 = row['time'] - pd.Timedelta("5 days")
          

    if (date < referencedate):
        datatuple = ("NaN","NaN")
    else:
        datatuple = diccovpostest[(row["numero"],date)]

    if (date2 < referencedate):
        prevdaycovidpostest = "NaN"
        prevdaytotalcovidcasescumulated ="Nan"

    else:   
        prevdaycovidpostest = diccovpostest[(row["numero"], date2)][0]
        prevdaytotalcovidcasescumulated = diccovpostest[(row["numero"], date2)][1]
    return (datatuple[0], datatuple[1], prevdaycovidpostest, prevdaytotalcovidcasescumulated)

@simple_time_tracker
def CovidPosTest_to_df(data):
    data[['CovidPosTest','totalcovidcasescumulated','covidpostestprevday',"prevdaytotalcovidcasescumulated"]] \
                = data.apply(CovidPosTest, axis=1).apply(pd.Series)
    print("\n")
    return data

df2 =  CovidPosTest_to_df(df2)
print(df2)
df2.to_csv("../data/train/all_data_merged/fr/Enriched_Covid_history_data.csv", index = False)
print('OK')
