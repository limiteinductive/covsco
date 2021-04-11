import pandas as pd
from tqdm import tqdm
import time

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

covpostesttuple = (df['dep'], df['jour'], df["P"], df["totalcovidcasescumulated"] )
diccovpostest = {(i, j) : (k,l) for (i, j, k, l) in zip(*covpostesttuple)}

referencedate = df["jour"].min()

def CovidPosTest(row):
    date = row['time']
    date2 = row['time'] - pd.Timedelta("1 days")
    if (date < referencedate):
        datatuple = ("NaN","NaN")
    else:
        datatuple = diccovpostest[(row["numero"],row["time"])]

    if (date2 < referencedate):
        prevdaycovidpostest = "NaN"

    else:   
        prevdaycovidpostest = diccovpostest[(row["numero"], date2)][0]

    return (datatuple[0], datatuple[1], prevdaycovidpostest)

@simple_time_tracker
def CovidPosTest_to_df(data):
    data[['CovidPosTest','TotalCovidCasesCumulated','covidpostestprevday']] \
                = data.apply(CovidPosTest, axis=1).apply(pd.Series)
    print("\n")
    return data

df2 =  CovidPosTest_to_df(df2)
print(df2)
df2.to_csv("../data/train/all_data_merged/fr/Enriched_Covid_history_data.csv", index = False)
print('OK')
