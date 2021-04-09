import pandas as pd
from tqdm import tqdm
# Get Covid Positive Test historical data and merge it by time & departement with the rest of the data
#  and export it to the Enriched_Covid_history_data.csv 
print("Processing Covid Positive Tests (Previous day) ...")
df = pd.read_csv ("../data/train/covidpostest/fr/covid_pos_test_hist_data.csv", sep =";")
df2 = pd.read_csv("../data/train/all_data_merged/fr/Enriched_Covid_history_data.csv", sep = ",") 
df['dep'] = df['dep'].replace({'2A':'201','2B':'202'}).astype(int)
df["jour"]=pd.to_datetime(df["jour"], dayfirst = True)
df2["time"]=pd.to_datetime(df2["time"])
df = df.groupby(["dep","jour"]).sum().reset_index()
print(df)
df = df[["dep","jour","P"]]
covidpostestdayminus1list = []
for i in tqdm(df.index):
    date0 = df.loc[i,"jour"]
    depnum = df.loc[i,"dep"]
    date1 = date0 -pd.Timedelta("1 days")
    dayminus1covidpostest  = df[(df["jour"]== date1) & (df["dep"]==depnum)].reset_index()["P"]
    if list(dayminus1covidpostest)==[]: 
        covidpostestdayminus1list.append("NaN") 
    else:
        covidpostestdayminus1list.append(list(dayminus1covidpostest)[0])

covidposttest = pd.DataFrame(covidpostestdayminus1list)
covidposttest.columns =["covidpostestprevday"]

df["covidpostestprevday"]=covidposttest["covidpostestprevday"]
df2 = df2.merge(df, left_on = ["time","numero"], right_on = ["jour","dep"])
df2.to_csv("../data/train/all_data_merged/fr/Enriched_Covid_history_data.csv", index = False)
print(df2)
print('OK')
