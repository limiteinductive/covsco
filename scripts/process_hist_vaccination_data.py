import pandas as pd
import datetime as dt

print("Vaccinnation data")
df = pd.read_csv ("../data/train/vaccination/fr/vaccination_hist_data.csv", sep =";")
df['departement'] = df['departement'].replace({'2A':'201','2B':'202'}).astype(int)
df = df[df['numero']<203]
df["date_debut_semaine"]=pd.to_datetime(df["date_debut_semaine"]) 
df2 = pd.read_csv("../data/train/all_data_merged/fr/Enriched_Covid_history_data.csv", sep = ",")
print(df)
print(df2)
dfvac1 = df[df["rang_vaccinal"]==1].reset_index()
dfvac2 = df[df["rang_vaccinal"]==2].reset_index()
print(dfvac1)
print(dfvac2)
dfvac1list = []
dfvac2list = []
referencedate = "2021-01-18"
referencedate=pd.to_datetime(referencedate)
referencedate2 = "2021-01-25"
referencedate2=pd.to_datetime(referencedate2)


df2["time"]=pd.to_datetime(df2["time"])
#df['departement'] = df['departement'].replace({'2A':'201','2B':'202'}).astype(int)


cumvac1list = []
cumvac2list = []
cum1 = dfvac1.groupby(['departement', 'date_debut_semaine']).sum().groupby(level=0).cumsum().sort_values("date_debut_semaine").reset_index().drop(columns = "index")
cum2 = dfvac2.groupby(['departement', 'date_debut_semaine']).sum().groupby(level=0).cumsum().sort_values("date_debut_semaine").reset_index().drop(columns = "index")

print(cum1)
print(cum2)

for i in df2.index:
    date = df2.loc[i,"time"]
    depnum = df2.loc[i,"numero"]
    datefound = False
    datefound2 = False
    for j in cum1.index:
        counter = 0
        if cum1.loc[j,"departement"]==depnum: 
            date1 = cum1.loc[j,"date_debut_semaine"]
            if (date < referencedate):
                dfvac1list.append((date,0))
                counter +=1
                datefound = True

            elif ((date >= referencedate) & (date1 <= date) & (date < (date1 + pd.Timedelta("7 days")))):
                cumvac1 = cum1.loc[j,"nb"]
                dfvac1list.append((date,cumvac1))
                counter += 1
                datefound = True
            if (counter == 1):
                break
    if datefound == False:
        dfvac2list.append((date, cumvac1))

    for k in cum2.index:
        counter = 0
        if cum2.loc[k,"departement"]==depnum: 
            date2 = cum2.loc[k,"date_debut_semaine"]
            if (date < referencedate2):
                dfvac2list.append((date,0))
                counter +=1
                datefound2 = True
            elif ((date >= referencedate2) & (date2 <= date) & (date < (date2 + pd.Timedelta("7 days")))):
                cumvac2 = cum2.loc[k,"nb"]
                dfvac2list.append((date, cumvac2))
                counter +=1
                datefound2 = True
            if (counter == 1):
                break
    if datefound2 == False:
        dfvac2list.append((date, cumvac2))
                
            
           

dfvac1 = pd.DataFrame(dfvac1list)
dfvac1.columns=["date", "vac1nb"]
print(dfvac1)
dfvac2 = pd.DataFrame(dfvac2list)
dfvac2.columns=["date", "vac2nb"]
print(dfvac2)

df2["vac1nb"]=dfvac1["vac1nb"]
df2["vac2nb"]=dfvac2["vac2nb"]
df2.dropna(inplace = True)
#df2.to_csv("../data/train/all_data_merged/fr/Enriched_Covid_history_data.csv", index = False)
print(df2)





