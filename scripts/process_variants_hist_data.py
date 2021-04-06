import pandas as pd
from tqdm import tqdm

df = pd.read_csv("../data/train/variants/fr/variants_hist_data.csv", sep =';')
df2 = pd.read_csv("../data/train/all_data_merged/fr/Enriched_Covid_history_data.csv")
df2["time"] = pd.to_datetime(df2["time"])
df['dep'] = df['dep'].replace({'2A':'201','2B':'202'}).astype(int)
df = df[df['dep']<203]

#df = df.groupby(["dep","semaine"], level = -1).sum()["dep","semaine","Nb_susp_501Y_V1","Nb_susp_501Y_V2_3"]
df= df.groupby(['dep', 'semaine'])[["dep","semaine","Nb_susp_501Y_V1","Nb_susp_501Y_V2_3"]].sum().drop(columns = ["dep"]).reset_index()


print(df)
datalist = []
referencedate = pd.to_datetime("2021-02-12")

for i in tqdm(df.index):
    date = pd.to_datetime(df.loc[i,"semaine"][0:10], yearfirst = True)
    datalist.append((date, df.loc[i,"dep"], df.loc[i,"Nb_susp_501Y_V1"],df.loc[i,"Nb_susp_501Y_V2_3"]))

datalistdf = pd.DataFrame(datalist)
datalistdf.columns=["semaine","dep", "Nb_susp_501Y_V1","Nb_susp_501Y_V2_3"]
print(datalistdf)

datalist2 = []
for i in tqdm(df2.index):
    date = df2.loc[i,"time"]
    depnum = df2.loc[i,"numero"]
    datefound = False

    for j in datalistdf.index:
        counter = 0
        if datalistdf.loc[j,"dep"]==depnum: 
            date1 = datalistdf.loc[j,"semaine"]
            if (date < referencedate):
                datalist2.append((date,0,0))
                counter +=1
                datefound = True

            elif ((date >= referencedate) & (date1 <= date) & (date < (date1 + pd.Timedelta("7 days")))):
                V1 = datalistdf.loc[j,"Nb_susp_501Y_V1"]
                V2 = datalistdf.loc[j,"Nb_susp_501Y_V2_3"]
                datalist2.append((date,V1,V2))
                counter += 1
                datefound = True
            if (counter == 1):
                break
    if datefound == False:
        datalist.append((date, V1,V2))  
          

dfvar = pd.DataFrame(datalist2)
dfvar.columns=["date", "V1","V2"]
print(dfvar)


df2["Nb_susp_501Y_V1"]=dfvar["V1"]
df2["Nb_susp_501Y_V2_3"]=dfvar["V2"]

print(df2)
df2.to_csv("../data/train/all_data_merged/fr/Enriched_Covid_history_data.csv", index = False)