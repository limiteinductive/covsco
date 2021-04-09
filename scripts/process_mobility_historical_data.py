import pandas as pd

print("Processing Mobility indices data ...")
df = pd.read_csv("../data/train/mobility/fr/mouvement-range-FRA-final.csv", sep = ';')
df["ds"]=pd.to_datetime(df["ds"], dayfirst = True)
df = df[df["ds"]<=pd.to_datetime("31/03/2021",dayfirst=True)]
print(df)
df2 = pd.read_csv('../data/train/all_data_merged/fr/Enriched_Covid_history_data.csv')
df2["time"]=pd.to_datetime(df2["time"])
df3 = pd.read_csv("../data/train/pop/fr/regions_departements.csv", sep = ";")

mdlist = []

df.reset_index(inplace=  True)
df2.reset_index(inplace = True)
df3.reset_index(inplace = True)
df.drop(columns = ["index"],inplace = True)
df2.drop(columns = ["index"],inplace = True)
df3.drop(columns = ["index"],inplace = True)

print("TRUTH")
print(df["polygon_name"].unique().sort()==df3["Region"].unique().sort())

#df3['depnum'] = df3['depnum'].replace({'2A':'201','2B':'202'}).astype(int)
df2 = df2.merge(df3, how='inner', left_on = "numero", right_on = "depnum")
#df2 = df2.merge(df, on = ["time, numero"])
df2 = df2.merge(df, how ="outer", left_on = ["Region","time"], right_on = ["polygon_name","ds"]).dropna()

print(df2)

df2.to_csv("../data/train/all_data_merged/fr/Enriched_Covid_history_data.csv", index = False)
print('OK')



