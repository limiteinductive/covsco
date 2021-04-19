import pandas as pd
from utilities import download_url
from datetime import datetime

print("Processing Variants data ...")

url3 ="https://www.data.gouv.fr/fr/datasets/r/16f4fd03-797f-4616-bca9-78ff212d06e8"
download_url(url3, "../data/train/variants/fr/variants_hist_data.csv", chunk_size=128)

df = pd.read_csv("../data/train/variants/fr/variants_hist_data.csv", sep=';')
df2 = pd.read_csv("../data/train/all_data_merged/fr/Enriched_Covid_history_data.csv")

df2["time"] = pd.to_datetime(df2["time"])
df['dep'] = df['dep'].replace({'2A': '201', '2B': '202'}).astype(int)
df = df[df['dep'] < 203]
df = df.groupby(['dep', 'semaine'
                 ])[["dep", "semaine", "Nb_susp_501Y_V1", "Nb_susp_501Y_V2_3"
                     ]].sum().drop(columns=["dep"]).reset_index()


def to_datalist(row):
    date = pd.to_datetime(row["semaine"][11:21], yearfirst=True)
    return date


df['jour'] = df.apply(to_datalist, axis=1)
df.drop(columns='semaine', inplace=True)
# df.rename(columns={'jour': 'semaine'}, inplace=True)


# def create_possibilities(row):
#     return pd.date_range(start=row['semaine'], periods=7).tolist()


# df['7_days'] = df.apply(create_possibilities, axis=1)

referencedate = df['jour'].min()

variantstuple = (df['dep'], df['jour'], df["Nb_susp_501Y_V1"], df["Nb_susp_501Y_V2_3"] )
dicvariant = {(i, j) : (k,l) for (i, j, k, l) in zip(*variantstuple)}

def enriched_variant(row):
    date = row['time']
    depnum = row['numero']
    week_days=["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    maxdate = df2["time"].max()
    now = datetime.now()
    current_time_hour = int(now.strftime("%H"))
    if(((date == maxdate)|
        (date == maxdate - pd.Timedelta("1 days"))| 
        (date == maxdate - pd.Timedelta("2 days"))|
        (date == maxdate - pd.Timedelta("3 days"))) & ((week_days[date.weekday()]=="Saturday") |
                                                      (week_days[date.weekday()]=="Sunday") |
                                                      ((week_days[date.weekday()]=="Monday") & current_time_hour < 12))):

        if week_days[date.weekday()]=="Saturday":
            date = row['time'] - pd.Timedelta("2 days")
        elif (week_days[date.weekday()]=="Sunday"):
            date = row['time'] - pd.Timedelta("3 days")
        elif ( (week_days[date.weekday()]=="Monday") & current_time_hour < 12):
            date = row['time'] - pd.Timedelta("4 days")
    
    if date < referencedate:
        return (0, 0)
    else:
        return (dicvariant[(depnum,date)])


df2[['Nb_susp_501Y_V1','Nb_susp_501Y_V2_3']] = df2.apply(enriched_variant, axis=1).apply(pd.Series)
df2.sort_values(by = ["numero","time"], inplace = True)
print(df2)
df2.to_csv("../data/train/all_data_merged/fr/Enriched_Covid_history_data.csv",\
           index=False)