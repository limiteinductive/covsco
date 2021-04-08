import pandas as pd

df = pd.read_csv("../data/train/variants/fr/variants_hist_data.csv", sep=';')
df2 = pd.read_csv(
    "../data/train/all_data_merged/fr/Enriched_Covid_history_data.csv")
df2["time"] = pd.to_datetime(df2["time"])
df['dep'] = df['dep'].replace({'2A': '201', '2B': '202'}).astype(int)
df = df[df['dep'] < 203]

#df = df.groupby(["dep","semaine"], level = -1).sum()["dep","semaine","Nb_susp_501Y_V1","Nb_susp_501Y_V2_3"]
df = df.groupby(['dep', 'semaine'
                 ])[["dep", "semaine", "Nb_susp_501Y_V1", "Nb_susp_501Y_V2_3"
                     ]].sum().drop(columns=["dep"]).reset_index()


def to_datalist(row):
    date = pd.to_datetime(row["semaine"][0:10], yearfirst=True)
    return date


df['jour'] = df.apply(to_datalist, axis=1)
df.drop(columns='semaine', inplace=True)
df.rename(columns={'jour': 'semaine'}, inplace=True)


def create_possibilities(row):
    return pd.date_range(start=row['semaine'], periods=7).tolist()


df['7_days'] = df.apply(create_possibilities, axis=1)

referencedate = df['semaine'].min()


def check_variant(v_row, date):
    if date in v_row['7_days']:
        return v_row['Nb_susp_501Y_V1'], v_row['Nb_susp_501Y_V2_3']


def enriched_variant(row):
    date = row['time']
    depnum = row['numero']
    if date < referencedate:
        row['Nb_susp_501Y_V1'], row['Nb_susp_501Y_V2_3'] = 0, 0
    else:
        df_dep = df[df['dep'] == depnum]
        ans = []
        res = df_dep.apply(check_variant, date=date, axis=1)
        ans.append(next((el for el in res if el is not None),
                        None))  #get the first non null element of res
        row['Nb_susp_501Y_V1'], row['Nb_susp_501Y_V2_3'] = ans[0][0], ans[0][1]
    return None


df2.apply(enriched_variant, axis=1)

print(df2)
df2.to_csv("../data/train/all_data_merged/fr/Enriched_Covid_history_data.csv",
           index=False)
