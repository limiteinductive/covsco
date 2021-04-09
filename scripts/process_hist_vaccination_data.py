import pandas as pd

print("Processing Vaccination historical data ...")
df = pd.read_csv("../data/train/vaccination/fr/vaccination_hist_data.csv",
                 sep=";")
df['departement'] = df['departement'].replace({
    '2A': '201',
    '2B': '202'
}).astype(int)
df = df[df['departement'] < 203]
df["date_debut_semaine"] = pd.to_datetime(df["date_debut_semaine"],
                                          dayfirst=True)

df2 = pd.read_csv(
    "../data/train/all_data_merged/fr/Enriched_Covid_history_data.csv",
    sep=",")
df2['vac1nb']=0
df2['vac2nb']=0
df2["time"] = pd.to_datetime(df2["time"])

dfvac1 = df[df["rang_vaccinal"] == 1].reset_index()
dfvac2 = df[df["rang_vaccinal"] == 2].reset_index()

referencedate1 = dfvac1['date_debut_semaine'].min()
referencedate2 = dfvac2['date_debut_semaine'].min()

cum1 = dfvac1.groupby(['departement', 'date_debut_semaine']).sum().groupby(
    level=0).cumsum().sort_values("date_debut_semaine").reset_index().drop(
        columns="index")
cum2 = dfvac2.groupby(['departement', 'date_debut_semaine']).sum().groupby(
    level=0).cumsum().sort_values("date_debut_semaine").reset_index().drop(
        columns="index")


def create_week(row):
    return pd.date_range(start=row['date_debut_semaine'], periods=7).tolist()


cum1['7_days'] = cum1.apply(create_week, axis=1)
cum2['7_days'] = cum2.apply(create_week, axis=1)


def check_vaccin(v_row, date):
    if date in v_row['7_days']:
        return v_row['nb']


def enriched_vaccin(row):
    date = row['time']
    depnum = row['numero']
    if date < referencedate1:
        (first1, first2) = (0, 0)
    else:
        cum1_dep = cum1[cum1['departement'] == depnum]
        res1 = cum1_dep.apply(check_vaccin, date=date, axis=1)
        first1 = [el for el in res1
                  if el == el][0]  #get the first non null element of res

        cum2_dep = cum2[cum2['departement'] == depnum]
        res2 = cum2_dep.apply(check_vaccin, date=date, axis=1)
        first2 = next((el for el in res2 if el == el),
                      None)  #get the first non null element of res
        if first2 is None:
            first2 = 0

    return ((first1, first2))


df2[['vac1nb','vac2nb' ]] = df2.apply(enriched_vaccin, axis=1).apply(pd.Series)
print(df2)
df2.to_csv("../data/train/all_data_merged/fr/Enriched_Covid_history_data.csv",
           index=False)
