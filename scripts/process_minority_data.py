import pandas as pd

data = pd.read_csv('data/train/pop/fr/minority.csv', sep=';')
data.rename(columns={
    'Corse du sud': 'Corse-du-Sud',
    'Haute Corse': 'Haute-Corse',
    "Côtes d'Armor": "Côtes-d'Armor"
},
            inplace=True)
df = pd.read_csv(
    'data/train/all_data_merged/fr/Enriched_Covid_history_data.csv')
depts = pd.read_csv('data/train/pop/fr/departements-francais.csv', sep=';')
depts_list = [element for element in depts['NOM']]
dic = {
    k: ('Unknown' if data[k][0] == 'nd' else
        float(data[k][0].replace("\u202f", '')) if k in depts_list else 'todo')
    for k in data.columns
}


def add_minority(row):
    return dic[row['nom']]


df['minority'] = df.apply(add_minority, axis=1)
df.to_csv(
    "data/train/all_data_merged/fr/Enriched_Covid_history_data_minority.csv",
    index=False)
print(df)
