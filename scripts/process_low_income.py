import pandas as pd

df = pd.read_csv('../data/train/low_income/fr/low_income.csv', sep=';')

df.rename(columns={
    'Corse du sud': 'Corse-du-Sud',
    'Haute Corse': 'Haute-Corse',
    "Côtes d'Armor": "Côtes-d'Armor"
},
          inplace=True)

data = pd.read_csv(
    '../data/train/all_data_merged/fr/Enriched_Covid_history_data.csv')
depts = pd.read_csv('../data/train/pop/fr/departements-francais.csv', sep=';')

depts_list = [element for element in depts['NOM']]

pauvrete_dic = {
    k: ('Unknown' if df[k][0] == 'nd' else float(df[k][0].replace(
        "\u202f", '.').replace(',', '.')) if k in depts_list else 'todo')
    for k in df.columns
}
rsa_dic = {
    k: ('Unknown' if df[k][1] == 'nd' else float(df[k][1].replace(
        "\u202f", '.').replace(',', '.')) if k in depts_list else 'todo')
    for k in df.columns
}
ouvriers_dic = {
    k: ('Unknown' if df[k][2] == 'nd' else float(df[k][2].replace(
        "\u202f", '.').replace(',', '.')) if k in depts_list else 'todo')
    for k in df.columns
}


def add_feature(row, feature):
    return feature[row['nom']]


data['pauvrete'] = data.apply(add_feature, axis=1, feature=pauvrete_dic)
data['rsa'] = data.apply(add_feature, axis=1, feature=rsa_dic)
data['ouvriers'] = data.apply(add_feature, axis=1, feature=ouvriers_dic)

print(data)
data.to_csv("../data/train/all_data_merged/fr/Enriched_Covid_history_data.csv",
          index=False)
