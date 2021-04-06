import pandas as pd

data = pd.read_csv(
    '../data/train/all_data_merged/fr/Enriched_Covid_history_data.csv')
pop2020 = pd.read_excel('../data/train/pop/fr/popfr.xls', sheet_name='2020')
pop2021 = pd.read_excel('../data/train/pop/fr/popfr.xls', sheet_name='2021')

dic2020 = {k: v for k, v in zip(pop2020['depname'], pop2020['pop'])}
dic2021 = {k: v for k, v in zip(pop2021['depname'], pop2021['pop'])}


def rectify_pop(row):
    #print(row['jour'][:4])
    if row['jour'][:4] == '2020':
        ans = dic2020[row['name']]
    elif row['jour'][:4] == '2021':
        ans = dic2021[row['name']]
    return ans


data['total'] = data.apply(rectify_pop, axis=1)
data['idx'] = data.apply(rectify_pop, axis=1)
data.to_csv(
    '../data/train/all_data_merged/fr/Enriched_Covid_history_data.csv',
    index=False)
print(data)
