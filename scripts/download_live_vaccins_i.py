import pandas as pd
import numpy as np

from utilities import download_url

url_positive_test = 'https://www.data.gouv.fr/es/datasets/r/59aeab47-c364-462c-9087-ce233b6acbbc'
download_url(url_positive_test, '../data/prediction/live_vaccins.csv')

live_vacc = pd.read_csv('../data/prediction/live_vaccins.csv')
live_vacc['date_debut_semaine'] = pd.to_datetime(
    vacc_1_reg['date_debut_semaine'])
date_max = vacc_1_reg['date_debut_semaine'].max()

vacc_1 = live_vacc[live_vacc['rang_vaccinal'] == 1]
vacc_2 = live_vacc[live_vacc['rang_vaccinal'] == 2]


def live_vacc_1(row):
    dep = row['dep_num']
    vacc_1_reg = vacc_1[vacc_1['code_region'] == dep]
    if vacc_1_reg.shape[0] != 0:
        nb_series = vacc_1_reg[vacc_1_reg['date_debut_semaine'] ==
                               date_max]['nb']
        nb = list(nb_series)[0]
    else:
        nb = 0
    return nb


def live_vacc_2(row):
    dep = row['dep_num']
    vacc_2_reg = vacc_2[vacc_2['code_region'] == dep]
    if vacc_2_reg.shape[0] != 0:
        nb_series = vacc_2_reg[vacc_2_reg['date_debut_semaine'] ==
                               date_max]['nb']
        nb = list(nb_series)[0]
    else:
        nb = 0
    return nb


prediction_data['vacc_1'] = prediction_data.apply(live_vacc_1, axis=1)
prediction_data['vacc_2'] = prediction_data.apply(live_vacc_2, axis=1)
prediction_data.to_csv('../data/predictiondata.csv', index=False)
