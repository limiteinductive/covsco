import pandas as pd
import numpy as np

from utilities import download_url


class LiveVaccin:
    def __init__(self):
        self.url = None
        self.vacc_1 = None
        self.vacc_2 = None
        self.initial_data = None
        self.date_max = None

    def get_file(self):
        self.url = 'https://www.data.gouv.fr/es/datasets/r/59aeab47-c364-462c-9087-ce233b6acbbc'
        download_url(self.url, '/home/ludo915/code/covsco/data/prediction/live_vaccins.csv')
        return self

    def preprocess_vaccin(self):
        data = pd.read_csv('/home/ludo915/code/covsco/data/prediction/live_vaccins.csv')
        data['date_debut_semaine'] = pd.to_datetime(data['date_debut_semaine'])
        self.date_max = data['date_debut_semaine'].max()

        self.vacc_1 = data[data['rang_vaccinal'] == 1]
        self.vacc_2 = data[data['rang_vaccinal'] == 2]
        return self

    def live_vacc_1(self, row):
        dep = row['dep_num']
        vacc_1_reg = self.vacc_1[self.vacc_1['code_region'] == dep]
        if vacc_1_reg.shape[0] != 0:
            nb_series = vacc_1_reg[vacc_1_reg['date_debut_semaine'] ==
                                   self.date_max]['nb']
            nb = list(nb_series)[0]
        else:
            nb = 0
        return nb

    def live_vacc_2(self, row):
        dep = row['dep_num']
        vacc_2_reg = self.vacc_2[self.vacc_2['code_region'] == dep]
        if vacc_2_reg.shape[0] != 0:
            nb_series = vacc_2_reg[vacc_2_reg['date_debut_semaine'] ==
                                   self.date_max]['nb']
            nb = list(nb_series)[0]
        else:
            nb = 0
        return nb

    def process_vacc(self):
        self.initial_data = pd.read_csv(
            '/home/ludo915/code/covsco/data/prediction/prediction_data.csv')
        self.initial_data['vacc_1'] = self.initial_data.apply(self.live_vacc_1,
                                                              axis=1)
        self.initial_data['vacc_2'] = self.initial_data.apply(self.live_vacc_2,
                                                              axis=1)
        self.initial_data.to_csv('/home/ludo915/code/covsco/data/prediction/prediction_data.csv',
                                 index=False)
        return self

    def compiling(self):
        print('Getting file')
        self.get_file()
        print('Preprocessing live_vaccin')
        self.preprocess_vaccin()
        print('Processing live_vaccin with prediction_data')
        self.process_vacc()
        return self


if __name__ == '__main__':
    vaccin = LiveVaccin()
    vaccin.compiling()
