import pandas as pd
import numpy as np

from utilities import download_url


class LiveVariant:
    def __init__(self):
        self.url = None
        self.file_name = None
        self.V1 = None
        self.V2 = None
        self.data = None
        self.initial_data = None
        self.date_max = None

    def get_file(self):
        self.url = 'https://www.data.gouv.fr/fr/datasets/r/16f4fd03-797f-4616-bca9-78ff212d06e8'
        self.file_name = '../data/prediction/live_variants.csv'
        download_url(self.url, self.file_name)
        return self

    def replace_dep(elf, row):
        if type(row['dep']) != int:
            return int(row['dep'].replace('2A', '201').replace('2B', '202'))

    def get_semaine(self, row):
        return pd.to_datetime(row['semaine'][-10:])

    def add_V1(self, row):
        correct_dep = self.data[self.data['dep'] == row['dep_num']]
        V1 = correct_dep[correct_dep['semaine'] ==
                         self.date_max]['Nb_susp_501Y_V1']
        return int(list(V1)[0])

    def add_V2(self, row):
        correct_dep = self.data[self.data['dep'] == row['dep_num']]
        V2 = correct_dep[correct_dep['semaine'] ==
                         self.date_max]['Nb_susp_501Y_V2_3']
        return int(list(V2)[0])

    def preprocess_variant(self):
        self.data = pd.read_csv(self.file_name, sep=';', low_memory=False)
        self.data = self.data[[
            'dep', 'semaine', 'Nb_susp_501Y_V1', 'Nb_susp_501Y_V2_3'
        ]]

        self.data['dep'] = self.data.apply(self.replace_dep, axis=1)
        self.data = self.data[self.data['dep'] < 203]
        self.data['dep'] = self.data['dep'].astype(int)

        self.data['semaine'] = self.data.apply(self.get_semaine, axis=1)
        self.data = self.data.groupby(['dep', 'semaine']).sum().reset_index()

        self.date_max = self.data['semaine'].max()
        return self

    def process_variant(self):
        self.initial_data = pd.read_csv(
            '../data/prediction/prediction_data.csv')
        self.initial_data['variant_1'] = self.initial_data.apply(self.add_V1,
                                                                 axis=1)
        self.initial_data['variant_2'] = self.initial_data.apply(self.add_V2,
                                                                 axis=1)
        self.initial_data.to_csv('../data/prediction/prediction_data.csv',
                                 index=False)
        return self

    def compiling(self):
        print('Getting file')
        self.get_file()
        print('Preprocessing live_variant')
        self.preprocess_variant()
        print('Processing live_variant with prediction_data')
        self.process_variant()
        return self


if __name__ == '__main__':
    variant = LiveVariant()
    variant.compiling()
