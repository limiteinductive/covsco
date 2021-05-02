import pandas as pd
import numpy as np
import requests
import csv


class LiveTest:
    def __init__(self):
        self.data = None
        self.initial_data = None

    def get_data(self):
        url_positive = 'https://www.data.gouv.fr/en/datasets/r/406c6a23-e283-4300-9484-54e78c8ae675'

        with requests.Session() as s:
            download = s.get(url_positive)

            decoded_content = download.content.decode('utf-8')

            cr = csv.reader(decoded_content.splitlines(), delimiter=';')
            my_list = list(cr)
        self.data = pd.DataFrame(my_list[1:], columns=my_list[0])
        return self

    def replace_dep(self, row):  
        return int(row['dep'].replace('2A','201').replace('2B', '202')) 

    def P_to_int(self, row):
        return int(row['P'])

    def preprocess_positive_test(self):
        self.data['dep'] = self.data.apply(self.replace_dep, axis=1)
        self.data['P'] = self.data.apply(self.P_to_int, axis=1)

        self.data = self.data[self.data["dep"] < 203]
        self.data["jour"] = pd.to_datetime(self.data["jour"], dayfirst=True)
        self.data = self.data[self.data['jour'] == self.data['jour'].max()]
        self.data = self.data.groupby('dep').sum()
        return self

    def add_positive_test(self, row):
        dep = row['dep_num']
        return self.data.loc[dep, 'P']

    def process_positive_test(self):
        self.initial_data = pd.read_csv(
            '/home/ludo915/code/covsco/data/prediction/prediction_data.csv')
        self.initial_data['positive_test'] = self.initial_data.apply(
            self.add_positive_test, axis=1)
        self.initial_data.to_csv('/home/ludo915/code/covsco/data/prediction/prediction_data.csv',
                                 index=False)
        return self

    def compiling(self):
        print('get file')
        self.get_data()
        print('preprocess positive test')
        self.preprocess_positive_test()
        print('process positive test')
        self.process_positive_test()

        return self


if __name__ == '__main__':
    positive = LiveTest()
    positive.compiling()
