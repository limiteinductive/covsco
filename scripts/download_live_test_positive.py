import pandas as pd
import numpy as np
import requests


class LiveTest:
    def __init__(self):
        self.data = None

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
        if type(row['dep']) != int:
            row['dep'] = int(row['dep'].replace('2A',
                                                '201').replace('2B', '202'))
        return None

    def P_to_int(self, row):
        return int(row['P'])

    def preprocess_positive_test(self):
        self.data.apply(self.replace_dep, axis=1)
        self.data['P'] = self.data.apply(self.P_to_int, axis=1)

        self.data = self.data[self.data["dep"] < 203]
        self.data["jour"] = pd.to_datetime(self.data["jour"], dayfirst=True)
        self.data = self.data[self.data['jour'] == self.data['jour'].max()]
        self.data = self.data.groupby('dep').sum().reset_index()
        return self
