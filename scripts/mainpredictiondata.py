### IMPORTS

from hdx.hdx_configuration import Configuration
from hdx.data.dataset import Dataset
import requests
import pandas as pd
import numpy as np
import csv
from zipfile import ZipFile
import os
from utilities import download_url
from download_initial_live_prediction import InitialLive
from download_live_mobility_i import LiveMobility
from download_live_vaccins_i import LiveVaccin
from download_live_test_positive import LiveTest


class PredictionData:
    def __init__(self):
        pass

    def get_initial_file(self):
        """ Initialize prediction file with departmts_nb, name, region, capital, area, total, density """
        initial = InitialLive()
        initial.get_data()
        self.data = initial.data
        return self

    def add_mobility(self):
        """ADD LIVE MOBILITY"""
        live_mob = LiveMobility()
        print('Getting url')
        live_mob.get_url()
        print('Getting file')
        live_mob.get_file()
        print('Unzipping file')
        live_mob.unzipp()
        print('Preprocessing Mobility')
        live_mob.preprocess_mobility()

        print('Processing Mobility with Prediction')
        self.data['go_out'] = self.data.apply(live_mob.add_go_out, axis=1)
        self.data['stay_put'] = self.data.apply(live_mob.add_stay_put, axis=1)
        self.data.to_csv('../data/prediction/prediction_data.csv', index=False)
        return self

    def add_vaccination(self):
        """ add live vaccination """
        live_vacc = LiveVaccin()
        print('Getting file')
        live_vacc.get_file()
        print('Preprocessing Mobility')
        live_vacc.preprocess_vaccin()

        print('Processing Vaccin with Prediction')
        self.data['vacc_1'] = self.data.apply(live_vacc.live_vacc_1, axis=1)
        self.data['vacc_2'] = self.data.apply(live_vacc.live_vacc_2, axis=1)
        self.data.to_csv('../data/prediction/prediction_data.csv', index=False)
        return self

    def add_positive_test(self):
        livetest = LiveTest()
        print('Getting file')
        livetest.get_data()
        print('Preprocessing positive test')
        livetest.preprocess_positive_test()
        self.data['positive_test'] = self.data.apply(
            livetest.add_positive_test, axis=1)
        self.data.to_csv('../data/prediction/prediction_data.csv', index=False)
        return self


if __name__ == '__main__':
    prediction = PredictionData()

    print('1.Get initial data')
    prediction.get_initial_file()

    print('2.Add mobility')
    prediction.add_mobility()

    print('3.Add vaccination')
    prediction.add_vaccination()

    print('4.Add positive_tests')
    prediction.add_positive_test()

    print(prediction.data)
