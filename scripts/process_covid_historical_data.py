import pandas as pd

class process_covid_historical_data:
    def __init__(self):

        self.file_name = None
        self.file_name2 = None
        self.covid = None
        self.population = None

    def process_covid_hist_data(self):
        print('Processing Covid data ... ', flush=True, end='')
        self.file_name = '../data/train/covid/fr/Covid_data_history.csv'
        self.covid = pd.read_csv(self.file_name, sep=',').dropna()
        self.covid['date'] = pd.to_datetime(self.covid['date'])
        self.covid = self.covid[self.covid['numero']<203]
        self.population  = pd.read_csv('../data/train/pop/fr/population_2020.csv')
        self.covid = self.covid.merge(self.population, how='inner', left_on='numero', right_on='dep_num')
        print('OK', flush=True)
        print("\n")

if __name__ == '__main__':

    ProcessCovidHistoricalData = process_covid_historical_data()
    ProcessCovidHistoricalData.process_covid_hist_data()