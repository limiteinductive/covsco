import pandas as pd

class process_population_rectified:

    def __init__(self):
        self.data = None
        self.dic2020 = None
        self.dic2021 = None

    def rectify_pop(self, row):
        if row['date'][:4] == '2020':
            ans = self.dic2020[row['nom']]
        elif row['date'][:4] == '2021':
            ans = self.dic2021[row['nom']]
        return ans

    def process_population_rect(self):
        self.data = pd.read_csv(
            '/home/ludo915/code/covsco/data/train/all_data_merged/fr/Enriched_Covid_history_data.csv')
        pop2020 = pd.read_excel('/home/ludo915/code/covsco/data/train/pop/fr/popfr.xls', sheet_name='2020')
        pop2021 = pd.read_excel('/home/ludo915/code/covsco/data/train/pop/fr/popfr.xls', sheet_name='2021')

        self.dic2020 = {k: v for k, v in zip(pop2020['depname'], pop2020['pop'])}
        self.dic2021 = {k: v for k, v in zip(pop2021['depname'], pop2021['pop'])}
        self.data['total'] = self.data.apply(self.rectify_pop, axis=1)
        self.data['idx'] = self.data.apply(self.rectify_pop, axis=1)
        self.data.to_csv(
            '/home/ludo915/code/covsco/data/train/all_data_merged/fr/Enriched_Covid_history_data.csv',
            index=False)
        print(self.data)
        return None

if __name__ == '__main__':

    ProcessPopRectified = process_population_rectified()
    ProcessPopRectified.process_population_rect()


