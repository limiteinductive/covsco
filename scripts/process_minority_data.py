import pandas as pd

class process_minority_data:

    def __init__(self):
        self.data = None
        self.dic = None
        self.depts = None

    def add_minority(self, row):
        return self.dic[row['nom']]

    def process_minority(self):

        self.data = pd.read_csv('../data/train/minority/fr/minority.csv', sep=';')
        self.data.rename(columns={
            'Corse du sud': 'Corse-du-Sud',
            'Haute Corse': 'Haute-Corse',
            "Côtes d'Armor": "Côtes-d'Armor"
        },
                    inplace=True)
        self.df = pd.read_csv(
            '../data/train/all_data_merged/fr/Enriched_Covid_history_data.csv')
        self.depts = pd.read_csv('../data/train/pop/fr/departements-francais.csv', sep=';')
        depts_list = [element for element in self.depts['NOM']]
        self.dic = {
            k: ('Unknown' if self.data[k][0] == 'nd' else
                float(self.data[k][0].replace("\u202f", '')) if k in depts_list else 'todo')
            for k in self.data.columns
        }

        self.df['minority'] = self.df.apply(self.add_minority, axis=1)
        print(self.df)
        self.df.to_csv("../data/train/all_data_merged/fr/Enriched_Covid_history_data.csv",
                index=False)
        print(self.df)

if __name__ == '__main__':

    ProcessMinority = process_minority_data()
    ProcessMinority.process_minority()