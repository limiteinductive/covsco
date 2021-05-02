import pandas as pd

class process_low_income:
    def __init__(self):
        self.df = None
        self.data = None
        self.depts = None
    
    def add_feature(self, row, feature):
        return feature[row['nom']]

    def process_li(self):
        self.df = pd.read_csv('/home/ludo915/code/covsco/data/train/low_income/fr/low_income.csv', sep=';')

        self.df.rename(columns={
            'Corse du sud': 'Corse-du-Sud',
            'Haute Corse': 'Haute-Corse',
            "Côtes d'Armor": "Côtes-d'Armor"
        },
                inplace=True)

        self.data = pd.read_csv(
            '/home/ludo915/code/covsco/data/train/all_data_merged/fr/Enriched_Covid_history_data.csv')
        self.depts = pd.read_csv('/home/ludo915/code/covsco/data/train/pop/fr/departements-francais.csv', sep=';')

        depts_list = [element for element in self.depts['NOM']]

        pauvrete_dic = {
            k: ('Unknown' if self.df[k][0] == 'nd' else float(self.df[k][0].replace(
                "\u202f", '.').replace(',', '.')) if k in depts_list else 'todo')
            for k in self.df.columns
        }
        rsa_dic = {
            k: ('Unknown' if self.df[k][1] == 'nd' else float(self.df[k][1].replace(
                "\u202f", '.').replace(',', '.')) if k in depts_list else 'todo')
            for k in self.df.columns
        }
        ouvriers_dic = {
            k: ('Unknown' if self.df[k][2] == 'nd' else float(self.df[k][2].replace(
                "\u202f", '.').replace(',', '.')) if k in depts_list else 'todo')
            for k in self.df.columns
        }

        self.data['pauvrete'] = self.data.apply(self.add_feature, axis=1, feature=pauvrete_dic)
        self.data['rsa'] = self.data.apply(self.add_feature, axis=1, feature=rsa_dic)
        self.data['ouvriers'] = self.data.apply(self.add_feature, axis=1, feature=ouvriers_dic)

        print(self.data)
        self.data.to_csv("/home/ludo915/code/covsco/data/train/all_data_merged/fr/Enriched_Covid_history_data.csv",
                index=False)

if __name__ == '__main__':
    ProcessLowIncome = process_low_income()
    ProcessLowIncome.process_li()
