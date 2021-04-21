import pandas as pd
from utilities import download_url

class process_hist_vaccination_data:

    def __init__(self):

        self.url = None
        self.df = None
        self.df2 = None
        self.dfvac1 = None
        self.dfvac2 = None
        self.cum1 = None
        self.cum2 = None
    
    def create_week(self, row):
        return pd.date_range(start=row['date_debut_semaine'], periods=7).tolist()

    def check_vaccin(self, v_row, date):
        if date in v_row['7_days']:
            return v_row['nb']

    def enriched_vaccin(self, row):
        date = row['date']
        depnum = row['numero']
        referencedate1 = self.dfvac1['date_debut_semaine'].min()
        #referencedate2 = self.dfvac2['date_debut_semaine'].min()

        if date < referencedate1:
            (first1, first2) = (0, 0)
        else:
            cum1_dep = self.cum1[self.cum1['departement'] == depnum]
            res1 = cum1_dep.apply(self.check_vaccin, date=date, axis=1)
            first1 = [el for el in res1
                    if el == el][0]  #get the first non null element of res

            cum2_dep = self.cum2[self.cum2['departement'] == depnum]
            res2 = cum2_dep.apply(self.check_vaccin, date=date, axis=1)
            first2 = next((el for el in res2 if el == el),
                        None)  #get the first non null element of res
            if first2 is None:
                first2 = 0

        return ((first1, first2))
 
    def process_hist_vaccination(self):

        self.url = "https://www.data.gouv.fr/es/datasets/r/59aeab47-c364-462c-9087-ce233b6acbbc"

        download_url(self.url, "../data/train/vaccination/fr/vaccination_hist_data.csv", chunk_size=128)

        self.df = pd.read_csv("../data/train/vaccination/fr/vaccination_hist_data.csv")
        print(self.df.columns)
        self.df['departement'] = self.df['departement'].replace({
            '2A': '201',
            '2B': '202'
        }).astype(int)
        self.df = self.df[self.df['departement'] < 203]
        self.df["date_debut_semaine"] = pd.to_datetime(self.df["date_debut_semaine"],
                                                dayfirst=True)

        self.df2 = pd.read_csv(
            "../data/train/all_data_merged/fr/Enriched_Covid_history_data.csv",
            sep=",")
        self.df2['vac1nb']=0
        self.df2['vac2nb']=0
        self.df2["date"] = pd.to_datetime(self.df2["date"])

        self.dfvac1 = self.df[self.df["rang_vaccinal"] == 1].reset_index()
        self.dfvac2 = self.df[self.df["rang_vaccinal"] == 2].reset_index()

        self.cum1 = self.dfvac1.groupby(['departement', 'date_debut_semaine']).sum().groupby(
            level=0).cumsum().sort_values("date_debut_semaine").reset_index().drop(
                columns="index")
        self.cum2 = self.dfvac2.groupby(['departement', 'date_debut_semaine']).sum().groupby(
            level=0).cumsum().sort_values("date_debut_semaine").reset_index().drop(
                columns="index")

        self.cum1['7_days'] = self.cum1.apply(self.create_week, axis=1)
        self.cum2['7_days'] = self.cum2.apply(self.create_week, axis=1)

        self.df2[['vac1nb','vac2nb' ]] = self.df2.apply(self.enriched_vaccin, axis=1).apply(pd.Series)
        print(self.df2)
        self.df2.to_csv("../data/train/all_data_merged/fr/Enriched_Covid_history_data.csv",
                index=False)

if __name__ == '__main__':
    ProcessVaccination = process_hist_vaccination_data()
    ProcessVaccination.process_hist_vaccination()
