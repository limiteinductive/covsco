import pandas as pd
from utilities import download_url
from datetime import datetime




class process_variants_hist_data:

    def __init__(self):

        self.df = None
        self.df2 = None
        self.url = None
        self.variantstuple = None
        self.dicvariant = None

    def extract_date(self, row):
        date = pd.to_datetime(row["semaine"][11:21], yearfirst=True)
        return date

    def enriched_variant(self, row):
        date = row['date']
        depnum = row['numero']
        week_days=["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        maxdate = self.df2["date"].max()
        now = datetime.now()
        referencedate = self.df['jour'].min()
        current_time_hour = int(now.strftime("%H"))
        
        if(((date == maxdate)|
            (date == maxdate - pd.Timedelta("1 days"))| 
            (date == maxdate - pd.Timedelta("2 days"))|
            (date == maxdate - pd.Timedelta("3 days"))) & ((week_days[date.weekday()]=="Saturday") |
                                                        (week_days[date.weekday()]=="Sunday") |
                                                        ((week_days[date.weekday()]=="Monday") & current_time_hour < 12))):

            if week_days[date.weekday()]=="Saturday":
                date = row['date'] - pd.Timedelta("2 days")
            elif (week_days[date.weekday()]=="Sunday"):
                date = row['date'] - pd.Timedelta("3 days")
            elif ( (week_days[date.weekday()]=="Monday") & current_time_hour < 12):
                date = row['date'] - pd.Timedelta("4 days")
        
        if date < referencedate:
            return (0, 0)
        else:
            return (self.dicvariant[(depnum,date)])
    def process_variants(self):

        print("Processing Variants data ...")

        self.url ="https://www.data.gouv.fr/fr/datasets/r/16f4fd03-797f-4616-bca9-78ff212d06e8"
        download_url(self.url, "../data/train/variants/fr/variants_hist_data.csv", chunk_size=128)

        self.df = pd.read_csv("../data/train/variants/fr/variants_hist_data.csv", sep=';')
        self.df2 = pd.read_csv("../data/train/all_data_merged/fr/Enriched_Covid_history_data.csv")

        self.df2["date"] = pd.to_datetime(self.df2["date"])
        self.df['dep'] = self.df['dep'].replace({'2A': '201', '2B': '202'}).astype(int)
        self.df = self.df[self.df['dep'] < 203]
        self.df = self.df.groupby(['dep', 'semaine'
                        ])[["dep", "semaine", "Nb_susp_501Y_V1", "Nb_susp_501Y_V2_3"
                            ]].sum().drop(columns=["dep"]).reset_index()

        self.df['jour'] = self.df.apply(self.extract_date, axis=1)
        self.df.drop(columns='semaine', inplace=True)

        self.variantstuple = (self.df['dep'], self.df['jour'], self.df["Nb_susp_501Y_V1"],self.df["Nb_susp_501Y_V2_3"] )
        self.dicvariant = {(i, j) : (k,l) for (i, j, k, l) in zip(*self.variantstuple)}

        self.df2[['Nb_susp_501Y_V1','Nb_susp_501Y_V2_3']] = self.df2.apply(self.enriched_variant, axis=1).apply(pd.Series)
        self.df2.sort_values(by = ["numero","date"], inplace = True)
        print(self.df2)
        self.df2.to_csv("../data/train/all_data_merged/fr/Enriched_Covid_history_data.csv",\
                index=False)
        return None


if __name__ == '__main__':
    ProcessVariantsHistoricalData = process_variants_hist_data()
    ProcessVariantsHistoricalData.process_variants()