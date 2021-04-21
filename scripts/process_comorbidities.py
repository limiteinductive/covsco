import pandas as pd

class process_comorbidities_data:

    def __init__(self):

        self.df = None
        self.df2 = None

    def process_comorbidities(self):

        print("Processing Comorbidities Data...")
        self.df = pd.read_excel("../data/train/comorbidities/fr/2019_ALD-prevalentes-par-departement_serie-annuelle.xls", skiprows = 1)
        self.df['Code département']=self.df['Code département'].astype(int)
        self.df= self.df[self.df['Code département']<203]
        self.df = self.df[["Code département", 'Insuffisance respiratoire chronique grave (ALD14)','Insuffisance cardiaque grave, troubles du rythme graves, cardiopathies valvulaires graves, cardiopathies congénitales graves (ALD5)']]
        print(self.df)
        print(self.df.columns)
        self.df2 = pd.read_csv("../data/train/all_data_merged/fr/Enriched_Covid_history_data.csv")
        self.df2 = self.df2.merge(self.df, how ="inner", left_on = "numero", right_on = "Code département")
        print(self.df2)
        self.df2.to_csv("../data/train/all_data_merged/fr/Enriched_Covid_history_data.csv", index = False)

if __name__ == '__main__':

    ProcessComorbidities = process_comorbidities_data()
    ProcessComorbidities.process_comorbidities()