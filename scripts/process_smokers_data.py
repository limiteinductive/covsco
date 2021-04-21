import pandas as pd


class process_smokers_data:
    def __init__(self):

        self.df = None
        self.df2 = None

    def process_smokers(self):
        print("Processing Smokers Data...")
        self.df = pd.read_csv("../data/train/smoker/fr/smoker_regions_departements.csv", sep =';')
        self.df2 = pd.read_csv("../data/train/all_data_merged/fr/Enriched_Covid_history_data.csv")
        print(self.df)
        self.df["depnum"]=self.df["depnum"].astype(int)
        self.df["Smokers"]=self.df["Smokers"].astype(float)
        self.df2 = self.df2.merge(self.df, how = "inner", left_on = "numero", right_on = "depnum")
        self.df2.to_csv("../data/train/all_data_merged/fr/Enriched_Covid_history_data.csv", index = False)
        print(self.df2)
        print("\n")
        return None

if __name__ == '__main__':
    ProcessSmokers = process_smokers_data()
    ProcessSmokers.process_smokers()

