import pandas as pd
import time
from tqdm import tqdm
import io

class compute_pollution_levels:
    def __init__(self):
        self.df = None
    
    def pm25levels(self, row):
        pm25 = row['pm25']
        minpm25 = self.df["pm25"].min()
        maxpm25 = self.df["pm25"].max()
        increment = (maxpm25 - minpm25)/4
        pm25levelslist =[]
        for i in range(5):
            pm25levelslist.append(minpm25 + i * increment)
        if (pm25 <= pm25levelslist[1]):
            level = 0
            levelstring = "Low"
        elif ((pm25levelslist[1] < pm25) & (pm25levelslist[2] >= pm25)):
            level = 1
            levelstring = "Medium"
        elif ((pm25levelslist[2] < pm25) & (pm25levelslist[3] >= pm25)):
            level = 2
            levelstring = "High"
        else:
            level = 3
            levelstring = "Very High"
            
        return (level, levelstring)
    
    def pm25levels_to_df(self, data):
            data[["pm25level","pm25levelstring"]] \
                        = data.apply(self.pm25levels, axis=1).apply(pd.Series)
            print("\n")
            return data

    def compute_levels(self):
        print("Computing pm2.5 Pollutions levels")
        self.df = pd.read_csv('/home/ludo915/code/covsco/data/train/all_data_merged/fr/Enriched_Covid_history_data.csv')
        self.df["date"]=pd.to_datetime(self.df["date"])
        self.df =  self.pm25levels_to_df(self.df)
        print(self.df)
        print(self.df[(self.df['date']==self.df["date"].max()) & ((self.df["pm25levelstring"]=="High") | (self.df["pm25levelstring"]=="Very High"))][["nom","pm25levelstring"]])
        self.df.to_csv("/home/ludo915/code/covsco/data/train/all_data_merged/fr/Enriched_Covid_history_data.csv", index = False)
        print('OK')

if __name__ == "__main__":

    ComputePollutionLevels = compute_pollution_levels()
    ComputePollutionLevels.compute_levels()

