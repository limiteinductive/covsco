import requests
import pandas as pd
import datetime
from tqdm import tqdm

url = "https://www.metaweather.com/api/location/search/?lattlong="
df = pd.read_csv("../data/train/all_data_merged/fr/Enriched_Covid_history_data.csv")
for i in tqdm(df.index):
    date = df.loc[i,"time"]
    if df.loc[i+1,"time"] != date:
        print ("CHANGEMENT DE VILLE")
    (lat,lon) = (df.loc[i,"latitude"],df.loc[i,"longitude"])
    url2 = url + str(lat) + "," + str(lon)
    response = requests.get(url2).json()
    print(response)
    referencedate = datetime.datetime.strptime("2020-04-08", '%Y-%m-%d')