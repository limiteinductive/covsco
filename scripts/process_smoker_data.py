import pandas as pd 

print("Processing Smokers Data...")
df = pd.read_csv("../data/train/smoker/fr/smoker_regions_departements.csv", sep =';')
df2 = pd.read_csv("../data/train/all_data_merged/fr/Enriched_Covid_history_data.csv")
print(df)
df["depnum"]=df["depnum"].astype(int)
df["Smokers"]=df["Smokers"].astype(float)
df2 = df2.merge(df, how = "inner", left_on = "numero", right_on = "depnum")
df2.to_csv("../data/train/all_data_merged/fr/Enriched_Covid_history_data.csv", index = False)
print(df2)