import pandas as pd

print("Processing Comorbidities Data...")
df = pd.read_excel("../data/train/comorbidities/fr/2019_ALD-prevalentes-par-departement_serie-annuelle.xls", skiprows = 1)
df['Code département']=df['Code département'].astype(int)
df= df[df['Code département']<203]
df = df[["Code département", 'Insuffisance respiratoire chronique grave (ALD14)','Insuffisance cardiaque grave, troubles du rythme graves, cardiopathies valvulaires graves, cardiopathies congénitales graves (ALD5)']]
print(df)
print(df.columns)
df2 = pd.read_csv("../data/train/all_data_merged/fr/Enriched_Covid_history_data.csv")
df2 = df2.merge(df, how ="inner", left_on = "numero", right_on = "Code département")
print(df2)
df2.to_csv("../data/train/all_data_merged/fr/Enriched_Covid_history_data.csv", index = False)