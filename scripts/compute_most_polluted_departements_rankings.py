import pandas as pd

df = pd.read_csv('../data/train/all_data_merged/fr/Enriched_Covid_history_data.csv')
pollutantslist = ["o3","co","no2","pm25","pm10","so2"]

with pd.ExcelWriter("../research/Pollution study by departement 1Y Max 1M-TMCs.xlsx") as writer:  
    
    for pollutant in pollutantslist:

        datalist =[]
        df2 = df.copy()
        df2 = df2[df["leadtime_hour"]==0]
        maximum_pollution_level = df2[(df2["1MMax"+pollutant]==df2["1MMax"+pollutant].max())]
        datalist.append((maximum_pollution_level["nom"].unique()[0],\
                        maximum_pollution_level["numero"].unique()[0],
                        maximum_pollution_level["date"].min(),\
                        maximum_pollution_level["1MMax"+pollutant].unique()[0],\
                        df2[df2["numero"]== maximum_pollution_level["numero"].unique()[0]]['totalcovidcasescumulated'].max(),\
                        maximum_pollution_level["idx"].unique()[0]))
        alreadyseen = maximum_pollution_level["nom"].unique()[0]
        print(maximum_pollution_level["numero"].unique())
        counter = 1



        while (counter !=95):
            counter += 1
            df2 = df2[df2["nom"]  != alreadyseen]
            maximum_pollution_level = df2[(df2["1MMax"+pollutant]==df2["1MMax"+pollutant].max())]
            datalist.append((maximum_pollution_level["nom"].unique()[0],\
                            maximum_pollution_level["numero"].unique()[0],
                            maximum_pollution_level["date"].min(),\
                            maximum_pollution_level["1MMax"+pollutant].unique()[0],\
                            df2[df2["nom"]==maximum_pollution_level["nom"].unique()[0]]['totalcovidcasescumulated'].max(),
                            maximum_pollution_level["idx"].unique()[0]))
                            
            
            alreadyseen = maximum_pollution_level["nom"].unique()[0]

        dfexport = pd.DataFrame(datalist)
        dfexport.columns=["Département","Numéro","Date of pollution peak","1MMax"+pollutant,'totalcovidcasescumulated',"Population Index"]
        print(dfexport)
        dfexport.to_excel(writer, sheet_name = pollutant, index = False)