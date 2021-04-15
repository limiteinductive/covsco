import pandas as pd
import time

def simple_time_tracker(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int(te - ts)
        else:
            print(method.__name__, round(te - ts, 2))
        return result
    return timed

df = pd.read_csv('../data/train/all_data_merged/fr/Enriched_Covid_history_data.csv')
df["time"]=pd.to_datetime(df["time"])
minpm25 = df["pm25"].min()
print(minpm25)
maxpm25 = df["pm25"].max()
print(maxpm25)

increment = (maxpm25 - minpm25)/4
pm25levelslist =[]
for i in range(5):
    pm25levelslist.append(minpm25 + i * increment)
print(pm25levelslist)

def pm25levels(row):
    pm25 = row['pm25']
    
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

@simple_time_tracker
def pm25levels_to_df(data):
    data[["pm25level","pm25levelstring"]] \
                = data.apply(pm25levels, axis=1).apply(pd.Series)
    print("\n")
    return data

df =  pm25levels_to_df(df)
print(df)
print(df[(df['time']==pd.to_datetime('2021-03-31')) & ((df["pm25levelstring"]=="High") | (df["pm25levelstring"]=="Very High"))][["nom","pm25levelstring"]])
df.to_csv("../data/train/all_data_merged/fr/Enriched_Covid_history_data.csv", index = False)
print('OK')