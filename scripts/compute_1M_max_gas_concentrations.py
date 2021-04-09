import pandas as pd
from datetime import timedelta
from datetime import datetime
from operator import itemgetter
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

print("Computing 1M trailing maximal pollution concentrations.. ")
data = pd.read_csv('../data/train/all_data_merged/fr/Enriched_Covid_history_data.csv')

data["time"]=pd.to_datetime(data["time"])

pm25tuple = (data['numero'], data['time'], data["pm25"])
no2tuple = (data['numero'], data['time'], data["no2"])
o3tuple = (data['numero'], data['time'], data["o3"])
pm10tuple = (data['numero'], data['time'], data["pm10"])
cotuple = (data['numero'], data['time'], data["co"])

dicpm25 = {(i, j) : k for (i, j, k) in zip(*pm25tuple)}
dicno2 = {(i, j) : k for (i, j, k) in zip(*no2tuple)}
dico3 = {(i, j) : k for (i, j, k) in zip(*o3tuple)}
dicpm10 = {(i, j) : k for (i, j, k) in zip(*pm10tuple)}
dicco = {(i, j) : k for (i, j, k) in zip(*cotuple)}

referencedate = data["time"].min()
def compute_1M_Max_conc(row):
    datalist = []
    date2 = row['time'] - pd.Timedelta("30 days")


    dates = pd.date_range(start = date2, periods=31).tolist()

    for valuedate in dates:
        if(valuedate < referencedate):
            datalist.append(('NaN','NaN','NaN','NaN','NaN'))
        
        else:
            datalist.append((dicpm25[(row['numero'], pd.to_datetime(str(valuedate)))], \
                            dicno2[(row['numero'], pd.to_datetime(str(valuedate)))], \
                            dico3[(row['numero'], pd.to_datetime(str(valuedate)))], \
                            dicpm10[(row['numero'], pd.to_datetime(str(valuedate)))],\
                            dicco[(row['numero'], pd.to_datetime(str(valuedate)))]))
    
    cleanedList = [(x,y,z,w,v) for (x,y,z,w,v) in datalist if (str(x),str(y),str(z),str(w),str(v)) != ('NaN','NaN','NaN','NaN','NaN')]

    return (max(cleanedList,key=itemgetter(0))[0],\
            max(cleanedList,key=itemgetter(1))[1],\
            max(cleanedList,key=itemgetter(2))[2],\
            max(cleanedList,key=itemgetter(3))[3],\
            max(cleanedList,key=itemgetter(4))[4])

@simple_time_tracker
def compute_1M_MCs_assign_to_df(data):
    print("Computing maximal pollution concentrations.. ")
    data[['1MMaxpm25','1MMaxno2','1MMaxo3','1MMaxpm10','1MMaxco']] = data.apply(compute_1M_Max_conc, axis=1).apply(pd.Series)
    print("\n")
    return data

data =  compute_1M_MCs_assign_to_df(data)
print(data)

data.to_csv('../data/train/all_data_merged/fr/Enriched_Covid_history_data.csv', index = False)

    




