import pandas as pd
from datetime import timedelta
from datetime import datetime
from operator import itemgetter
import time

def max_normalize(x):
    return (x - x.min()) / (x.max() - x.min())

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

data["normpm25"]=max_normalize(data["pm25"])
data["normno2"]=max_normalize(data["no2"])
data["normo3"]=max_normalize(data["o3"])
data["normpm10"]=max_normalize(data["pm10"])
data["normco"]=max_normalize(data["co"])

data["time"]=pd.to_datetime(data["time"])

pm25tuple = (data['numero'], data['time'], data["pm25"], data["normpm25"] )
no2tuple = (data['numero'], data['time'], data["no2"], data["normno2"])
o3tuple = (data['numero'], data['time'], data["o3"], data["normo3"])
pm10tuple = (data['numero'], data['time'],data["pm10"], data["normpm10"])
cotuple = (data['numero'], data['time'], data["co"], data["normco"])

dicpm25 = {(i, j) : (k,l) for (i, j, k, l) in zip(*pm25tuple)}
dicno2 = {(i, j) : (k,l)  for (i, j, k, l) in zip(*no2tuple)}
dico3 = {(i, j) : (k,l)  for (i, j, k, l) in zip(*o3tuple)}
dicpm10 = {(i, j) : (k,l)  for (i, j, k, l) in zip(*pm10tuple)}
dicco = {(i, j) : (k,l) for (i, j, k, l) in zip(*cotuple)}

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

    return (max(cleanedList,key=itemgetter(0))[0][0],\
            max(cleanedList,key=itemgetter(1))[1][0],\
            max(cleanedList,key=itemgetter(2))[2][0],\
            max(cleanedList,key=itemgetter(3))[3][0],\
            max(cleanedList,key=itemgetter(4))[4][0],\
            max(cleanedList,key=itemgetter(0))[0][1],\
            max(cleanedList,key=itemgetter(1))[1][1],\
            max(cleanedList,key=itemgetter(2))[2][1],\
            max(cleanedList,key=itemgetter(3))[3][1],\
            max(cleanedList,key=itemgetter(4))[4][1]\
             )

@simple_time_tracker
def compute_1M_MCs_assign_to_df(data):
    print("Computing maximal pollution concentrations.. ")
    data[['1MMaxpm25','1MMaxno2','1MMaxo3','1MMaxpm10','1MMaxco',\
            '1MMaxnormpm25','1MMaxnormno2','1MMaxnormo3','1MMaxnormpm10','1MMaxnormco']] \
                = data.apply(compute_1M_Max_conc, axis=1).apply(pd.Series)
    print("\n")
    return data

data =  compute_1M_MCs_assign_to_df(data)
print(data)

data.to_csv('../data/train/all_data_merged/fr/Enriched_Covid_history_data.csv', index = False)

    




