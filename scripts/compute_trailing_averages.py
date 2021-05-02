# Compute the engineered features: 7 trailing day averages of gas' concentrations
# and the previous' day total hospitalizations

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

print("Computing the engineered features: 7 trailing day averages of gas' concentrations ...")
data = pd.read_csv('/home/ludo915/code/covsco/data/train/all_data_merged/fr/Enriched_Covid_history_data.csv')

data["time"]=pd.to_datetime(data["time"])

print(data)

pm25tuple = (data['numero'], data['time'], data["pm25"])
no2tuple = (data['numero'], data['time'], data["no2"])
o3tuple = (data['numero'], data['time'], data["o3"])
pm10tuple = (data['numero'], data['time'], data["pm10"])
cotuple = (data['numero'], data['time'], data["co"])
tothospituple = (data['numero'], data['time'], data["hospi"])
covidpostesttuple = (data['numero'], data['time'], data["'covidpostestprevday'"])

dicpm25 = {(i, j) : k for (i, j, k) in zip(*pm25tuple)}
dicno2 = {(i, j) : k for (i, j, k) in zip(*no2tuple)}
dico3 = {(i, j) : k for (i, j, k) in zip(*o3tuple)}
dicpm10 = {(i, j) : k for (i, j, k) in zip(*pm10tuple)}
dicco = {(i, j) : k for (i, j, k) in zip(*cotuple)}
dictothospi = {(i, j) : k for (i, j, k) in zip(*tothospituple)}
diccovidpostest = {(i, j) : k for (i, j, k) in zip(*covidpostesttuple)}

referencedate = data["time"].min()

def compute_avg_conc(row):
    datalist = []
    datalist2 = []
    date = row['time'] - pd.Timedelta("30 days")
    date2 = row['time'] - pd.Timedelta("6 days")
    dateprevday = row['time'] - pd.Timedelta("1 days") 
    dates = pd.date_range(start = date2, periods=7).tolist()
    dates2 = pd.date_range(start = date, periods=31).tolist()
    
    for valuedate in dates:
        if(valuedate < referencedate):
            datalist.append(('NaN','NaN','NaN','NaN','NaN'))
        
        else:
            datalist.append((dicpm25[(row['numero'], pd.to_datetime(str(valuedate)))], \
                            dicno2[(row['numero'], pd.to_datetime(str(valuedate)))], \
                            dico3[(row['numero'], pd.to_datetime(str(valuedate)))], \
                            dicpm10[(row['numero'], pd.to_datetime(str(valuedate)))],\
                            dicco[(row['numero'], pd.to_datetime(str(valuedate)))]),
                            )
        
    for valuedate in dates2:
        if(valuedate < referencedate):
            datalist2.append(('NaN','NaN','NaN','NaN','NaN'))
        
        else:
            datalist2.append((dicpm25[(row['numero'], pd.to_datetime(str(valuedate)))], \
                            dicno2[(row['numero'], pd.to_datetime(str(valuedate)))], \
                            dico3[(row['numero'], pd.to_datetime(str(valuedate)))], \
                            dicpm10[(row['numero'], pd.to_datetime(str(valuedate)))],\
                            dicco[(row['numero'], pd.to_datetime(str(valuedate)))]))

    if (dateprevday < referencedate):
        prevdaytothospi = "NaN"
        prevdaycovidpostest = "Nan"
    else:
        prevdaytothospi = dictothospi[(row['numero'], dateprevday)]
        prevdaycovidpostest = diccovidpostest[(row['numero'], dateprevday)]

    cleanedList = [(float(x),float(y),float(z),float(w),float(v)) for (x,y,z,w, v) in datalist if (str(x),str(y),str(z),str(w),str(v)) != ('NaN','NaN','NaN','NaN','NaN')]
    cleanedList2 = [(float(x),float(y),float(z),float(w),float(v)) for (x,y,z,w, v) in datalist2 if (str(x),str(y),str(z),str(w),str(v)) != ('NaN','NaN','NaN','NaN','NaN')]
  
    means = [sum(ele) / len(cleanedList) for ele in zip(*cleanedList) if ele != 'NaN']
    means2 = [sum(ele) / len(cleanedList2) for ele in zip(*cleanedList2) if ele != 'NaN']
    return (means[0],\
            means[1],\
            means[2],\
            means[3],\
            means[4],\
            means2[0],\
            means2[1],\
            means2[2],\
            means2[3],\
            means2[4],
            prevdaytothospi)

@simple_time_tracker
def compute_trailing_avg_assign_to_df(dfpar):
    print("Computing 7D & 1M trailing averages in pollution concentrations & the previous' day total hospitalisations.. ")
    dfpar[['pm257davg','no27davg','o37davg', 'pm107davg','co7davg',\
            'pm251Mavg','no21Mavg','o31Mavg','pm101Mavg','co1Mavg',"hospiprevday"]] = dfpar.apply(compute_avg_conc, axis=1).apply(pd.Series)
    return dfpar

print("\n")

data = compute_trailing_avg_assign_to_df(data)
print(data)

data.to_csv('/home/ludo915/code/covsco/data/train/all_data_merged/fr/Enriched_Covid_history_data.csv', index = False)

    
