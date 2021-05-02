import pandas as pd
from datetime import timedelta
from datetime import datetime
from operator import itemgetter
import time
import itertools

itertools.imap = lambda *args, **kwargs: list(map(*args, **kwargs))

def max_normalize(self, x):
    return (x - x.min()) / (x.max() - x.min())

def simple_time_tracker(self, method):
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

print("Computing then engineered features: 1M Trailing Maximal Pollution Concentrations (1M-TMPCs),1M Trailing Average Pollution Concentrations (1M-TAPCs), 7D Trailing Average Pollution Concentrations (7D-TAPCs) and the previous day's total hospitalizations...")
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
tothospituple = (data['numero'], data['time'], data["hospi"])

dicpm25 = {(i, j) : (k,l) for (i, j, k, l) in zip(*pm25tuple)}
dicno2 = {(i, j) : (k,l)  for (i, j, k, l) in zip(*no2tuple)}
dico3 = {(i, j) : (k,l)  for (i, j, k, l) in zip(*o3tuple)}
dicpm10 = {(i, j) : (k,l)  for (i, j, k, l) in zip(*pm10tuple)}
dicco = {(i, j) : (k,l) for (i, j, k, l) in zip(*cotuple)}
dictothospi = {(i, j) : k for (i, j, k) in zip(*tothospituple)}


referencedate = data["time"].min()
def compute_Engineered_Features(row):
    datalist = []
    datalist2 = []
    date = row['time'] - pd.Timedelta("30 days")
    date2 = row['time'] - pd.Timedelta("6 days")
    dateprevday = row['time'] - pd.Timedelta("1 days")

    dates = pd.date_range(start = date, periods=31).tolist()
    dates2 = pd.date_range(start = date2, periods=7).tolist()

    if (dateprevday < referencedate):
        prevdaytothospi = "NaN"
    else:
        prevdaytothospi = dictothospi[(row['numero'], dateprevday)] 

    for valuedate in dates:
        if(valuedate < referencedate):
            datalist.append((('NaN','Nan'),('NaN','Nan'),('NaN','Nan'),('NaN','Nan'),('NaN','Nan')))
        
        else:
            datalist.append((dicpm25[(row['numero'], pd.to_datetime(str(valuedate)))], \
                            dicno2[(row['numero'], pd.to_datetime(str(valuedate)))], \
                            dico3[(row['numero'], pd.to_datetime(str(valuedate)))], \
                            dicpm10[(row['numero'], pd.to_datetime(str(valuedate)))],\
                            dicco[(row['numero'], pd.to_datetime(str(valuedate)))]))

    if (dateprevday < referencedate):
        prevdaytothospi = "NaN"
    else:
        prevdaytothospi = dictothospi[(row['numero'], dateprevday)]

    for valuedate in dates2:
        if(valuedate < referencedate):
            datalist2.append((('NaN','Nan'),('NaN','Nan'),('NaN','Nan'),('NaN','Nan'),('NaN','Nan')))
        
        else:
            datalist2.append((dicpm25[(row['numero'], pd.to_datetime(str(valuedate)))], \
                            dicno2[(row['numero'], pd.to_datetime(str(valuedate)))], \
                            dico3[(row['numero'], pd.to_datetime(str(valuedate)))], \
                            dicpm10[(row['numero'], pd.to_datetime(str(valuedate)))],\
                            dicco[(row['numero'], pd.to_datetime(str(valuedate)))]))
    
    cleanedList = [((float(x),float(a)),(float(y),float(b)),(float(z),float(c)),(float(w),float(d)),(float(v),float(e))) \
        for ((x,a),(y,b),(z,c),(w,d),(v,e)) in datalist \
            if ((str(x),str(a)),(str(y),str(b)),(str(z),str(c)),(str(w),str(d)),(str(v),str(e))) \
                 != (('NaN','Nan'),('NaN','Nan'),('NaN','Nan'),('NaN','Nan'),('NaN','Nan'))]

    cleanedList2 = [((float(x),float(a)),(float(y),float(b)),(float(z),float(c)),(float(w),float(d)),(float(v),float(e))) \
        for ((x,a),(y,b),(z,c),(w,d), (v,e)) in datalist2 \
            if ((str(x),str(a)),(str(y),str(b)),(str(z),str(c)),(str(w),str(d)),(str(v),str(e))) \
                 != (('NaN','Nan'),('NaN','Nan'),('NaN','Nan'),('NaN','Nan'),('NaN','Nan'))]

    avg = [tuple(sum(j)/len(cleanedList) for j in zip(*i)) for i in zip(*cleanedList)]
    avg2 = [tuple(sum(j)/len(cleanedList2) for j in zip(*i)) for i in zip(*cleanedList2)]
    
    return (max(cleanedList,key=itemgetter(0))[0][0],\
            max(cleanedList,key=itemgetter(1))[1][0],\
            max(cleanedList,key=itemgetter(2))[2][0],\
            max(cleanedList,key=itemgetter(3))[3][0],\
            max(cleanedList,key=itemgetter(4))[4][0],\
            max(cleanedList,key=itemgetter(0))[0][1],\
            max(cleanedList,key=itemgetter(1))[1][1],\
            max(cleanedList,key=itemgetter(2))[2][1],\
            max(cleanedList,key=itemgetter(3))[3][1],\
            max(cleanedList,key=itemgetter(4))[4][1],\
            prevdaytothospi,\
            avg[0][0],\
            avg[1][0],\
            avg[2][0],\
            avg[3][0],\
            avg[4][0],\
            avg2[0][0],\
            avg2[1][0],\
            avg2[2][0],\
            avg2[3][0],\
            avg2[4][0],
            avg[0][1],\
            avg[1][1],\
            avg[2][1],\
            avg[3][1],\
            avg[4][1],\
            avg2[0][1],\
            avg2[1][1],\
            avg2[2][1],\
            avg2[3][1],\
            avg2[4][1])
            
@simple_time_tracker
def compute_Engineered_features_assign_to_df(data):
    data[['1MMaxpm25','1MMaxno2','1MMaxo3','1MMaxpm10','1MMaxco',\
            '1MMaxnormpm25','1MMaxnormno2','1MMaxnormo3','1MMaxnormpm10','1MMaxnormco', 
            'hospiprevday',
            'pm257davg','no27davg','o37davg', 'pm107davg','co7davg',\
            'pm251Mavg','no21Mavg','o31Mavg','pm101Mavg','co1Mavg',\
            "normpm257davg","normno27davg","normo37davg","normpm107davg","normco7davg",\
            "normpm251Mavg","normno21Mavg","normo31Mavg","normpm101Mavg","normco1Mavg"]] \
                = data.apply(compute_Engineered_Features, axis=1).apply(pd.Series)
    print("\n")
    return data

data =  compute_Engineered_features_assign_to_df(data)
print(data)

data.to_csv('../data/train/all_data_merged/fr/Enriched_Covid_history_data.csv', index = False)
    




