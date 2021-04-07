import pandas as pd
from datetime import timedelta
from datetime import datetime

data = pd.read_csv('../data/train/all_data_merged/fr/Enriched_Covid_history_data.csv')

data["time"]=pd.to_datetime(data["time"])

print(data)
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

referencedate = pd.to_datetime("2020-05-14")
def compute_1M_Maxpm25(row):
    datalist = []
    date2 = row['time'] - pd.Timedelta("30 days")


    dates = pd.date_range(start = date2, periods=31).tolist()

    for valuedate in dates:
        if(valuedate < referencedate):
            datalist.append("NaN")
        
        else:
            datalist.append(dicpm25[(row['numero'], pd.to_datetime(str(valuedate)))])
    
    cleanedList = [x for x in datalist if str(x) != 'NaN']
    cleanedList = [float(x) for x in datalist if str(x) != 'NaN']
    return max(cleanedList)

def compute_1M_Maxno2(row):
    datalist = []
    date2 = row['time'] - pd.Timedelta("30 days")


    dates = pd.date_range(start = date2, periods=31).tolist()

    for valuedate in dates:
        if(valuedate < referencedate):
            datalist.append("NaN")
        
        else:
            datalist.append(dicno2[(row['numero'], pd.to_datetime(str(valuedate)))])
    
    cleanedList = [x for x in datalist if str(x) != 'NaN']
    cleanedList = [float(x) for x in datalist if str(x) != 'NaN']
    return max(cleanedList)

def compute_1M_Maxo3(row):
    datalist = []
    date2 = row['time'] - pd.Timedelta("30 days")


    dates = pd.date_range(start = date2, periods=31).tolist()

    for valuedate in dates:
        if(valuedate < referencedate):
            datalist.append("NaN")
        
        else:
            datalist.append(dico3[(row['numero'], pd.to_datetime(str(valuedate)))])
    
    cleanedList = [x for x in datalist if str(x) != 'NaN']
    cleanedList = [float(x) for x in datalist if str(x) != 'NaN']
    return max(cleanedList)

def compute_1M_Maxpm10(row):
    datalist = []
    date2 = row['time'] - pd.Timedelta("30 days")


    dates = pd.date_range(start = date2, periods=31).tolist()

    for valuedate in dates:
        if(valuedate < referencedate):
            datalist.append("NaN")
        
        else:
            datalist.append(dicpm10[(row['numero'], pd.to_datetime(str(valuedate)))])
    
    cleanedList = [x for x in datalist if str(x) != 'NaN']
    cleanedList = [float(x) for x in datalist if str(x) != 'NaN']
    return max(cleanedList)

def compute_1M_Maxco(row):
    datalist = []
    date2 = row['time'] - pd.Timedelta("30 days")


    dates = pd.date_range(start = date2, periods=31).tolist()

    for valuedate in dates:
        if(valuedate < referencedate):
            datalist.append("NaN")
        
        else:
            datalist.append(dicco[(row['numero'], pd.to_datetime(str(valuedate)))])
    
    cleanedList = [x for x in datalist if str(x) != 'NaN']
    cleanedList = [float(x) for x in datalist if str(x) != 'NaN']
    return max(cleanedList)

print("Computing 1MMaxpm25... ")
data['1MMaxpm25'] = data.apply(compute_1M_Maxpm25, axis=1)
print("\n")
print("Computing 1MMaxno2... ")
data['1MMaxno2'] = data.apply(compute_1M_Maxno2, axis=1)
print("\n")
print("Computing 1MMaxno3... ")
data['1MMaxo3'] = data.apply(compute_1M_Maxo3, axis=1)
print("\n")
print("Computing 1MMaxpm10... ")
data['1MMaxpm10'] = data.apply(compute_1M_Maxpm10, axis=1)
print("\n")
print("Computing 1MMaxco... ")
data['1MMaxco'] = data.apply(compute_1M_Maxco, axis=1)
print("\n")

print(data)

data.to_csv('../data/train/all_data_merged/fr/Enriched_Covid_history_data.csv', index = False)

    




