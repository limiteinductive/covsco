import pandas as pd
from datetime import timedelta
from datetime import datetime
from operator import itemgetter
import time
import itertools
import statistics
itertools.imap = lambda *args, **kwargs: list(map(*args, **kwargs))

class Compute_Engineered_Features_for_df:

    def __init__(self):
        self.file_name = None
        self.data = None
        self.pm25tuple = None
        self.no2tuple = None
        self.o3tuple = None
        self.pm10tuple = None
        self.cotuple = None
        self.so2tuple = None
        self.tothospituple = None
        self.newhospituple = None
        self.newreanim = None
        self.dicpm25 = None
        self.dicno2 = None
        self.dico3 = None
        self.dicpm10 = None
        self.dicco = None
        self.dicso2 = None
        self.dictothospi = None
        self.dicnewhospi = None
        self.dicnewreanim = None
        self.pm25EngFeattuple = None
        self.no2EngFeattuple =None
        self.o3EngFeattuple = None
        self.pm10EngFeattuple = None
        self.coEngFeattuple = None
        self.so2EngFeattuple = None
        self.dicpm25EngFeat = None
        self.dicno2EngFeat = None
        self.dico3EngFeat = None
        self.dicpm10EngFeat = None
        self.diccoEngFeat = None
        self.dicso2EngFeat = None
        self.predictiondatadf = None
        self.day1pm25forecast = None
        self.day1no2forecast = None
        self.day1o3forecast = None
        self.day1pm10forecast = None
        self.day1coforecast = None
        self.day1so2forecast = None
        self.day2pm25forecast = None
        self.day2no2forecast = None
        self.day2o3forecast = None
        self.day2pm10forecast = None
        self.day2coforecast = None
        self.day2so2forecast = None
        self.day3pm25forecast = None
        self.day3no2forecast = None
        self.day3o3forecast = None
        self.day3pm10forecast = None
        self.day3coforecast = None
        self.day3so2forecast = None
        self.day4pm25forecast = None
        self.day4no2forecast = None
        self.day4o3forecast = None
        self.day4pm10forecast = None
        self.day4coforecast = None
        self.day4so2forecast = None

    def max_normalize(self, x):
        return (x - x.min()) / (x.max() - x.min())

    def get_data(self):
        self.file_name = '../data/train/all_data_merged/fr/Enriched_Covid_history_data.csv'
        self.data = pd.read_csv(self.file_name)
        return None

    def max_normalize_data(self):
        self.data["normpm25"]=self.max_normalize(self.data["pm25"])
        self.data["normno2"]=self.max_normalize(self.data["no2"])
        self.data["normo3"]=self.max_normalize(self.data["o3"])
        self.data["normpm10"]=self.max_normalize(self.data["pm10"])
        self.data["normco"]=self.max_normalize(self.data["co"])
        self.data["normso2"]=self.max_normalize(self.data["so2"])
        self.data["date"]=pd.to_datetime(self.data["date"])
        return None
    
    def compute_dictionnaries(self):

        self.pm25tuple = (self.data['numero'], self.data["date"], self.data["leadtime_hour"], self.data["pm25"], self.data["normpm25"] )
        self.no2tuple = (self.data['numero'], self.data["date"], self.data["leadtime_hour"], self.data["no2"], self.data["normno2"])
        self.o3tuple = (self.data['numero'], self.data["date"], self.data["leadtime_hour"], self.data["o3"], self.data["normo3"])
        self.pm10tuple = (self.data['numero'], self.data["date"],self.data["leadtime_hour"], self.data["pm10"], self.data["normpm10"])
        self.cotuple = (self.data['numero'], self.data["date"],self.data["leadtime_hour"], self.data["co"], self.data["normco"])
        self.so2tuple = (self.data['numero'], self.data["date"],self.data["leadtime_hour"], self.data["so2"], self.data["normso2"])
        self.tothospituple = (self.data['numero'], self.data["date"],self.data["leadtime_hour"], self.data["hospi"])
        self.newhospituple = (self.data['numero'], self.data["date"],self.data["leadtime_hour"],self.data["newhospi"])
        self.newreanimtuple = (self.data['numero'], self.data["date"],self.data["leadtime_hour"], self.data["newreanim"])

        self.dicpm25 = {(i, j, lh) : (k,l) for (i, j, lh, k, l) in zip(*self.pm25tuple)}
        self.dicno2 = {(i, j, lh) : (k,l)  for (i, j, lh, k, l) in zip(*self.no2tuple)}
        self.dico3 = {(i, j, lh) : (k,l)  for (i, j, lh, k, l) in zip(*self.o3tuple)}
        self.dicpm10 = {(i, j, lh) : (k,l)  for (i, j, lh, k, l) in zip(*self.pm10tuple)}
        self.dicco = {(i, j, lh) : (k,l) for (i, j, lh, k, l) in zip(*self.cotuple)}
        self.dicso2 = {(i, j, lh) : (k,l)  for (i, j, lh, k, l) in zip(*self.so2tuple)}
        self.dictothospi = {(i, j, lh) : k for (i, j, lh, k) in zip(*self.tothospituple)}
        self.dicnewhospi = {(i, j, lh) : k for (i, j, lh, k) in zip(*self.newhospituple)}
        self.dicnewreanim = {(i, j, lh) : k for (i, j, lh, k) in zip(*self.newreanimtuple)}

        return None

    def compute_avg_and_max_dictionnaries(self):

        self.pm25EngFeattuple = (self.data['numero'], self.data["date"], self.data["leadtime_hour"], self.data["pm257davg"], self.data["pm251Mavg"], self.data["1MMaxpm25"] )
        self.no2EngFeattuple = (self.data['numero'], self.data["date"], self.data["leadtime_hour"], self.data["no27davg"], self.data["no21Mavg"], self.data["1MMaxno2"])
        self.o3EngFeattuple = (self.data['numero'], self.data["date"], self.data["leadtime_hour"], self.data["o37davg"], self.data["o31Mavg"], self.data["1MMaxo3"] )
        self.pm10EngFeattuple = (self.data['numero'], self.data["date"],self.data["leadtime_hour"], self.data["pm107davg"], self.data["pm101Mavg"], self.data["1MMaxpm10"])
        self.coEngFeattuple = (self.data['numero'], self.data["date"],self.data["leadtime_hour"], self.data["co7davg"], self.data["co1Mavg"], self.data["1MMaxco"])
        self.so2EngFeattuple = (self.data['numero'], self.data["date"],self.data["leadtime_hour"], self.data["so27davg"], self.data["so21Mavg"],self.data["1MMaxso2"])
        
        
        self.dicpm25EngFeat = {(i, j, lh) : (k,l,m) for (i, j, lh, k, l,m) in zip(*self.pm25EngFeattuple)}
        self.dicno2EngFeat = {(i, j, lh) : (k,l,m)  for (i, j, lh, k, l,m) in zip(*self.no2EngFeattuple)}
        self.dico3EngFeat = {(i, j, lh) : (k,l,m)  for (i, j, lh, k, l,m) in zip(*self.o3EngFeattuple)}
        self.dicpm10EngFeat = {(i, j, lh) : (k,l,m)  for (i, j, lh, k, l,m) in zip(*self.pm10EngFeattuple)}
        self.diccoEngFeat = {(i, j, lh) : (k,l,m) for (i, j, lh, k, l,m) in zip(*self.coEngFeattuple)}
        self.dicso2EngFeat = {(i, j, lh) : (k,l,m)  for (i, j, lh, k, l,m) in zip(*self.so2EngFeattuple)}

        return None
    def compute_input_df_model1(self,row):
        datalist = []
        datalist2 = []
        maxdate = self.data["date"].max()
        day1idx = row["idx"]
        self.day1pm25forecast = self.dicpm25[(row['numero'], maxdate, 24)]
        self.day1no2forecast =self.dicno2[(row['numero'], maxdate, 24)]
        self.day1o3forecast = self.dico3[(row['numero'], maxdate, 24)]
        self.day1pm10forecast = self.dicpm10[(row['numero'], maxdate, 24)]
        self.day1coforecast = self.dicco[(row['numero'], maxdate, 24)]
        self.day1so2forecast = self.dicso2[(row['numero'], maxdate, 24)]
        date = row["date"] + pd.Timedelta("1 days") - pd.Timedelta("30 days")
        date2 = row["date"] + pd.Timedelta("1 days") - pd.Timedelta("6 days")
        dates = pd.date_range(start = date, periods=30).tolist()
        dates2 = pd.date_range(start = date2, periods=6).tolist()
        
        for valuedate in dates:
            datalist.append((self.dicpm25[(row['numero'], pd.to_datetime(str(valuedate)), 0)], \
                            self.dicno2[(row['numero'], pd.to_datetime(str(valuedate)), 0)], \
                            self.dico3[(row['numero'], pd.to_datetime(str(valuedate)), 0)], \
                            self.dicpm10[(row['numero'], pd.to_datetime(str(valuedate)), 0)],\
                            self.dicco[(row['numero'], pd.to_datetime(str(valuedate)), 0)],
                            self.dicso2[(row['numero'], pd.to_datetime(str(valuedate)), 0)]))
        datalist.append((self.day1pm25forecast,self.day1no2forecast,self.day1o3forecast,self.day1pm10forecast,\
            self.day1coforecast,self.day1so2forecast ))
        
        for valuedate in dates2:
            datalist2.append((self.dicpm25[(row['numero'], pd.to_datetime(str(valuedate)), 0)], \
                            self.dicno2[(row['numero'], pd.to_datetime(str(valuedate)), 0)], \
                            self.dico3[(row['numero'], pd.to_datetime(str(valuedate)), 0)], \
                            self.dicpm10[(row['numero'], pd.to_datetime(str(valuedate)), 0)],\
                            self.dicco[(row['numero'], pd.to_datetime(str(valuedate)), 0)],
                            self.dicso2[(row['numero'], pd.to_datetime(str(valuedate)), 0)]))
        datalist2.append((self.day1pm25forecast,self.day1no2forecast,self.day1o3forecast,self.day1pm10forecast,\
            self.day1coforecast,self.day1so2forecast ))
        
        avg = [tuple(sum(j)/len(datalist) for j in zip(*i)) for i in zip(*datalist)]
        avg2 = [tuple(sum(j)/len(datalist2) for j in zip(*i)) for i in zip(*datalist2)]

        day1pm257daverage=avg[0][0]
        day1no27daverage= avg[1][0]
        day1o37daverage=  avg[2][0]
        day1pm107daverage= avg[3][0]
        day1co7daverage= avg[4][0]
        day1so27daverage= avg[5][0]
        day1pm251Maverage= avg2[0][0]
        day1no21Maverage= avg2[1][0]
        day1o31Maverage= avg2[2][0]
        day1pm101Maverage= avg2[3][0]
        day1co1Maverage= avg2[4][0]
        day1so21Maverage= avg2[5][0]
        day1pm251MMax = max(datalist,key=itemgetter(0))[0][0],\
        day1no21MMax =max(datalist,key=itemgetter(1))[1][0],\
        day1o31MMax =max(datalist,key=itemgetter(2))[2][0],\
        day1pm101MMax =max(datalist,key=itemgetter(3))[3][0],\
        day1co1MMax =max(datalist,key=itemgetter(4))[4][0],
        day1so21MMax =max(datalist,key=itemgetter(5))[5][0],\
        day1hospi =row['hospi']
        day1newhospi = row['newhospi']
        day1CovidPosTest = row['CovidPosTest']
        day1Mob1 =row['all_day_bing_tiles_visited_relative_change']
        day1Mob2 = row['all_day_ratio_single_tile_users']
        day1vac1nb = row['vac1nb']
        day1vac2nb = row['vac2nb'] 
        day1ALD14 = row['Insuffisance respiratoire chronique grave (ALD14)']
        day1ALD5 = row['Insuffisance cardiaque grave, troubles du rythme graves, cardiopathies valvulaires graves, cardiopathies congénitales graves (ALD5)']
        day1Smokers = row['Smokers'] 
        day1minority = row["minority"]
        day1pauvrete = row["pauvrete"]
        day1rsa = row["rsa"]
        day1ouvriers = row["ouvriers"]
        day1variant1 = row["Nb_susp_501Y_V1"]
        day1variant2 = row["Nb_susp_501Y_V2_3"]
        return (day1idx,\
                    self.day1pm25forecast,\
                    self.day1no2forecast,\
                    self.day1o3forecast,\
                    self.day1pm10forecast,\
                    self.day1coforecast,\
                    self.day1so2forecast,\
                    day1pm257daverage,\
                    day1no27daverage,\
                    day1o37daverage,\
                    day1pm107daverage,\
                    day1co7daverage,\
                    day1so27daverage,\
                    day1pm251Maverage,\
                    day1no21Maverage,\
                    day1o31Maverage,\
                    day1pm101Maverage,\
                    day1co1Maverage,\
                    day1so21Maverage,\
                    day1pm251MMax,\
                    day1no21MMax,\
                    day1o31MMax,\
                    day1pm101MMax,\
                    day1co1MMax,\
                    day1so21MMax,\
                    day1hospi,\
                    day1newhospi,\
                    day1CovidPosTest,\
                    day1Mob1,\
                    day1Mob2,\
                    day1vac1nb,\
                    day1vac2nb,\
                    day1ALD14,\
                    day1ALD5,\
                    day1Smokers,\
                    day1minority,\
                    day1pauvrete,\
                    day1rsa,\
                    day1ouvriers,\
                    day1variant1,\
                    day1variant2)

    def compute_input_df_model2(self,row):
        datalist =[]
        datalist2 =[]
        maxdate = self.data["date"].max()
        day2idx = row["idx"]
        self.day2pm25forecast = self.dicpm25[(row['numero'], maxdate, 48)]
        self.day2no2forecast = self.dicno2[(row['numero'], maxdate, 48)]
        self.day2o3forecast = self.dico3[(row['numero'], maxdate, 48)]
        self.day2pm10forecast = self.dicpm10[(row['numero'], maxdate, 48)]
        self.day2coforecast = self.dicco[(row['numero'], maxdate, 48)]
        self.day2so2forecast = self.dicso2[(row['numero'], maxdate, 48)]
        date = row["date"] + pd.Timedelta("2 days") - pd.Timedelta("30 days")
        date2 = row["date"] + pd.Timedelta("2 days") - pd.Timedelta("6 days")
        dates = pd.date_range(start = date, periods=29).tolist()
        dates2 = pd.date_range(start = date2, periods=5).tolist()
        
        for valuedate in dates:
            datalist.append((self.dicpm25[(row['numero'], pd.to_datetime(str(valuedate)), 0)], \
                            self.dicno2[(row['numero'], pd.to_datetime(str(valuedate)), 0)], \
                            self.dico3[(row['numero'], pd.to_datetime(str(valuedate)), 0)], \
                            self.dicpm10[(row['numero'], pd.to_datetime(str(valuedate)), 0)],\
                            self.dicco[(row['numero'], pd.to_datetime(str(valuedate)), 0)],
                            self.dicso2[(row['numero'], pd.to_datetime(str(valuedate)), 0)]))

        datalist.append((self.day1pm25forecast,self.day1no2forecast,self.day1o3forecast,self.day1pm10forecast,self.day1coforecast,\
            self.day1so2forecast ))
        datalist.append((self.day2pm25forecast,self.day2no2forecast,self.day2o3forecast,self.day2pm10forecast,self.day2coforecast,\
            self.day2so2forecast ))
        
        for valuedate in dates2:
            datalist2.append((self.dicpm25[(row['numero'], pd.to_datetime(str(valuedate)), 0)], \
                            self.dicno2[(row['numero'], pd.to_datetime(str(valuedate)), 0)], \
                            self.dico3[(row['numero'], pd.to_datetime(str(valuedate)), 0)], \
                            self.dicpm10[(row['numero'], pd.to_datetime(str(valuedate)), 0)],\
                            self.dicco[(row['numero'], pd.to_datetime(str(valuedate)), 0)],
                            self.dicso2[(row['numero'], pd.to_datetime(str(valuedate)), 0)]))
        datalist2.append((self.day1pm25forecast,self.day1no2forecast,self.day1o3forecast,self.day1pm10forecast,\
            self.day1coforecast,self.day1so2forecast ))
        datalist2.append((self.day2pm25forecast,self.day2no2forecast,self.day2o3forecast,self.day2pm10forecast,\
            self.day2coforecast,self.day2so2forecast ))
        
        avg = [tuple(sum(j)/len(datalist) for j in zip(*i)) for i in zip(*datalist)]
        avg2 = [tuple(sum(j)/len(datalist2) for j in zip(*i)) for i in zip(*datalist2)]
        
        day2pm257daverage=avg[0][0]
        day2no27daverage= avg[1][0]
        day2o37daverage=  avg[2][0]
        day2pm107daverage= avg[3][0]
        day2co7daverage= avg[4][0]
        day2so27daverage= avg[5][0]
        day2pm251Maverage= avg2[0][0]
        day2no21Maverage= avg2[1][0]
        day2o31Maverage= avg2[2][0]
        day2pm101Maverage= avg2[3][0]
        day2co1Maverage= avg2[4][0]
        day2so21Maverage= avg2[5][0]
        day2pm251MMax = max(datalist,key=itemgetter(0))[0][0],\
        day2no21MMax =max(datalist,key=itemgetter(1))[1][0],\
        day2o31MMax =max(datalist,key=itemgetter(2))[2][0],\
        day2pm101MMax =max(datalist,key=itemgetter(3))[3][0],\
        day2co1MMax =max(datalist,key=itemgetter(4))[4][0],
        day2so21MMax =max(datalist,key=itemgetter(5))[5][0],\
        day2hospi =row['hospi']
        day2newhospi = row['newhospi']
        day2CovidPosTest = row['CovidPosTest']
        day2Mob1 =row['all_day_bing_tiles_visited_relative_change']
        day2Mob2 = row['all_day_ratio_single_tile_users']
        day2vac1nb = row['vac1nb']
        day2vac2nb = row['vac2nb'] 
        day2ALD14 = row['Insuffisance respiratoire chronique grave (ALD14)']
        day2ALD5 = row['Insuffisance cardiaque grave, troubles du rythme graves, cardiopathies valvulaires graves, cardiopathies congénitales graves (ALD5)']
        day2Smokers = row['Smokers'] 
        day2minority = row["minority"]
        day2pauvrete = row["pauvrete"]
        day2rsa = row["rsa"]
        day2ouvriers = row["ouvriers"]
        day2variant1 = row["Nb_susp_501Y_V1"]
        day2variant2 = row["Nb_susp_501Y_V2_3"]
        return (day2idx,\
                    self.day2pm25forecast,\
                    self.day2no2forecast,\
                    self.day2o3forecast,\
                    self.day2pm10forecast,\
                    self.day2coforecast,\
                    self.day2so2forecast,\
                    day2pm257daverage,\
                    day2no27daverage,\
                    day2o37daverage,\
                    day2pm107daverage,\
                    day2co7daverage,\
                    day2so27daverage,\
                    day2pm251Maverage,\
                    day2no21Maverage,\
                    day2o31Maverage,\
                    day2pm101Maverage,\
                    day2co1Maverage,\
                    day2so21Maverage,\
                    day2pm251MMax,\
                    day2no21MMax,\
                    day2o31MMax,\
                    day2pm101MMax,\
                    day2co1MMax,\
                    day2so21MMax,\
                    day2hospi,\
                    day2newhospi,\
                    day2CovidPosTest,\
                    day2Mob1,\
                    day2Mob2,\
                    day2vac1nb,\
                    day2vac2nb,\
                    day2ALD14,\
                    day2ALD5,\
                    day2Smokers,\
                    day2minority,\
                    day2pauvrete,\
                    day2rsa,\
                    day2ouvriers,\
                    day2variant1,\
                    day2variant2)

    def compute_input_df_model3(self,row):
        datalist =[]
        datalist2 =[]
        maxdate = self.data["date"].max()
        day3idx = row["idx"]
        self.day3pm25forecast = self.dicpm25[(row['numero'], maxdate, 48)]
        self.day3no2forecast = self.dicno2[(row['numero'], maxdate, 48)]
        self.day3o3forecast = self.dico3[(row['numero'], maxdate, 48)]
        self.day3pm10forecast = self.dicpm10[(row['numero'], maxdate, 48)]
        self.day3coforecast = self.dicco[(row['numero'], maxdate, 48)]
        self.day3so2forecast = self.dicso2[(row['numero'], maxdate, 48)]
        date = row["date"] + pd.Timedelta("3 days") - pd.Timedelta("30 days")
        date2 = row["date"] + pd.Timedelta("3 days") - pd.Timedelta("6 days")
        dates = pd.date_range(start = date, periods=28).tolist()
        dates2 = pd.date_range(start = date2, periods=4).tolist()
        
        for valuedate in dates:
            datalist.append((self.dicpm25[(row['numero'], pd.to_datetime(str(valuedate)), 0)], \
                            self.dicno2[(row['numero'], pd.to_datetime(str(valuedate)), 0)], \
                            self.dico3[(row['numero'], pd.to_datetime(str(valuedate)), 0)], \
                            self.dicpm10[(row['numero'], pd.to_datetime(str(valuedate)), 0)],\
                            self.dicco[(row['numero'], pd.to_datetime(str(valuedate)), 0)],
                            self.dicso2[(row['numero'], pd.to_datetime(str(valuedate)), 0)]))

        datalist.append((self.day1pm25forecast,self.day1no2forecast,self.day1o3forecast,self.day1pm10forecast,self.day1coforecast,\
            self.day1so2forecast ))
        datalist.append((self.day2pm25forecast,self.day2no2forecast,self.day2o3forecast,self.day2pm10forecast,self.day2coforecast,\
            self.day2so2forecast ))
        datalist.append((self.day3pm25forecast,self.day3no2forecast,self.day3o3forecast,self.day3pm10forecast,self.day3coforecast,\
            self.day3so2forecast ))
        
        for valuedate in dates2:
            datalist2.append((self.dicpm25[(row['numero'], pd.to_datetime(str(valuedate)), 0)], \
                            self.dicno2[(row['numero'], pd.to_datetime(str(valuedate)), 0)], \
                            self.dico3[(row['numero'], pd.to_datetime(str(valuedate)), 0)], \
                            self.dicpm10[(row['numero'], pd.to_datetime(str(valuedate)), 0)],\
                            self.dicco[(row['numero'], pd.to_datetime(str(valuedate)), 0)],
                            self.dicso2[(row['numero'], pd.to_datetime(str(valuedate)), 0)]))

        datalist2.append((self.day1pm25forecast,self.day1no2forecast,self.day1o3forecast,self.day1pm10forecast,\
            self.day1coforecast,self.day1so2forecast ))
        datalist2.append((self.day2pm25forecast,self.day2no2forecast,self.day2o3forecast,self.day2pm10forecast,\
            self.day2coforecast,self.day2so2forecast ))
        datalist2.append((self.day3pm25forecast,self.day3no2forecast,self.day3o3forecast,self.day3pm10forecast,\
            self.day3coforecast,self.day3so2forecast ))
        
        avg = [tuple(sum(j)/len(datalist) for j in zip(*i)) for i in zip(*datalist)]
        avg2 = [tuple(sum(j)/len(datalist2) for j in zip(*i)) for i in zip(*datalist2)]
        
        day3pm257daverage=avg[0][0]
        day3no27daverage= avg[1][0]
        day3o37daverage=  avg[2][0]
        day3pm107daverage= avg[3][0]
        day3co7daverage= avg[4][0]
        day3so27daverage= avg[5][0]
        day3pm251Maverage= avg2[0][0]
        day3no21Maverage= avg2[1][0]
        day3o31Maverage= avg2[2][0]
        day3pm101Maverage= avg2[3][0]
        day3co1Maverage= avg2[4][0]
        day3so21Maverage= avg2[5][0]
        day3pm251MMax = max(datalist,key=itemgetter(0))[0][0],\
        day3no21MMax =max(datalist,key=itemgetter(1))[1][0],\
        day3o31MMax =max(datalist,key=itemgetter(2))[2][0],\
        day3pm101MMax =max(datalist,key=itemgetter(3))[3][0],\
        day3co1MMax =max(datalist,key=itemgetter(4))[4][0],
        day3so21MMax =max(datalist,key=itemgetter(5))[5][0],\
        day3hospi =row['hospi']
        day3newhospi = row['newhospi']
        day3CovidPosTest = row['CovidPosTest']
        day3Mob1 =row['all_day_bing_tiles_visited_relative_change']
        day3Mob2 = row['all_day_ratio_single_tile_users']
        day3vac1nb = row['vac1nb']
        day3vac2nb = row['vac2nb'] 
        day3ALD14 = row['Insuffisance respiratoire chronique grave (ALD14)']
        day3ALD5 = row['Insuffisance cardiaque grave, troubles du rythme graves, cardiopathies valvulaires graves, cardiopathies congénitales graves (ALD5)']
        day3Smokers = row['Smokers'] 
        day3minority = row["minority"]
        day3pauvrete = row["pauvrete"]
        day3rsa = row["rsa"]
        day3ouvriers = row["ouvriers"]
        day3variant1 = row["Nb_susp_501Y_V1"]
        day3variant2 = row["Nb_susp_501Y_V2_3"]
        return (day3idx,\
                    self.day3pm25forecast,\
                    self.day3no2forecast,\
                    self.day3o3forecast,\
                    self.day3pm10forecast,\
                    self.day3coforecast,\
                    self.day3so2forecast,\
                    day3pm257daverage,\
                    day3no27daverage,\
                    day3o37daverage,\
                    day3pm107daverage,\
                    day3co7daverage,\
                    day3so27daverage,\
                    day3pm251Maverage,\
                    day3no21Maverage,\
                    day3o31Maverage,\
                    day3pm101Maverage,\
                    day3co1Maverage,\
                    day3so21Maverage,\
                    day3pm251MMax,\
                    day3no21MMax,\
                    day3o31MMax,\
                    day3pm101MMax,\
                    day3co1MMax,\
                    day3so21MMax,\
                    day3hospi,\
                    day3newhospi,\
                    day3CovidPosTest,\
                    day3Mob1,\
                    day3Mob2,\
                    day3vac1nb,\
                    day3vac2nb,\
                    day3ALD14,\
                    day3ALD5,\
                    day3Smokers,\
                    day3minority,\
                    day3pauvrete,\
                    day3rsa,\
                    day3ouvriers,\
                    day3variant1,\
                    day3variant2)

    def compute_input_df_model4(self,row):
        datalist =[]
        datalist2 =[]
        maxdate = self.data["date"].max()
        day4idx = row["idx"]
        self.day4pm25forecast = self.dicpm25[(row['numero'], maxdate, 48)]
        self.day4no2forecast = self.dicno2[(row['numero'], maxdate, 48)]
        self.day4o3forecast = self.dico3[(row['numero'], maxdate, 48)]
        self.day4pm10forecast = self.dicpm10[(row['numero'], maxdate, 48)]
        self.day4coforecast = self.dicco[(row['numero'], maxdate, 48)]
        self.day4so2forecast = self.dicso2[(row['numero'], maxdate, 48)]
        date = row["date"] + pd.Timedelta("3 days") - pd.Timedelta("30 days")
        date2 = row["date"] + pd.Timedelta("3 days") - pd.Timedelta("6 days")
        dates = pd.date_range(start = date, periods=28).tolist()
        dates2 = pd.date_range(start = date2, periods=4).tolist()
        
        for valuedate in dates:
            datalist.append((self.dicpm25[(row['numero'], pd.to_datetime(str(valuedate)), 0)], \
                            self.dicno2[(row['numero'], pd.to_datetime(str(valuedate)), 0)], \
                            self.dico3[(row['numero'], pd.to_datetime(str(valuedate)), 0)], \
                            self.dicpm10[(row['numero'], pd.to_datetime(str(valuedate)), 0)],\
                            self.dicco[(row['numero'], pd.to_datetime(str(valuedate)), 0)],
                            self.dicso2[(row['numero'], pd.to_datetime(str(valuedate)), 0)]))

        datalist.append((self.day1pm25forecast,self.day1no2forecast,self.day1o3forecast,self.day1pm10forecast,self.day1coforecast,\
            self.day1so2forecast ))
        datalist.append((self.day2pm25forecast,self.day2no2forecast,self.day2o3forecast,self.day2pm10forecast,self.day2coforecast,\
            self.day2so2forecast ))
        datalist.append((self.day3pm25forecast,self.day3no2forecast,self.day3o3forecast,self.day3pm10forecast,self.day3coforecast,\
            self.day3so2forecast ))
        datalist.append((self.day4pm25forecast,self.day4no2forecast,self.day4o3forecast,self.day4pm10forecast,self.day4coforecast,\
            self.day4so2forecast ))  

        for valuedate in dates2:
            datalist2.append((self.dicpm25[(row['numero'], pd.to_datetime(str(valuedate)), 0)], \
                            self.dicno2[(row['numero'], pd.to_datetime(str(valuedate)), 0)], \
                            self.dico3[(row['numero'], pd.to_datetime(str(valuedate)), 0)], \
                            self.dicpm10[(row['numero'], pd.to_datetime(str(valuedate)), 0)],\
                            self.dicco[(row['numero'], pd.to_datetime(str(valuedate)), 0)],
                            self.dicso2[(row['numero'], pd.to_datetime(str(valuedate)), 0)]))

        datalist2.append((self.day1pm25forecast,self.day1no2forecast,self.day1o3forecast,self.day1pm10forecast,\
            self.day1coforecast,self.day1so2forecast ))
        datalist2.append((self.day2pm25forecast,self.day2no2forecast,self.day2o3forecast,self.day2pm10forecast,\
            self.day2coforecast,self.day2so2forecast ))
        datalist2.append((self.day3pm25forecast,self.day3no2forecast,self.day3o3forecast,self.day3pm10forecast,\
            self.day3coforecast,self.day3so2forecast ))
        datalist2.append((self.day4pm25forecast,self.day4no2forecast,self.day4o3forecast,self.day4pm10forecast,\
            self.day4coforecast,self.day4so2forecast ))  

        avg = [tuple(sum(j)/len(datalist) for j in zip(*i)) for i in zip(*datalist)]
        avg2 = [tuple(sum(j)/len(datalist2) for j in zip(*i)) for i in zip(*datalist2)]
        
        day4pm257daverage=avg[0][0]
        day4no27daverage= avg[1][0]
        day4o37daverage=  avg[2][0]
        day4pm107daverage= avg[3][0]
        day4co7daverage= avg[4][0]
        day4so27daverage= avg[5][0]
        day4pm251Maverage= avg2[0][0]
        day4no21Maverage= avg2[1][0]
        day4o31Maverage= avg2[2][0]
        day4pm101Maverage= avg2[3][0]
        day4co1Maverage= avg2[4][0]
        day4so21Maverage= avg2[5][0]
        day4pm251MMax = max(datalist,key=itemgetter(0))[0][0],\
        day4no21MMax =max(datalist,key=itemgetter(1))[1][0],\
        day4o31MMax =max(datalist,key=itemgetter(2))[2][0],\
        day4pm101MMax =max(datalist,key=itemgetter(3))[3][0],\
        day4co1MMax =max(datalist,key=itemgetter(4))[4][0],
        day4so21MMax =max(datalist,key=itemgetter(5))[5][0],\
        day4hospi =row['hospi']
        day4newhospi = row['newhospi']
        day4CovidPosTest = row['CovidPosTest']
        day4Mob1 =row['all_day_bing_tiles_visited_relative_change']
        day4Mob2 = row['all_day_ratio_single_tile_users']
        day4vac1nb = row['vac1nb']
        day4vac2nb = row['vac2nb'] 
        day4ALD14 = row['Insuffisance respiratoire chronique grave (ALD14)']
        day4ALD5 = row['Insuffisance cardiaque grave, troubles du rythme graves, cardiopathies valvulaires graves, cardiopathies congénitales graves (ALD5)']
        day4Smokers = row['Smokers'] 
        day4minority = row["minority"]
        day4pauvrete = row["pauvrete"]
        day4rsa = row["rsa"]
        day4ouvriers = row["ouvriers"]
        day4variant1 = row["Nb_susp_501Y_V1"]
        day4variant2 = row["Nb_susp_501Y_V2_3"]
        return (day4idx,\
                    self.day4pm25forecast,\
                    self.day4no2forecast,\
                    self.day4o3forecast,\
                    self.day4pm10forecast,\
                    self.day4coforecast,\
                    self.day4so2forecast,\
                    day4pm257daverage,\
                    day4no27daverage,\
                    day4o37daverage,\
                    day4pm107daverage,\
                    day4co7daverage,\
                    day4so27daverage,\
                    day4pm251Maverage,\
                    day4no21Maverage,\
                    day4o31Maverage,\
                    day4pm101Maverage,\
                    day4co1Maverage,\
                    day4so21Maverage,\
                    day4pm251MMax,\
                    day4no21MMax,\
                    day4o31MMax,\
                    day4pm101MMax,\
                    day4co1MMax,\
                    day4so21MMax,\
                    day4hospi,\
                    day4newhospi,\
                    day4CovidPosTest,\
                    day4Mob1,\
                    day4Mob2,\
                    day4vac1nb,\
                    day4vac2nb,\
                    day4ALD14,\
                    day4ALD5,\
                    day4Smokers,\
                    day4minority,\
                    day4pauvrete,\
                    day4rsa,\
                    day4ouvriers,\
                    day4variant1,\
                    day4variant2)

    def compute_forecast_data_engineered_features_day0(self,row):
        datemax = self.data["date"].max()   
        modelday0features = ['idx', 'pm25', 'no2','o3','pm10','co','so2',\
            'pm257davg','no27davg','o37davg','co7davg', 'pm107davg','so27davg',\
                'pm251Mavg','no21Mavg','o31Mavg','pm101Mavg','co1Mavg','so21Mavg',\
                    '1MMaxpm25','1MMaxpm10','1MMaxo3','1MMaxno2','1MMaxco','1MMaxso2',\
                        'hospi','newhospi','CovidPosTest',\
                            'all_day_bing_tiles_visited_relative_change','all_day_ratio_single_tile_users',\
                                'vac1nb', 'vac2nb',\
                                    'Insuffisance respiratoire chronique grave (ALD14)', \
                                        'Insuffisance cardiaque grave, troubles du rythme graves, cardiopathies valvulaires graves, cardiopathies congénitales graves (ALD5)',\
                                            'Smokers',\
                                                "minority","pauvrete","rsa","ouvriers",\
                                                    "Nb_susp_501Y_V1","Nb_susp_501Y_V2_3"\
                                                        ]

        day0df = self.data[(self.data["date"]==datemax) & (self.data["leadtime_hour"]== 0)][modelday0features]
        day1df = self.data[(self.data["date"]==datemax) & (self.data["leadtime_hour"]== 24)][modelday0features]
        day1df = day1df[modelday0features].apply(self.compute_input_df_model1, axis=1).apply(pd.Series)
        day2df = self.data[(self.data["date"]==datemax) & (self.data["leadtime_hour"]== 48)][modelday0features]
        day2df = day2df[modelday0features].apply(self.compute_input_df_model2, axis=1).apply(pd.Series)
        day3df = self.data[(self.data["date"]==datemax) & (self.data["leadtime_hour"]== 72)][modelday0features]
        day3df = day3df[modelday0features].apply(self.compute_input_df_model3, axis=1).apply(pd.Series)
        day4df = self.data[(self.data["date"]==datemax) & (self.data["leadtime_hour"]== 96)][modelday0features]
        day4df = day4df[modelday0features].apply(self.compute_input_df_model4, axis=1).apply(pd.Series)
    
        return (day0df,day1df,day2df,day3df,day4df)

    def compute_forecast_data_for_training_models(self,row):
        print("Computing forecast data for training models...")
        self.compute_avg_and_max_dictionnaries()
        dateiminusone = row["date"] - pd.Timedelta("1 days") 
        dateiminustwo = row["date"] - pd.Timedelta("2 days") 
        dateiminusthree = row["date"] - pd.Timedelta("3 days") 
        dateiminusfour = row["date"] - pd.Timedelta("4 days") 

        if (dateiminusfour < self.data["date"].min()):
            dateiminusonepm25dayiforecast= "NaN"
            dateiminusoneno2dayiforecast= "NaN"
            dateiminusoneo3dayiforecast= "NaN"
            dateiminusonepm10dayiforecast= "NaN"
            dateiminusonecodayiforecast= "NaN"
            dateiminusoneso2dayiforecast= "NaN"
            dateiminustwopm25dayiforecast= "NaN"
            dateiminustwono2dayiforecast= "NaN"
            dateiminustwoo3dayiforecast= "NaN"
            dateiminustwopm10dayiforecast= "NaN"
            dateiminustwocodayiforecast= "NaN"
            dateiminustwoso2dayiforecast= "NaN"
            dateiminusthreepm25dayiforecast= "NaN"
            dateiminusthreeno2dayiforecast= "NaN"
            dateiminusthreeo3dayiforecast= "NaN"
            dateiminusthreepm10dayiforecast= "NaN"
            dateiminusthreecodayiforecast= "NaN"
            dateiminusthreeso2dayiforecast= "NaN"
            dateiminusfourpm25dayiforecast= "NaN"
            dateiminusfourno2dayiforecast= "NaN"
            dateiminusfouro3dayiforecast= "NaN"
            dateiminusfourpm10dayiforecast= "NaN"
            dateiminusfourcodayiforecast= "NaN"
            dateiminusfourso2dayiforecast = "NaN"
            dateiminusonepm25dayiforecast7davg = "NaN"
            dateiminusoneno2dayiforecast7davg= "NaN"
            dateiminusoneo3dayiforecast7davg= "NaN"
            dateiminusonepm10dayiforecast7davg= "NaN"
            dateiminusonecodayiforecast7davg= "NaN"
            dateiminusoneso2dayiforecast7davg= "NaN"
            dateiminusonepm25dayiforecast1Mavg= "NaN"
            dateiminusoneno2dayiforecast1Mavg= "NaN"
            dateiminusoneo3dayiforecast1Mavg= "NaN"
            dateiminusonepm10dayiforecast1Mavg= "NaN"
            dateiminusonecodayiforecast1Mavg= "NaN"
            dateiminusoneso2dayiforecast1Mavg= "NaN"
            dateiminusonepm25dayiforecast1MMax= "NaN"
            dateiminusoneno2dayiforecast1MMax= "NaN"
            dateiminusoneo3dayiforecast1MMax= "NaN"
            dateiminusonepm10dayiforecast1MMax= "NaN"
            dateiminusonecodayiforecast1MMax= "NaN"
            dateiminusoneso2dayiforecast1MMax= "NaN"
            dateiminustwopm25dayiforecast7davg= "NaN"
            dateiminustwono2dayiforecast7davg= "NaN"
            dateiminustwoo3dayiforecast7davg= "NaN"
            dateiminustwopm10dayiforecast7davg= "NaN"
            dateiminustwocodayiforecast7davg= "NaN"
            dateiminustwoso2dayiforecast7davg= "NaN"
            dateiminustwopm25dayiforecast1Mavg= "NaN"
            dateiminustwono2dayiforecast1Mavg= "NaN"
            dateiminustwoo3dayiforecast1Mavg= "NaN"
            dateiminustwopm10dayiforecast1Mavg= "NaN"
            dateiminustwocodayiforecast1Mavg= "NaN"
            dateiminustwoso2dayiforecast1Mavg= "NaN"
            dateiminustwopm25dayiforecast1MMax= "NaN"
            dateiminustwono2dayiforecast1MMax= "NaN"
            dateiminustwoo3dayiforecast1MMax= "NaN"
            dateiminustwopm10dayiforecast1MMax= "NaN"
            dateiminustwocodayiforecast1MMax= "NaN"
            dateiminustwoso2dayiforecast1MMax= "NaN"
            dateiminusthreepm25dayiforecast7davg= "NaN"
            dateiminusthreeno2dayiforecast7davg= "NaN"
            dateiminusthreeo3dayiforecast7davg= "NaN"
            dateiminusthreepm10dayiforecast7davg= "NaN"
            dateiminusthreecodayiforecast7davg= "NaN"
            dateiminusthreeso2dayiforecast7davg= "NaN"
            dateiminusthreepm25dayiforecast1Mavg= "NaN"
            dateiminusthreeno2dayiforecast1Mavg= "NaN"
            dateiminusthreeo3dayiforecast1Mavg= "NaN"
            dateiminusthreepm10dayiforecast1Mavg= "NaN"
            dateiminusthreecodayiforecast1Mavg= "NaN"
            dateiminusthreeso2dayiforecast1Mavg= "NaN"
            dateiminusthreepm25dayiforecast1MMax= "NaN"
            dateiminusthreeno2dayiforecast1MMax= "NaN"
            dateiminusthreeo3dayiforecast1MMax= "NaN"
            dateiminusthreepm10dayiforecast1MMax= "NaN"
            dateiminusthreecodayiforecast1MMax= "NaN"
            dateiminusthreeso2dayiforecast1MMax= "NaN"
            dateiminusfourpm25dayiforecast7davg= "NaN"
            dateiminusfourno2dayiforecast7davg= "NaN"
            dateiminusfouro3dayiforecast7davg= "NaN"
            dateiminusfourpm10dayiforecast7davg= "NaN"
            dateiminusfourcodayiforecast7davg= "NaN"
            dateiminusfourso2dayiforecast7davg= "NaN"
            dateiminusfourpm25dayiforecast1Mavg= "NaN"
            dateiminusfourno2dayiforecast1Mavg= "NaN"
            dateiminusfouro3dayiforecast1Mavg= "NaN"
            dateiminusfourpm10dayiforecast1Mavg= "NaN"
            dateiminusfourcodayiforecast1Mavg= "NaN"
            dateiminusfourso2dayiforecast1Mavg= "NaN"
            dateiminusfourpm25dayiforecast1MMax= "NaN"
            dateiminusfourno2dayiforecast1MMax= "NaN"
            dateiminusfouro3dayiforecast1MMax= "NaN"
            dateiminusfourpm10dayiforecast1MMax= "NaN"
            dateiminusfourcodayiforecast1MMax= "NaN"
            dateiminusfourso2dayiforecast1MMax= "NaN"
        
        else:

            dateiminusonepm25dayiforecast= self.dicpm25[(row['numero'], pd.to_datetime(str(dateiminusone)), 24)][0]
            dateiminusoneno2dayiforecast= self.dicno2[(row['numero'], pd.to_datetime(str(dateiminusone)), 24)][0]
            dateiminusoneo3dayiforecast= self.dico3[(row['numero'], pd.to_datetime(str(dateiminusone)), 24)][0]
            dateiminusonepm10dayiforecast= self.dicpm10[(row['numero'], pd.to_datetime(str(dateiminusone)), 24)][0]
            dateiminusonecodayiforecast= self.dicco[(row['numero'], pd.to_datetime(str(dateiminusone)), 24)][0]
            dateiminusoneso2dayiforecast= self.dicso2[(row['numero'], pd.to_datetime(str(dateiminusone)), 24)][0]

            dateiminusonepm25dayiforecast7davg= self.dicpm25EngFeat[(row['numero'], pd.to_datetime(str(dateiminusone)), 0)][0]
            dateiminusoneno2dayiforecast7davg= self.dicno2EngFeat[(row['numero'], pd.to_datetime(str(dateiminusone)), 0)][0]
            dateiminusoneo3dayiforecast7davg= self.dico3EngFeat[(row['numero'], pd.to_datetime(str(dateiminusone)), 0)][0]
            dateiminusonepm10dayiforecast7davg= self.dicpm10EngFeat[(row['numero'], pd.to_datetime(str(dateiminusone)), 0)][0]
            dateiminusonecodayiforecast7davg= self.diccoEngFeat[(row['numero'], pd.to_datetime(str(dateiminusone)), 0)][0]
            dateiminusoneso2dayiforecast7davg= self.dicso2EngFeat[(row['numero'], pd.to_datetime(str(dateiminusone)), 0)][0]

            dateiminusonepm25dayiforecast1Mavg= self.dicpm25EngFeat[(row['numero'], pd.to_datetime(str(dateiminusone)),0)][1]
            dateiminusoneno2dayiforecast1Mavg= self.dicno2EngFeat[(row['numero'], pd.to_datetime(str(dateiminusone)), 0)][1]
            dateiminusoneo3dayiforecast1Mavg= self.dico3EngFeat[(row['numero'], pd.to_datetime(str(dateiminusone)), 0)][1]
            dateiminusonepm10dayiforecast1Mavg= self.dicpm10EngFeat[(row['numero'], pd.to_datetime(str(dateiminusone)),0)][1]
            dateiminusonecodayiforecast1Mavg= self.diccoEngFeat[(row['numero'], pd.to_datetime(str(dateiminusone)), 0)][1]
            dateiminusoneso2dayiforecast1Mavg= self.dicso2EngFeat[(row['numero'], pd.to_datetime(str(dateiminusone)), 0)][1]

            dateiminusonepm25dayiforecast1MMax= self.dicpm25EngFeat[(row['numero'], pd.to_datetime(str(dateiminusone)), 0)][2]
            dateiminusoneno2dayiforecast1MMax= self.dicno2EngFeat[(row['numero'], pd.to_datetime(str(dateiminusone)), 0)][2]
            dateiminusoneo3dayiforecast1MMax= self.dico3EngFeat[(row['numero'], pd.to_datetime(str(dateiminusone)), 0)][2]
            dateiminusonepm10dayiforecast1MMax= self.dicpm10EngFeat[(row['numero'], pd.to_datetime(str(dateiminusone)), 0)][2]
            dateiminusonecodayiforecast1MMax= self.diccoEngFeat[(row['numero'], pd.to_datetime(str(dateiminusone)), 0)][2]
            dateiminusoneso2dayiforecast1MMax= self.dicso2EngFeat[(row['numero'], pd.to_datetime(str(dateiminusone)), 0)][2]

            dateiminustwopm25dayiforecast= self.dicpm25[(row['numero'], pd.to_datetime(str(dateiminustwo)), 48)][0]
            dateiminustwono2dayiforecast= self.dicno2[(row['numero'], pd.to_datetime(str(dateiminustwo)),48)][0]
            dateiminustwoo3dayiforecast= self.dico3[(row['numero'], pd.to_datetime(str(dateiminustwo)), 48)][0]
            dateiminustwopm10dayiforecast= self.dicpm10[(row['numero'], pd.to_datetime(str(dateiminustwo)), 48)][0]
            dateiminustwocodayiforecast= self.dicco[(row['numero'], pd.to_datetime(str(dateiminustwo)), 48)][0]
            dateiminustwoso2dayiforecast= self.dicso2[(row['numero'], pd.to_datetime(str(dateiminustwo)), 48)][0]

            dateiminustwopm25dayiforecast7davg= self.dicpm25EngFeat[(row['numero'], pd.to_datetime(str(dateiminustwo)), 0)][0]
            dateiminustwono2dayiforecast7davg= self.dicno2EngFeat[(row['numero'], pd.to_datetime(str(dateiminustwo)), 0)][0]
            dateiminustwoo3dayiforecast7davg= self.dico3EngFeat[(row['numero'], pd.to_datetime(str(dateiminustwo)), 0)][0]
            dateiminustwopm10dayiforecast7davg= self.dicpm10EngFeat[(row['numero'], pd.to_datetime(str(dateiminustwo)), 0)][0]
            dateiminustwocodayiforecast7davg= self.diccoEngFeat[(row['numero'], pd.to_datetime(str(dateiminustwo)), 0)][0]
            dateiminustwoso2dayiforecast7davg= self.dicso2EngFeat[(row['numero'], pd.to_datetime(str(dateiminustwo)), 0)][0]

            dateiminustwopm25dayiforecast1Mavg= self.dicpm25EngFeat[(row['numero'], pd.to_datetime(str(dateiminustwo)),0)][1]
            dateiminustwono2dayiforecast1Mavg= self.dicno2EngFeat[(row['numero'], pd.to_datetime(str(dateiminustwo)), 0)][1]
            dateiminustwoo3dayiforecast1Mavg= self.dico3EngFeat[(row['numero'], pd.to_datetime(str(dateiminustwo)), 0)][1]
            dateiminustwopm10dayiforecast1Mavg= self.dicpm10EngFeat[(row['numero'], pd.to_datetime(str(dateiminustwo)),0)][1]
            dateiminustwocodayiforecast1Mavg= self.diccoEngFeat[(row['numero'], pd.to_datetime(str(dateiminustwo)), 0)][1]
            dateiminustwoso2dayiforecast1Mavg= self.dicso2EngFeat[(row['numero'], pd.to_datetime(str(dateiminustwo)), 0)][1]

            dateiminustwopm25dayiforecast1MMax= self.dicpm25EngFeat[(row['numero'], pd.to_datetime(str(dateiminustwo)), 0)][2]
            dateiminustwono2dayiforecast1MMax= self.dicno2EngFeat[(row['numero'], pd.to_datetime(str(dateiminustwo)), 0)][2]
            dateiminustwoo3dayiforecast1MMax= self.dico3EngFeat[(row['numero'], pd.to_datetime(str(dateiminustwo)), 0)][2]
            dateiminustwopm10dayiforecast1MMax= self.dicpm10EngFeat[(row['numero'], pd.to_datetime(str(dateiminustwo)), 0)][2]
            dateiminustwocodayiforecast1MMax= self.diccoEngFeat[(row['numero'], pd.to_datetime(str(dateiminustwo)), 0)][2]
            dateiminustwoso2dayiforecast1MMax= self.dicso2EngFeat[(row['numero'], pd.to_datetime(str(dateiminustwo)), 0)][2]

            dateiminusthreepm25dayiforecast= self.dicpm25[(row['numero'], pd.to_datetime(str(dateiminusthree)), 72)][0]
            dateiminusthreeno2dayiforecast= self.dicno2[(row['numero'], pd.to_datetime(str(dateiminusthree)),72)][0]
            dateiminusthreeo3dayiforecast= self.dico3[(row['numero'], pd.to_datetime(str(dateiminusthree)), 72)][0]
            dateiminusthreepm10dayiforecast= self.dicpm10[(row['numero'], pd.to_datetime(str(dateiminusthree)), 72)][0]
            dateiminusthreecodayiforecast= self.dicco[(row['numero'], pd.to_datetime(str(dateiminusthree)), 72)][0]
            dateiminusthreeso2dayiforecast= self.dicso2[(row['numero'], pd.to_datetime(str(dateiminusthree)), 72)][0]

            dateiminusthreepm25dayiforecast7davg= self.dicpm25EngFeat[(row['numero'], pd.to_datetime(str(dateiminusthree)), 0)][0]
            dateiminusthreeno2dayiforecast7davg= self.dicno2EngFeat[(row['numero'], pd.to_datetime(str(dateiminusthree)), 0)][0]
            dateiminusthreeo3dayiforecast7davg= self.dico3EngFeat[(row['numero'], pd.to_datetime(str(dateiminusthree)), 0)][0]
            dateiminusthreepm10dayiforecast7davg= self.dicpm10EngFeat[(row['numero'], pd.to_datetime(str(dateiminusthree)), 0)][0]
            dateiminusthreecodayiforecast7davg= self.diccoEngFeat[(row['numero'], pd.to_datetime(str(dateiminusthree)), 0)][0]
            dateiminusthreeso2dayiforecast7davg= self.dicso2EngFeat[(row['numero'], pd.to_datetime(str(dateiminusthree)), 0)][0]

            dateiminusthreepm25dayiforecast1Mavg= self.dicpm25EngFeat[(row['numero'], pd.to_datetime(str(dateiminusthree)),0)][1]
            dateiminusthreeno2dayiforecast1Mavg= self.dicno2EngFeat[(row['numero'], pd.to_datetime(str(dateiminusthree)), 0)][1]
            dateiminusthreeo3dayiforecast1Mavg= self.dico3EngFeat[(row['numero'], pd.to_datetime(str(dateiminusthree)), 0)][1]
            dateiminusthreepm10dayiforecast1Mavg= self.dicpm10EngFeat[(row['numero'], pd.to_datetime(str(dateiminusthree)),0)][1]
            dateiminusthreecodayiforecast1Mavg= self.diccoEngFeat[(row['numero'], pd.to_datetime(str(dateiminusthree)), 0)][1]
            dateiminusthreeso2dayiforecast1Mavg= self.dicso2EngFeat[(row['numero'], pd.to_datetime(str(dateiminusthree)), 0)][1]

            dateiminusthreepm25dayiforecast1MMax= self.dicpm25EngFeat[(row['numero'], pd.to_datetime(str(dateiminusthree)), 0)][2]
            dateiminusthreeno2dayiforecast1MMax= self.dicno2EngFeat[(row['numero'], pd.to_datetime(str(dateiminusthree)), 0)][2]
            dateiminusthreeo3dayiforecast1MMax= self.dico3EngFeat[(row['numero'], pd.to_datetime(str(dateiminusthree)), 0)][2]
            dateiminusthreepm10dayiforecast1MMax= self.dicpm10EngFeat[(row['numero'], pd.to_datetime(str(dateiminusthree)), 0)][2]
            dateiminusthreecodayiforecast1MMax= self.diccoEngFeat[(row['numero'], pd.to_datetime(str(dateiminusthree)), 0)][2]
            dateiminusthreeso2dayiforecast1MMax= self.dicso2EngFeat[(row['numero'], pd.to_datetime(str(dateiminusthree)), 0)][2]


            dateiminusfourpm25dayiforecast= self.dicpm25[(row['numero'], pd.to_datetime(str(dateiminusfour)), 96)][0]
            dateiminusfourno2dayiforecast= self.dicno2[(row['numero'], pd.to_datetime(str(dateiminusfour)),96)][0]
            dateiminusfouro3dayiforecast= self.dico3[(row['numero'], pd.to_datetime(str(dateiminusfour)), 96)][0]
            dateiminusfourpm10dayiforecast= self.dicpm10[(row['numero'], pd.to_datetime(str(dateiminusfour)), 96)][0]
            dateiminusfourcodayiforecast= self.dicco[(row['numero'], pd.to_datetime(str(dateiminusfour)), 96)][0]
            dateiminusfourso2dayiforecast= self.dicso2[(row['numero'], pd.to_datetime(str(dateiminusfour)), 96)][0]

            dateiminusfourpm25dayiforecast7davg= self.dicpm25EngFeat[(row['numero'], pd.to_datetime(str(dateiminusfour)), 0)][0]
            dateiminusfourno2dayiforecast7davg= self.dicno2EngFeat[(row['numero'], pd.to_datetime(str(dateiminusfour)), 0)][0]
            dateiminusfouro3dayiforecast7davg= self.dico3EngFeat[(row['numero'], pd.to_datetime(str(dateiminusfour)), 0)][0]
            dateiminusfourpm10dayiforecast7davg= self.dicpm10EngFeat[(row['numero'], pd.to_datetime(str(dateiminusfour)), 0)][0]
            dateiminusfourcodayiforecast7davg= self.diccoEngFeat[(row['numero'], pd.to_datetime(str(dateiminusfour)), 0)][0]
            dateiminusfourso2dayiforecast7davg= self.dicso2EngFeat[(row['numero'], pd.to_datetime(str(dateiminusfour)), 0)][0]

            dateiminusfourpm25dayiforecast1Mavg= self.dicpm25EngFeat[(row['numero'], pd.to_datetime(str(dateiminusfour)),0)][1]
            dateiminusfourno2dayiforecast1Mavg= self.dicno2EngFeat[(row['numero'], pd.to_datetime(str(dateiminusfour)), 0)][1]
            dateiminusfouro3dayiforecast1Mavg= self.dico3EngFeat[(row['numero'], pd.to_datetime(str(dateiminusfour)), 0)][1]
            dateiminusfourpm10dayiforecast1Mavg= self.dicpm10EngFeat[(row['numero'], pd.to_datetime(str(dateiminusfour)),0)][1]
            dateiminusfourcodayiforecast1Mavg= self.diccoEngFeat[(row['numero'], pd.to_datetime(str(dateiminusfour)), 0)][1]
            dateiminusfourso2dayiforecast1Mavg= self.dicso2EngFeat[(row['numero'], pd.to_datetime(str(dateiminusfour)), 0)][1]

            dateiminusfourpm25dayiforecast1MMax= self.dicpm25EngFeat[(row['numero'], pd.to_datetime(str(dateiminusfour)), 0)][2]
            dateiminusfourno2dayiforecast1MMax= self.dicno2EngFeat[(row['numero'], pd.to_datetime(str(dateiminusfour)), 0)][2]
            dateiminusfouro3dayiforecast1MMax= self.dico3EngFeat[(row['numero'], pd.to_datetime(str(dateiminusfour)), 0)][2]
            dateiminusfourpm10dayiforecast1MMax= self.dicpm10EngFeat[(row['numero'], pd.to_datetime(str(dateiminusfour)), 0)][2]
            dateiminusfourcodayiforecast1MMax= self.diccoEngFeat[(row['numero'], pd.to_datetime(str(dateiminusfour)), 0)][2]
            dateiminusfourso2dayiforecast1MMax= self.dicso2EngFeat[(row['numero'], pd.to_datetime(str(dateiminusfour)), 0)][2]


        return(dateiminusonepm25dayiforecast,\
                dateiminusoneno2dayiforecast,\
                dateiminusoneo3dayiforecast,\
                dateiminusonepm10dayiforecast,\
                dateiminusonecodayiforecast,\
                dateiminusoneso2dayiforecast,\
                dateiminustwopm25dayiforecast,\
                dateiminustwono2dayiforecast,\
                dateiminustwoo3dayiforecast,\
                dateiminustwopm10dayiforecast,\
                dateiminustwocodayiforecast,\
                dateiminustwoso2dayiforecast,\
                dateiminusthreepm25dayiforecast,\
                dateiminusthreeno2dayiforecast,\
                dateiminusthreeo3dayiforecast,\
                dateiminusthreepm10dayiforecast,\
                dateiminusthreecodayiforecast,\
                dateiminusthreeso2dayiforecast,\
                dateiminusfourpm25dayiforecast,\
                dateiminusfourno2dayiforecast,\
                dateiminusfouro3dayiforecast,\
                dateiminusfourpm10dayiforecast,\
                dateiminusfourcodayiforecast,\
                dateiminusfourso2dayiforecast,\
                dateiminusonepm25dayiforecast7davg,\
                dateiminusoneno2dayiforecast7davg,\
                dateiminusoneo3dayiforecast7davg,\
                dateiminusonepm10dayiforecast7davg,\
                dateiminusonecodayiforecast7davg,\
                dateiminusoneso2dayiforecast7davg,\
                dateiminusonepm25dayiforecast1Mavg,\
                dateiminusoneno2dayiforecast1Mavg,\
                dateiminusoneo3dayiforecast1Mavg,\
                dateiminusonepm10dayiforecast1Mavg,\
                dateiminusonecodayiforecast1Mavg,\
                dateiminusoneso2dayiforecast1Mavg,\
                dateiminusonepm25dayiforecast1MMax,\
                dateiminusoneno2dayiforecast1MMax,\
                dateiminusoneo3dayiforecast1MMax,\
                dateiminusonepm10dayiforecast1MMax,\
                dateiminusonecodayiforecast1MMax,\
                dateiminusoneso2dayiforecast1MMax,\
                dateiminustwopm25dayiforecast7davg,\
                dateiminustwono2dayiforecast7davg,\
                dateiminustwoo3dayiforecast7davg,\
                dateiminustwopm10dayiforecast7davg,\
                dateiminustwocodayiforecast7davg,\
                dateiminustwoso2dayiforecast7davg,\
                dateiminustwopm25dayiforecast1Mavg,\
                dateiminustwono2dayiforecast1Mavg,\
                dateiminustwoo3dayiforecast1Mavg,\
                dateiminustwopm10dayiforecast1Mavg,\
                dateiminustwocodayiforecast1Mavg,\
                dateiminustwoso2dayiforecast1Mavg,\
                dateiminustwopm25dayiforecast1MMax,\
                dateiminustwono2dayiforecast1MMax,\
                dateiminustwoo3dayiforecast1MMax,\
                dateiminustwopm10dayiforecast1MMax,\
                dateiminustwocodayiforecast1MMax,\
                dateiminustwoso2dayiforecast1MMax,\
                dateiminusthreepm25dayiforecast7davg,\
                dateiminusthreeno2dayiforecast7davg,\
                dateiminusthreeo3dayiforecast7davg,\
                dateiminusthreepm10dayiforecast7davg,\
                dateiminusthreecodayiforecast7davg,\
                dateiminusthreeso2dayiforecast7davg,\
                dateiminusthreepm25dayiforecast1Mavg,\
                dateiminusthreeno2dayiforecast1Mavg,\
                dateiminusthreeo3dayiforecast1Mavg,\
                dateiminusthreepm10dayiforecast1Mavg,\
                dateiminusthreecodayiforecast1Mavg,\
                dateiminusthreeso2dayiforecast1Mavg,\
                dateiminusthreepm25dayiforecast1MMax,\
                dateiminusthreeno2dayiforecast1MMax,\
                dateiminusthreeo3dayiforecast1MMax,\
                dateiminusthreepm10dayiforecast1MMax,\
                dateiminusthreecodayiforecast1MMax,\
                dateiminusthreeso2dayiforecast1MMax,\
                dateiminusfourpm25dayiforecast7davg,\
                dateiminusfourno2dayiforecast7davg,\
                dateiminusfouro3dayiforecast7davg,\
                dateiminusfourpm10dayiforecast7davg,\
                dateiminusfourcodayiforecast7davg,\
                dateiminusfourso2dayiforecast7davg,\
                dateiminusfourpm25dayiforecast1Mavg,\
                dateiminusfourno2dayiforecast1Mavg,\
                dateiminusfouro3dayiforecast1Mavg,\
                dateiminusfourpm10dayiforecast1Mavg,\
                dateiminusfourcodayiforecast1Mavg,\
                dateiminusfourso2dayiforecast1Mavg,\
                dateiminusfourpm25dayiforecast1MMax,\
                dateiminusfourno2dayiforecast1MMax,\
                dateiminusfouro3dayiforecast1MMax,\
                dateiminusfourpm10dayiforecast1MMax,\
                dateiminusfourcodayiforecast1MMax,\
                dateiminusfourso2dayiforecast1MMax)

    def compute_target(self, row):
        date = row["date"] + pd.Timedelta("1 days")
        referencedate = self.data["date"].max()
        if date > referencedate:
            return "NaN"
        else:
            return self.dicnewhospi[(row['numero'], pd.to_datetime(str(date)), 0)]

    def compute_Engineered_Features(self, row):
        referencedate = self.data["date"].min()
        datalist = []
        datalist2 = []
        datalist3 = []
        datalist4 = []
        datalist5 = []
        datalist6 = []

        date = row["date"] - pd.Timedelta("30 days")
        date2 = row["date"] - pd.Timedelta("6 days")
        date3 = row["date"] - pd.Timedelta("2 days") 

        dateprevday = row["date"] - pd.Timedelta("1 days")

        dates = pd.date_range(start = date, periods=31).tolist()
        dates2 = pd.date_range(start = date2, periods=7).tolist()
        dates3 = pd.date_range(start = date3, periods = 3).tolist()

        if (dateprevday < referencedate):
            prevdaytothospi = "NaN"
        else:
            prevdaytothospi = self.dictothospi[(row['numero'], dateprevday, 0)] 

        for valuedate in dates:
            if(valuedate < referencedate):
                datalist.append((('NaN','Nan'),('NaN','Nan'),('NaN','Nan'),('NaN','NaN'),('NaN','Nan'),('NaN','NaN')))
            
            else:
                datalist.append((self.dicpm25[(row['numero'], pd.to_datetime(str(valuedate)), 0)], \
                                self.dicno2[(row['numero'], pd.to_datetime(str(valuedate)), 0)], \
                                self.dico3[(row['numero'], pd.to_datetime(str(valuedate)), 0)], \
                                self.dicpm10[(row['numero'], pd.to_datetime(str(valuedate)), 0)],\
                                self.dicco[(row['numero'], pd.to_datetime(str(valuedate)), 0)],
                                self.dicso2[(row['numero'], pd.to_datetime(str(valuedate)), 0)]))


        for valuedate in dates2:
            if(valuedate < referencedate):
                datalist2.append((('NaN','Nan'),('NaN','Nan'),('NaN','Nan'),('NaN','Nan'),('NaN','Nan'),('NaN','NaN')))
                datalist3.append('NaN')
                datalist4.append("NaN")
            
            else:
                datalist2.append((self.dicpm25[(row['numero'], pd.to_datetime(str(valuedate)), 0)], \
                                self.dicno2[(row['numero'], pd.to_datetime(str(valuedate)), 0)], \
                                self.dico3[(row['numero'], pd.to_datetime(str(valuedate)), 0)], \
                                self.dicpm10[(row['numero'], pd.to_datetime(str(valuedate)), 0)],\
                                self.dicco[(row['numero'], pd.to_datetime(str(valuedate)), 0)],
                                self.dicso2[(row['numero'], pd.to_datetime(str(valuedate)), 0)]))
                datalist3.append(self.dicnewhospi[(row['numero'], pd.to_datetime(str(valuedate)), 0)])
                datalist4.append(self.dicnewreanim[(row['numero'], pd.to_datetime(str(valuedate)), 0)])
        
        for valuedate in dates3:
            if(valuedate < referencedate):
                datalist5.append('NaN')
                datalist6.append("NaN")
            
            else:
                datalist5.append(self.dicnewhospi[(row['numero'], pd.to_datetime(str(valuedate)), 0)])
                datalist6.append(self.dicnewreanim[(row['numero'], pd.to_datetime(str(valuedate)), 0)])

        
        cleanedList = [((float(x),float(a)),(float(y),float(b)),(float(z),float(c)),(float(w),float(d)),(float(v),float(e)),(float(r),(float(s)))) \
            for ((x,a),(y,b),(z,c),(w,d),(v,e),(r,s)) in datalist \
                if ((str(x),str(a)),(str(y),str(b)),(str(z),str(c)),(str(w),str(d)),(str(v),str(e)),(str(r),str(s))) \
                    != (('NaN','Nan'),('NaN','Nan'),('NaN','Nan'),('NaN','Nan'),('NaN','Nan'),('NaN','NaN'))]

        cleanedList2 = [((float(x),float(a)),(float(y),float(b)),(float(z),float(c)),(float(w),float(d)),(float(v),float(e)),(float(r),(float(s)))) \
            for ((x,a),(y,b),(z,c),(w,d),(v,e),(r,s)) in datalist2 \
                if ((str(x),str(a)),(str(y),str(b)),(str(z),str(c)),(str(w),str(d)),(str(v),str(e)),(str(r),str(s))) \
                    != (('NaN','Nan'),('NaN','Nan'),('NaN','Nan'),('NaN','Nan'),('NaN','Nan'),('NaN','NaN'))]

        cleanedList3 = [int(x) for x in datalist3 if str(x) != 'NaN']
        cleanedList4 = [int(x) for x in datalist4 if str(x) != 'NaN']
        cleanedList5 = [int(x) for x in datalist5 if str(x) != 'NaN']
        cleanedList6 = [int(x) for x in datalist6 if str(x) != 'NaN']

        avg = [tuple(sum(j)/len(cleanedList) for j in zip(*i)) for i in zip(*cleanedList)]
        avg2 = [tuple(sum(j)/len(cleanedList2) for j in zip(*i)) for i in zip(*cleanedList2)]
        avg3 = statistics.mean(cleanedList3)
        avg4 = statistics.mean(cleanedList4)
        avg5 = statistics.mean(cleanedList5)
        avg6 = statistics.mean(cleanedList6)
        
        return (max(cleanedList,key=itemgetter(0))[0][0],\
                max(cleanedList,key=itemgetter(1))[1][0],\
                max(cleanedList,key=itemgetter(2))[2][0],\
                max(cleanedList,key=itemgetter(3))[3][0],\
                max(cleanedList,key=itemgetter(4))[4][0],
                max(cleanedList,key=itemgetter(5))[5][0],\
                max(cleanedList,key=itemgetter(0))[0][1],\
                max(cleanedList,key=itemgetter(1))[1][1],\
                max(cleanedList,key=itemgetter(2))[2][1],\
                max(cleanedList,key=itemgetter(3))[3][1],\
                max(cleanedList,key=itemgetter(4))[4][1],
                max(cleanedList,key=itemgetter(5))[5][1],\
                prevdaytothospi,\
                avg[0][0],\
                avg[1][0],\
                avg[2][0],\
                avg[3][0],\
                avg[4][0],\
                avg[5][0],\
                avg2[0][0],\
                avg2[1][0],\
                avg2[2][0],\
                avg2[3][0],\
                avg2[4][0],\
                avg2[5][0],\
                avg[0][1],\
                avg[1][1],\
                avg[2][1],\
                avg[3][1],\
                avg[4][1],\
                avg[5][1],\
                avg2[0][1],\
                avg2[1][1],\
                avg2[2][1],\
                avg2[3][1],\
                avg2[4][1],\
                avg2[5][1],\
                avg3,\
                avg4,\
                avg5,\
                avg6)

    def compute_target_assign_to_df(self):
        self.data[['newhospinextday']] = self.data.apply(self.compute_target,axis = 1)
        print("\n")     
        print(self.data)
        print("\n")
        self.data.to_csv('../data/train/all_data_merged/fr/Enriched_Covid_history_data.csv', index = False)
        return None

    def compute_Engineered_features_assign_to_df(self):
        self.data[['1MMaxpm25','1MMaxno2','1MMaxo3','1MMaxpm10','1MMaxco','1MMaxso2',\
                '1MMaxnormpm25','1MMaxnormno2','1MMaxnormo3','1MMaxnormpm10','1MMaxnormco','1MMaxnormso2', 
                'hospiprevday',
                'pm251Mavg','no21Mavg','o31Mavg','pm101Mavg','co1Mavg','so21Mavg',\
                'pm257davg','no27davg','o37davg', 'pm107davg','co7davg','so27davg',\
                "normpm251Mavg","normno21Mavg","normo31Mavg","normpm101Mavg","normco1Mavg",'normso27davg',\
                "normpm257davg","normno27davg","normo37davg","normpm107davg","normco7davg",'normso27davg',\
                "newhospi7davg","newreanim7davg","newhospi3davg","newreanim3davg"]] \
                    = self.data.apply(self.compute_Engineered_Features, axis=1).apply(pd.Series)
        print("\n")
        print(self.data)
        print("\n")
        self.data.to_csv('../data/train/all_data_merged/fr/Enriched_Covid_history_data.csv', index = False)
        return None
    
    def compute_dayi_past_forecasts_assign_to_df(self):
        self.data[["dateiminusonepm25dayiforecast",\
                "dateiminusoneno2dayiforecast",\
                "dateiminusoneo3dayiforecast",\
                "dateiminusonepm10dayiforecast",\
                "dateiminusonecodayiforecast",\
                "dateiminusoneso2dayiforecast",\
                "dateiminustwopm25dayiforecast",\
                "dateiminustwono2dayiforecast",\
                "dateiminustwoo3dayiforecast",\
                "dateiminustwopm10dayiforecast",\
                "dateiminustwocodayiforecast",\
                "dateiminustwoso2dayiforecast",\
                "dateiminusthreepm25dayiforecast",\
                "dateiminusthreeno2dayiforecast",\
                "dateiminusthreeo3dayiforecast",\
                "dateiminusthreepm10dayiforecast",\
                "dateiminusthreecodayiforecast",\
                "dateiminusthreeso2dayiforecast",\
                "dateiminusfourpm25dayiforecast",\
                "dateiminusfourno2dayiforecast",\
                "dateiminusfouro3dayiforecast",\
                "dateiminusfourpm10dayiforecast",\
                "dateiminusfourcodayiforecast",\
                "dateiminusfourso2dayiforecast",
                "dateiminusonepm25dayiforecast7davg",\
                "dateiminusoneno2dayiforecast7davg",\
                "dateiminusoneo3dayiforecast7davg",\
                "dateiminusonepm10dayiforecast7davg",\
                "dateiminusonecodayiforecast7davg",\
                "dateiminusoneso2dayiforecast7davg",\
                "dateiminusonepm25dayiforecast1Mavg",\
                "dateiminusoneno2dayiforecast1Mavg",\
                "dateiminusoneo3dayiforecast1Mavg",\
                "dateiminusonepm10dayiforecast1Mavg",\
                "dateiminusonecodayiforecast1Mavg",\
                "dateiminusoneso2dayiforecast1Mavg",\
                "dateiminusonepm25dayiforecast1MMax",\
                "dateiminusoneno2dayiforecast1MMax",\
                "dateiminusoneo3dayiforecast1MMax",\
                "dateiminusonepm10dayiforecast1MMax",\
                "dateiminusonecodayiforecast1MMax",\
                "dateiminusoneso2dayiforecast1MMax",\
                "dateiminustwopm25dayiforecast7davg",\
                "dateiminustwono2dayiforecast7davg",\
                "dateiminustwoo3dayiforecast7davg",\
                "dateiminustwopm10dayiforecast7davg",\
                "dateiminustwocodayiforecast7davg",\
                "dateiminustwoso2dayiforecast7davg",\
                "dateiminustwopm25dayiforecast1Mavg",\
                "dateiminustwono2dayiforecast1Mavg",\
                "dateiminustwoo3dayiforecast1Mavg",\
                "dateiminustwopm10dayiforecast1Mavg",\
                "dateiminustwocodayiforecast1Mavg",\
                "dateiminustwoso2dayiforecast1Mavg",\
                "dateiminustwopm25dayiforecast1MMax",\
                "dateiminustwono2dayiforecast1MMax",\
                "dateiminustwoo3dayiforecast1MMax",\
                "dateiminustwopm10dayiforecast1MMax",\
                "dateiminustwocodayiforecast1MMax",\
                "dateiminustwoso2dayiforecast1MMax",\
                "dateiminusthreepm25dayiforecast7davg",\
                "dateiminusthreeno2dayiforecast7davg",\
                "dateiminusthreeo3dayiforecast7davg",\
                "dateiminusthreepm10dayiforecast7davg",\
                "dateiminusthreecodayiforecast7davg",\
                "dateiminusthreeso2dayiforecast7davg",\
                "dateiminusthreepm25dayiforecast1Mavg",\
                "dateiminusthreeno2dayiforecast1Mavg",\
                "dateiminusthreeo3dayiforecast1Mavg",\
                "dateiminusthreepm10dayiforecast1Mavg",\
                "dateiminusthreecodayiforecast1Mavg",\
                "dateiminusthreeso2dayiforecast1Mavg",\
                "dateiminusthreepm25dayiforecast1MMax",\
                "dateiminusthreeno2dayiforecast1MMax",\
                "dateiminusthreeo3dayiforecast1MMax",\
                "dateiminusthreepm10dayiforecast1MMax",\
                "dateiminusthreecodayiforecast1MMax",\
                "dateiminusthreeso2dayiforecast1MMax",\
                "dateiminusfourpm25dayiforecast7davg",\
                "dateiminusfourno2dayiforecast7davg",\
                "dateiminusfouro3dayiforecast7davg",\
                "dateiminusfourpm10dayiforecast7davg",\
                "dateiminusfourcodayiforecast7davg",\
                "dateiminusfourso2dayiforecast7davg",\
                "dateiminusfourpm25dayiforecast1Mavg",\
                "dateiminusfourno2dayiforecast1Mavg",\
                "dateiminusfouro3dayiforecast1Mavg",\
                "dateiminusfourpm10dayiforecast1Mavg",\
                "dateiminusfourcodayiforecast1Mavg",\
                "dateiminusfourso2dayiforecast1Mavg",\
                "dateiminusfourpm25dayiforecast1MMax",\
                "dateiminusfourno2dayiforecast1MMax",\
                "dateiminusfouro3dayiforecast1MMax",\
                "dateiminusfourpm10dayiforecast1MMax",\
                "dateiminusfourcodayiforecast1MMax",\
                "dateiminusfourso2dayiforecast1MMax"]] = self.data.apply(self.compute_forecast_data_for_training_models, axis = 1).apply(pd.Series)
        
        print("\n")
        print(self.data)
        print("\n")
        self.data.to_csv('../data/train/all_data_merged/fr/Enriched_Covid_history_data.csv', index = False)
        return None

if __name__ == '__main__':
    Engineered_Features = Compute_Engineered_Features_for_df()
    Engineered_Features.get_data()
    Engineered_Features.max_normalize_data()
    Engineered_Features.compute_dictionnaries()
    #Engineered_Features.compute_Engineered_features_assign_to_df()
    #Engineered_Features.compute_dayi_past_forecasts_assign_to_df()
    Engineered_Features.compute_target_assign_to_df()


   

    
        




