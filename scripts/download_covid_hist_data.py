import requests
import pandas as pd
import datetime
from tqdm import tqdm

class download_covid_hist_data:

    def __init__(self):

        self.file_name = None
        self.departements = None
        self.url = None
        self.data = None

    def GetData(self):
        print("Downloading Covid Numbers from coronavirusapi-france.now ...")
        self.file_name = "../data/train/pop/fr/departements-francais.csv"
        self.departements = pd.read_csv(self.file_name, sep = ";")
        self.url = "https://coronavirusapi-france.now.sh/AllDataByDepartement?Departement="

        datapointlist = []

        for nom in tqdm(self.departements.NOM):
            url2 = self.url + nom
            numero = str(self.departements[self.departements["NOM"]==nom]["NUMÉRO"].values[0])
            response = requests.get(url2).json()
            referencedate = datetime.datetime.strptime("2020-04-08", '%Y-%m-%d')
            for i in range(len(response["allDataByDepartement"])):
                date = datetime.datetime.strptime(response["allDataByDepartement"][i]["date"], '%Y-%m-%d')
                if (date > referencedate):
                    if "hospitalises" in response["allDataByDepartement"][i].keys():
                        hospitalises = response["allDataByDepartement"][i]["hospitalises"]
                    else:
                        hospitalises = response["allDataByDepartement"][i]["hospitalisation"]
                    reanimation = response["allDataByDepartement"][i]["reanimation"]
                    nouvellesHospitalisations = response["allDataByDepartement"][i]["nouvellesHospitalisations"]
                    nouvellesReanimations = response["allDataByDepartement"][i]["nouvellesReanimations"]
                    deces = response["allDataByDepartement"][i]["deces"]
                    gueris = response["allDataByDepartement"][i]["gueris"] 
                    datapointlist.append((nom,numero,date,hospitalises,reanimation,nouvellesHospitalisations,nouvellesReanimations,deces,gueris))

        dfcolumns = ['nom', 'numero','date','hospi','reanim','newhospi','newreanim','deces','gueris']
        self.data = pd.DataFrame(datapointlist, columns = dfcolumns)
        self.data.to_csv('../data/train/covid/fr/Covid_data_history.csv', index = False)
        print(self.data)
        return None


if __name__ == '__main__':
    CovidHistData = download_covid_hist_data()
    CovidHistData.GetData()
