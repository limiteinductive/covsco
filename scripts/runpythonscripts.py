
from maintraindata import  maintraindata
from compute_engineered_features import Compute_Engineered_Features_for_df
from maintrain import maintrain
from compute_covid_risk_heat_map import compute_covid_risk_heat_map
from compute_pollution_levels_maps import compute_maps
from compute_new_hospi_map import compute_new_hospi_map
class runpythonscripts:
    
    def __init__(self):

        self.status = None
    
    def runscripts(self, datastartdate, skiplevelmaps = None, skipcovidriskheatmap = None, skiptpot = None, skipimpute = None, skipalltrain = None, train = None,skipcrossval =None, skipgetdata = None, skiptrain = None, load = None):
        GetData = maintraindata(datastartdate)
        if skipgetdata == None:
            GetData.GetHistoricalData()
        
        TrainModel = maintrain(skipcrossval, skipimpute, skiptpot)
        TrainModel.initdata()
        if skipalltrain == None:
            
            if skiptrain == None:
                TrainModel.CurrentBestModel()

        TrainModel.predict()
        
        if skipcovidriskheatmap == None:
            ComputeMap = compute_covid_risk_heat_map()
            ComputeMap.compute_map()

        if skiplevelmaps == None:
            ComputeMaps = compute_maps()
            ComputeMaps.compute_maps()

        NewHospiMap = compute_new_hospi_map()
        NewHospiMap.compute_map()
        self.status = "OK Computed Maps"
        print(self.status)

        return self.status

if __name__ == "__main__":

    Run = runpythonscripts()
    Run.runscripts("2020-04-08", skiplevelmaps = "True",\
                   skipcovidriskheatmap = "Y",\
                   skiptpot = "y",\
                   skipimpute = "Y",\
                   skiptrain = "Y",\
                   skipalltrain = "Y",\
                   skipgetdata = "Y",\
                   load = 1, skipcrossval = "Y")
    #Run.runscripts("2020-04-08", skipcovidriskheatmap='Y', skiptpot='Y')

        