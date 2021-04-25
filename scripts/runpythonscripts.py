
from maintraindata import  maintraindata
from compute_engineered_features import Compute_Engineered_Features_for_df
from maintrain import maintrain
from compute_covid_risk_heat_map import compute_covid_risk_heat_map
from compute_pollution_levels_maps import compute_maps
class runpythonscripts:
    
    def __init__(self):

        self.status = None
    
    def runscripts(self,skiptpot = None, skipimpute = None, skipalltrain = None, train = None,skipcrossval =None, skipgetdata = None, skipallengineer = None,  skipengineerfeatures = None, skipcomputedayipastdata = None, skiptrain = None, load = None):
        GetData = maintraindata()
        if skipgetdata == None:
            GetData.GetHistoricalData()
        EngineerFeatures = Compute_Engineered_Features_for_df()
        
        if skipallengineer == None:
            EngineerFeatures.get_data(load)
            EngineerFeatures.max_normalize_data()
            EngineerFeatures.compute_dictionnaries()
            if skipengineerfeatures == None:
                EngineerFeatures.compute_Engineered_features_assign_to_df()
            EngineerFeatures.compute_avg_and_max_dictionnaries()
            if skipcomputedayipastdata == None:
                EngineerFeatures.compute_dayi_past_data_assign_to_df()

            EngineerFeatures.compute_target_assign_to_df()
            EngineerFeatures.compute_dfs_from_which_to_make_predictions()
        
        if skipalltrain == None:
            TrainModel = maintrain(skipcrossval, skipimpute, skiptpot)
            TrainModel.initdata()
            if skiptrain == None:
                TrainModel.CurrentBestModel()

            TrainModel.predict()
        ComputeMap = compute_covid_risk_heat_map()
        ComputeMap.compute_map()
        ComputeMaps = compute_maps()
        ComputeMaps.compute_maps()

        self.status = "OK Computed Maps"
        print(self.status)

        return self.status

if __name__ == "__main__":

    Run = runpythonscripts()
    Run.runscripts(skiptpot = "y",\
                   skipimpute = "Y",\
                   skiptrain = "Y",\
                   skipalltrain = "Y",\
                   skipgetdata = "Y",\
                   skipallengineer= "y",\
                   skipengineerfeatures = "y",\
                   skipcomputedayipastdata = "y",\
                   load = 1, skipcrossval = "Y")
    #Run.runscripts()

        