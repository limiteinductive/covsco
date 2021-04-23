
from maintraindata import  maintraindata
from compute_engineered_features import Compute_Engineered_Features_for_df
from maintrain import maintrain
from compute_covid_risk_heat_map import compute_covid_risk_heat_map

class runpythonscripts:
    
    def __init__(self):

        self.status = None
    
    def runscripts(self, train = None, skipgetdata = None, skipengineerfeatures = None, skipcomputedayipastdata = None, skiptrain = None, load = None):
        GetData = maintraindata()
        if skipgetdata == None:
            GetData.GetHistoricalData()
        EngineerFeatures = Compute_Engineered_Features_for_df()
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
        
        TrainModel = maintrain()
        TrainModel.initdata()
        if skiptrain == None:
            TrainModel.CurrentBestModel()

        TrainModel.predict()
        ComputeMap = compute_covid_risk_heat_map()
        ComputeMap.compute_map()
        self.status = "OK Computed Map"
        print(self.status)

        return self.status

if __name__ == "__main__":

    Run = runpythonscripts()
    Run.runscripts(skipgetdata = "Y", skipengineerfeatures = "Y", skipcomputedayipastdata = None, load = 1)
    #Run.runscripts()

        