import pandas as pd
import re
# Population
# -----------------------------------------------------------------------------
class process_population_hist:
    def __init__(self):
        self.file_name = None
        self.file_name2 = None
        self.population = None
        self.dep_centre = None

    def parse_dsm(self, coord):
        deg, min, sec, dir = re.split('[°\'"]', coord)
        dd = float(deg) + (float(min)/60) + (float(sec)/60/60)
        if (dir == 'W') | (dir == 'S'):
            dd *= -1
        return dd
    # =============================================================================
    # Data
    # =============================================================================
   
    def get_data(self):
        print('Processing Population data ... ', flush=True, end='')
        self.file_name = '/home/ludo915/code/covsco/data/train/pop/fr/departements-francais.csv'
        self.population  = pd.read_csv(self.file_name, sep=';')
        self.population.columns = ['dep_num', 'name', 'region', 'capital', 'area', 'total', 'density']
        #self.population['dep_num'] = self.population['dep_num'].replace({'2A':'201','2B':'202'}).astype(int)
        self.population = self.population.sort_values('dep_num')
        self.population = self.population[:-5]
        self.file_name2 = '/home/ludo915/code/covsco/data/train/pop/fr/Centre_departement.xlsx'
        self.dep_centre = pd.read_excel(self.file_name2, engine='openpyxl', header=1, usecols=[0,1,2,3,4])
        self.dep_centre.columns = ['dep_num','name','area', 'lon', 'lat']
        self.dep_centre['dep_num'] = self.dep_centre['dep_num'].replace({'2A':'201','2B':'202'}).astype(int)
        self.dep_centre = self.dep_centre.sort_values('dep_num')
        self.dep_centre['lon'] = self.dep_centre['lon'].apply(lambda x: self.parse_dsm(x))
        self.dep_centre['lat'] = self.dep_centre['lat'].apply(lambda x: self.parse_dsm(x))
        self.dep_centre = self.dep_centre.merge(self.population, on=['dep_num'], how='outer')
        self.dep_centre = self.dep_centre.drop(columns=['name_x', 'area_x', 'region'])
        self.dep_centre.columns = ['dep_num','lon','lat','name','captial','area','total','density']
        self.dep_centre.to_csv('/home/ludo915/code/covsco/data/train/pop/fr/population_2020.csv', index=False)
        self.population  = pd.read_csv('/home/ludo915/code/covsco/data/train/pop/fr/population_2020.csv')
        self.population['idx'] = self.population['total']
        self.population.reset_index(inplace = True, drop=True)
        print('OK', flush=True)
        print("\n")


if __name__ == '__main__':
    GetPopulationData = process_population_hist()
    GetPopulationData.get_data()