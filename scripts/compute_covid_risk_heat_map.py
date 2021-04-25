import sys
import datetime as dt
from datetime import datetime
import numpy as np
import pandas as pd
import xarray as xr
import regionmask
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import io
import imageio
from os import listdir
from os.path import isfile, join
from tqdm import tqdm

class compute_covid_risk_heat_map:

  def __init__(self):

    self.status = None
  
  def max_normalize(self, x):
    return (x - x.min()) / (x.max() - x.min())

  def progressbar(self, it, prefix="", size=60, file=sys.stdout):
      count = len(it)
      def show(j):
          x = int(size*j/count)
          file.write("%s[%s%s] %i/%i\r" % (prefix, "#"*x, "."*(size-x), j, count))
          file.flush()
      show(0)
      for i, item in enumerate(it):
          yield item
          show(i+1)
      file.write("\n")
      file.flush()
      return None
  
  def findlatestdateofcamsdata(self, mypath):
          dates = []
          onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
          for filename in onlyfiles:
              dates.append(pd.to_datetime(filename[14:24]))
          
          if dates != []:
              return (dates, max(dates))
          else:
              return (dates, dt.date.today() - pd.Timedelta("3 Y"))

  def compute_map(self):

    grayDark = '#e1e1e1'
    grayLight = '#404040'

    sns.set(
      context 	= 'paper',
      style 		= 'dark',
      palette 	= 'muted',
      color_codes = True,
        font_scale  = 2.0,
      font 		= 'sans-serif',
      rc={
        'axes.edgecolor'	: grayDark
        ,'text.color' 		: grayDark
        ,'axes.labelcolor' 	: grayDark
        ,'xtick.color' 		: grayDark
        ,'ytick.color' 		: grayDark
            ,'figure.facecolor' : grayLight
            ,'axes.facecolor'   : grayLight
            ,'savefig.facecolor': grayLight
        }
    )

    print('Data pre-processing:', flush=True)

    countryEU = regionmask.defined_regions.natural_earth.countries_50
    #currentDate = datetime.today().strftime('%Y-%m-%d')
    

    print('Population ... ', flush=True, end='')
    pop         = pd.read_csv('../data/pop/fr/pop.csv', usecols=[0,1,2,3,4,5,6,42])
    pop.columns = ['reg', 'dep', 'com', 'article', 'com_nom', 'lon', 'lat', 'total']
    popDEP          = pop.copy().groupby('dep').median()
    popDEP['total'] = pop.groupby('dep').sum()['total']
    pop['idx']    = self.max_normalize((pop['total']))
    popDEP['idx'] = self.max_normalize((popDEP['total']))
    print('OK', flush=True)

    print('Covid ... ', flush=True, end='')
    filePath = '../data/train/covid/fr/'
    fileName = 'Covid_data_history.csv'
    covid = pd.read_csv(filePath + fileName, sep=',').dropna()

    dfpollution2 = pd.read_csv("../data/train/all_data_merged/fr/Enriched_Covid_history_data.csv")
    dfpollution2= dfpollution2.dropna()
    dfpollution = pd.read_csv("../data/train/all_data_merged/fr/Enriched_Covid_history_data.csv")
    dfpollution3 = pd.read_csv("../data/train/all_data_merged/fr/traindf.csv")
   
    dfpollution= dfpollution.dropna()
    dfpollution2 = dfpollution2.dropna()
    dfpollution3['newhospinextday'] = dfpollution3['newhospinextday'].fillna(9999)
    dfpollution3 = dfpollution3.dropna()
    currentDate = dfpollution3["date"].max()
    dfpollution=dfpollution[dfpollution["date"]==dfpollution["date"].max()]
    dfpollution3=dfpollution3[dfpollution3["date"]==dfpollution3["date"].max()]
    print(dfpollution3)
    covid = covid.groupby('numero').rolling(window=7).mean()
    covid = covid.groupby(level=0).tail(1).reset_index(drop=True)
    print ("Computing all department longitudes and latitudes in dataframes...")
    popSubset = pop[['lon','lat','dep']].drop_duplicates(subset=['dep'])
    covid['lon'] = [popSubset[popSubset['dep']==int(depNum)].lon.values.squeeze() for depNum in covid['numero']]
    covid['lat'] = [popSubset[popSubset['dep']==int(depNum)].lat.values.squeeze() for depNum in covid['numero']]
    dfpollution['lon'] =[popSubset[popSubset['dep']==int(depNum)].lon.values.squeeze() for depNum in dfpollution['numero']]
    dfpollution['lat'] = [popSubset[popSubset['dep']==int(depNum)].lat.values.squeeze() for depNum in dfpollution['numero']]
    dfpollution3['lon'] =[popSubset[popSubset['dep']==int(depNum)].lon.values.squeeze() for depNum in dfpollution3['numero']]
    dfpollution3['lat'] = [popSubset[popSubset['dep']==int(depNum)].lat.values.squeeze() for depNum in dfpollution3['numero']]
    # remove French oversea departments
    covid = covid[:-5]

    # extrapolate covid cases from deprtement to commune level
    covidExtraToCom = pop.copy()
    covidExtraToCom['hospi'] = [covid[covid['numero'] == depNum].hospi.values.squeeze() for depNum in covidExtraToCom['dep']]

    times = ["0 days","1 days","2 days", "3 days",'4 days']
    counter = 0
    images = []
    riskMaps = []
    currentDatestring = (pd.to_datetime(dfpollution3["date"].max())+pd.Timedelta("1 Days")).strftime('%Y-%m-%d')
    print(currentDatestring)
    print("Interpolating engineered features to commune level...")
      # covidExtraToCom[['1MMaxpm25','1MMaxpm10','1MMaxo3','1MMaxno2','1MMaxco','pm107davg','pm257davg','o37davg','no27davg','co7davg','pm101Mavg',\
      #   'pm251Mavg','o31Mavg','no21Mavg','co1Mavg','population','hospi','CovidPosTest' ]] \
      #     = [dfpollution3[dfpollution3['numero'] == depNum].reindex(columns = ['1MMaxpm25','1MMaxpm10','1MMaxo3','1MMaxno2','1MMaxco','pm107davg','pm257davg','o37davg','no27davg','co7davg','pm101Mavg',\
      #   'pm251Mavg','o31Mavg','no21Mavg','co1Mavg','idx','hospi','CovidPosTest' ]).values.squeeze() for depNum in covidExtraToCom['dep']]
    columns = ['1MMaxpm25','1MMaxpm10','1MMaxo3','1MMaxno2','1MMaxco','pm107davg','pm257davg','o37davg','no27davg','co7davg','pm101Mavg',\
                 'pm251Mavg','o31Mavg','no21Mavg','co1Mavg','hospi','CovidPosTest' ]

    for col in columns:
      covidExtraToCom[col] = [dfpollution3[dfpollution3['numero'] == depNum][col].values.squeeze() for depNum in covidExtraToCom['dep']]
    # covidExtraToCom['1MMaxpm25'] = [dfpollution3[dfpollution3['numero'] == depNum]["1MMaxpm25"].values.squeeze() for depNum in covidExtraToCom['dep']]
    # print("1MMaxpm25 interpolated!")
    # covidExtraToCom['1MMaxpm10'] = [dfpollution3[dfpollution3['numero'] == depNum]["1MMaxpm10"].values.squeeze() for depNum in covidExtraToCom['dep']]
    # print("1MMaxpm10 interpolated!")
    # covidExtraToCom['1MMaxo3'] = [dfpollution3[dfpollution3['numero'] == depNum]["1MMaxo3"].values.squeeze() for depNum in covidExtraToCom['dep']]
    # print("1MMaxo3 interpolated!")
    # covidExtraToCom['1MMaxno2'] = [dfpollution3[dfpollution3['numero'] == depNum]["1MMaxno2"].values.squeeze() for depNum in covidExtraToCom['dep']]
    # print("1MMaxno2 interpolated!")
    # covidExtraToCom['1MMaxco'] = [dfpollution3[dfpollution3['numero'] == depNum]["1MMaxco"].values.squeeze() for depNum in covidExtraToCom['dep']]
    # print("1MMaxco interpolated!")
    # covidExtraToCom['pm107davg'] = [dfpollution3[dfpollution3['numero'] == depNum]["pm107davg"].values.squeeze() for depNum in covidExtraToCom['dep']]
    # print("pm107davg interpolated!")
    # covidExtraToCom['pm257davg'] = [dfpollution3[dfpollution3['numero'] == depNum]["pm257davg"].values.squeeze() for depNum in covidExtraToCom['dep']]
    # print("pm257davg interpolated!")
    # covidExtraToCom['o37davg'] = [dfpollution3[dfpollution3['numero'] == depNum]["o37davg"].values.squeeze() for depNum in covidExtraToCom['dep']]
    # print("o37davg interpolated!")
    # covidExtraToCom['no27davg'] = [dfpollution3[dfpollution3['numero'] == depNum]["no27davg"].values.squeeze() for depNum in covidExtraToCom['dep']]
    # print("no27davg interpolated!")
    # covidExtraToCom['co7davg'] = [dfpollution3[dfpollution3['numero'] == depNum]["co7davg"].values.squeeze() for depNum in covidExtraToCom['dep']]
    # print("co7davg interpolated!")
    # covidExtraToCom['pm101Mavg'] = [dfpollution3[dfpollution3['numero'] == depNum]["pm101Mavg"].values.squeeze() for depNum in covidExtraToCom['dep']]
    # print("pm101Mavg interpolated!")
    # covidExtraToCom['pm251Mavg'] = [dfpollution3[dfpollution3['numero'] == depNum]["pm251Mavg"].values.squeeze() for depNum in covidExtraToCom['dep']]
    # print("pm251Mavg interpolated!")
    # covidExtraToCom['o31Mavg'] = [dfpollution3[dfpollution3['numero'] == depNum]["o31Mavg"].values.squeeze() for depNum in covidExtraToCom['dep']]
    # print("o31Mavg interpolated!")
    # covidExtraToCom['no21Mavg'] = [dfpollution3[dfpollution3['numero'] == depNum]["no21Mavg"].values.squeeze() for depNum in covidExtraToCom['dep']]
    # print("no21Mavg interpolated!")
    # covidExtraToCom['co1Mavg'] = [dfpollution3[dfpollution3['numero'] == depNum]["co1Mavg"].values.squeeze() for depNum in covidExtraToCom['dep']]
    # print("co1Mavg interpolated!")
    # covidExtraToCom['population'] = [dfpollution3[dfpollution3['numero'] == depNum]["idx"].values.squeeze() for depNum in covidExtraToCom['dep']]
    # print("population interpolated!")
    # covidExtraToCom['hospi'] = [dfpollution3[dfpollution3['numero'] == depNum]["hospi"].values.squeeze() for depNum in covidExtraToCom['dep']]
    # print("hospi interpolated!")
    # covidExtraToCom['CovidPosTest'] = [dfpollution3[dfpollution3['numero'] == depNum]["CovidPosTest"].values.squeeze() for depNum in covidExtraToCom['dep']]
    # print("CovidPosTest interpolated!")

    for j in tqdm(times):
      filename = "../predictions/fr/" + currentDatestring + "_predictions_for_day_" + str(counter) +".csv"
      newhospipredictionsdf = pd.read_csv(filename)
      print(filename + " Read!")

      print("Interpolating newhospi predictions to commune level...")
      covidExtraToCom['newhospipred'] = [newhospipredictionsdf[newhospipredictionsdf['depnum'] == depNum]["newhospipred"].values.squeeze() for depNum in covidExtraToCom['dep']]
      print("newhospipred interpolated")
      print('OK', flush=True)

      filePath = '../data/train/cams/fr/forecast/'
      #latestfiledatestring = self.findlatestdateofcamsdata(filePath)[1].strftime('%Y-%m-%d')
      #latestfiledatestring = self.findlatestdateofcamsdata(filePath)[1].strftime('%Y-%m-%d')
      fileName = "cams-forecast-"+currentDatestring +".nc"
      pollutants = xr.open_dataset(filePath + fileName).sel(time = j)

      pm25 = pollutants.pm2p5_conc
      o3 = pollutants.o3_conc
      no2 = pollutants.no2_conc
      co = pollutants.co_conc
      pm10 = pollutants.pm10_conc
      so2 = pollutants.so2_conc

      pm25 = pm25.drop('time').squeeze().drop('level').squeeze()
      o3 = o3.drop('time').squeeze().drop('level').squeeze()
      no2 = no2.drop('time').squeeze().drop('level').squeeze()
      co = co.drop('time').squeeze().drop('level').squeeze()
      pm10 = pm10.drop('time').squeeze().drop('level').squeeze()
      so2 = so2.drop('time').squeeze().drop('level').squeeze()

      pm25.coords['longitude'] = (pm25.coords['longitude'] + 180) % 360 - 180
      o3.coords['longitude'] = (o3.coords['longitude'] + 180) % 360 - 180
      no2.coords['longitude'] = (no2.coords['longitude'] + 180) % 360 - 180
      co.coords['longitude'] = (co.coords['longitude'] + 180) % 360 - 180
      pm10.coords['longitude'] = (pm10.coords['longitude'] + 180) % 360 - 180
      so2.coords['longitude'] = (so2.coords['longitude'] + 180) % 360 -180

      pm25 = pm25.sortby('longitude')
      pm25 = pm25.sel(longitude=slice(-10,10),latitude=slice(55,40))
      o3 = o3.sortby('longitude')
      o3 = o3.sel(longitude=slice(-10,10),latitude=slice(55,40))
      no2 = no2.sortby('longitude')
      no2 = no2.sel(longitude=slice(-10,10),latitude=slice(55,40))
      co = co.sortby('longitude')
      co = co.sel(longitude=slice(-10,10),latitude=slice(55,40))
      pm10 = pm10.sortby('longitude')
      pm10 = pm10.sel(longitude=slice(-10,10),latitude=slice(55,40))
      so2 = so2.sortby('longitude')
      so2 = so2.sel(longitude=slice(-10,10),latitude=slice(55,40))
      print('OK')

      # =============================================================================
      # Interpolation
      # =============================================================================
      lons, lats = pop.lon, pop.lat
      xrLons = xr.DataArray(lons, dims='com')
      xrLats = xr.DataArray(lats, dims='com')
      pm25Interpolated = pm25.interp(longitude=xrLons, latitude=xrLats)
      o3Interpolated = o3.interp(longitude=xrLons, latitude=xrLats)
      no2Interpolated = no2.interp(longitude=xrLons, latitude=xrLats)
      coInterpolated = co.interp(longitude=xrLons, latitude=xrLats)
      pm10Interpolated = pm10.interp(longitude=xrLons, latitude=xrLats)
      so2Interpolated = so2.interp(longitude=xrLons, latitude=xrLats)
      
      #endpart

      # =============================================================================
      # Risk Assessment
      # =============================================================================
      #part| #%%
     
      #for lead in progressbar(range(97), 'Compute risk: ', 60):
      print("Computing Risk of severe respiratory diseases summoned by pollution ...")
      risk =    0.0397*self.max_normalize(coInterpolated)\
              + 0.0414*self.max_normalize(o3Interpolated)\
              + 0.0371 * self.max_normalize(pm25Interpolated)\
              + 0.0327* self.max_normalize(no2Interpolated) \
              + 0.0361 * self.max_normalize(pm10Interpolated) \
              + 0.0350* self.max_normalize(so2Interpolated) \
              + 0.0250 * self.max_normalize(covidExtraToCom["1MMaxo3"])\
              + 0.0192*self.max_normalize(covidExtraToCom["1MMaxpm10"])\
              + 0.0174*self.max_normalize(covidExtraToCom["1MMaxno2"])\
              + 0.0287*self.max_normalize(covidExtraToCom["1MMaxpm25"])\
              + 0.0632*self.max_normalize(covidExtraToCom["1MMaxco"])\
              + 0.0201*self.max_normalize(covidExtraToCom["1MMaxso2"])\
              + 0.0530*self.max_normalize(covidExtraToCom["o37davg"])\
              + 0.0485*self.max_normalize(covidExtraToCom["pm107davg"])\
              + 0.0233*self.max_normalize(covidExtraToCom["no27davg"])\
              + 0.0327*self.max_normalize(covidExtraToCom["pm257davg"])\
              + 0.1773*self.max_normalize(covidExtraToCom["co7davg"])\
              + 0.0355*self.max_normalize(covidExtraToCom["so27davg"])\
              + 0.0515*self.max_normalize(covidExtraToCom["o31Mavg"])\
              + 0.0320*self.max_normalize(covidExtraToCom["pm101Mavg"])\
              + 0.0298*self.max_normalize(covidExtraToCom["no21Mavg"])\
              + 0.0293*self.max_normalize(covidExtraToCom["pm251Mavg"])\
              + 0.0293*self.max_normalize(covidExtraToCom["so21Mavg"])\
              + 0.0337*self.max_normalize(covidExtraToCom["co1Mavg"])\


      risk = np.array(risk)
      risk = np.vstack((lons,lats,risk))
      risk = risk.T

      risk = pd.DataFrame(risk, columns = ['lon', 'lat', 'idx'])

      riskMaps.append(risk)

      markersize = .1
     

      fig = plt.figure(figsize=(8,8))
      gs = fig.add_gridspec(1, 1)
      ax1 = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
      axes = [ax1]
      ax1.background_patch.set_fill(False)
      for a in axes:
          a.add_geometries(countryEU['F'].polygon, ccrs.PlateCarree(),
          edgecolor=grayDark, lw=2, facecolor=grayDark, alpha=0.6, zorder=0)
          a.set_extent([-5,10,41,52])
          a.set_aspect('auto')
          a.outline_patch.set_linewidth(0.)
          pass

      cax = ax1.scatter(riskMaps[counter].lon,riskMaps[counter].lat,c=riskMaps[counter].idx,
      cmap='RdYlGn_r', s=markersize*5, vmin=0, vmax=1, zorder=4)
      cbar = fig.colorbar(cax, orientation='horizontal', pad=0, aspect=50,
      fraction=.01, extend='max', drawedges=False, ticks=[0, 1])
      cbar.ax.set_xticklabels(['low', 'high'])
      cbar.ax.xaxis.set_ticks_position('top')
      cbar.ax.xaxis.set_label_position('top')

      ax1.text(0,.0,'Data \nCAMS \ndata.gouv.fr', transform=ax1.transAxes,fontdict={'size':12})

      currentDateWD = (pd.to_datetime(dfpollution3["date"].max()) +pd.Timedelta("1 Days")).strftime('%a, %d %b %Y')
      ax1.set_title('Risk of severe Covid-19 cases: \n{:}\n'.format(currentDateWD + " + "+ str (counter) + " days"),
      loc='left', pad=-60)

      fig.subplots_adjust(bottom=.01, left=.01, right=.99, top=.99)
      plt.show()
      buffer = io.BytesIO()
      plt.savefig(buffer, format='png', dpi=70)
      buffer.seek(0)
      images.append(imageio.imread(buffer))
      buffer.close()
      plt.close()
      counter += 1

    print('Create gif ...', flush=True, end='')
    gifPath = '../forecast/fr/'
    gifName = 'covid-risk-fc-{:}.gif'.format(currentDate)
    kargs = { 'duration': 1 }
    imageio.mimwrite(gifPath + gifName, images, 'GIF', **kargs)
    print('OK')
    print('Finished.')

    return None

if __name__ == '__main__':
  RiskLevelMaps = compute_covid_risk_heat_map()
  RiskLevelMaps.compute_map()
