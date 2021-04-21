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
    countryBorders = cfeature.NaturalEarthFeature(
        category    = 'cultural',
        name        = 'ne_admin_0_countries',
        scale       = '10m',
        facecolor   = 'none'
    )

    countryEU = regionmask.defined_regions.natural_earth.countries_50
    currentDate = datetime.today().strftime('%Y-%m-%d')

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
    dfpollution= dfpollution.dropna()

    dfpollution=dfpollution[dfpollution["date"]==dfpollution["date"].max()]
   
    covid = covid.groupby('numero').rolling(window=7).mean()
    covid = covid.groupby(level=0).tail(1).reset_index(drop=True)

    popSubset = pop[['lon','lat','dep']].drop_duplicates(subset=['dep'])
    covid['lon'] = [popSubset[popSubset['dep']==int(depNum)].lon.values.squeeze() for depNum in covid['numero']]
    covid['lat'] = [popSubset[popSubset['dep']==int(depNum)].lat.values.squeeze() for depNum in covid['numero']]
    dfpollution['lon'] =[popSubset[popSubset['dep']==int(depNum)].lon.values.squeeze() for depNum in dfpollution['numero']]
    dfpollution['lat'] = [popSubset[popSubset['dep']==int(depNum)].lat.values.squeeze() for depNum in dfpollution['numero']]
    # remove French oversea departments
    covid = covid[:-5]

    # extrapolate covid cases from deprtement to commune level
    covidExtraToCom = pop.copy()
    covidExtraToCom['hospi'] = [covid[covid['numero'] == depNum].hospi.values.squeeze() for depNum in covidExtraToCom['dep']]
    covidExtraToCom['idx']  = self.max_normalize(covidExtraToCom['hospi'])

    maxriskmap = (0.1283*(0.029469479*self.max_normalize(dfpollution2["co"]).max()
        + 0.031129209*self.max_normalize(dfpollution2["o3"]).max()
        + 0.025763024*self.max_normalize(dfpollution2["pm25"]).max()\
        + 0.023641346*self.max_normalize(dfpollution2["no2"]).max() 
        + 0.021235887 * self.max_normalize(dfpollution2["pm10"]).max()\
        + 0.352199212*self.max_normalize(dfpollution2["1MMaxo3"]).max()
        + 0.027367485*self.max_normalize(dfpollution2["1MMaxpm10"]).max()
        + 0.025778019*self.max_normalize(dfpollution2["1MMaxno2"]).max()
        + 0.048095527*self.max_normalize(dfpollution2["1MMaxpm25"]).max()
        + 0.026836554*self.max_normalize(dfpollution2["1MMaxco"]).max()
        + 0.06857573*self.max_normalize(dfpollution2["o37davg"]).max()
        + 0.023853625*self.max_normalize(dfpollution2["pm107davg"]).max()
        + 0.031856764*self.max_normalize(dfpollution2["no27davg"]).max()
        + 0.026716503*self.max_normalize(dfpollution2["pm257davg"]).max()
        + 0.056620756*self.max_normalize(dfpollution2["co7davg"]).max()
        + 0.033909774*self.max_normalize(dfpollution2["o31Mavg"]).max()
        + 0.022787624*self.max_normalize(dfpollution2["pm101Mavg"]).max()
        + 0.022010391*self.max_normalize(dfpollution2["no21Mavg"]).max()
        + 0.021063666*self.max_normalize(dfpollution2["pm251Mavg"]).max()
        + 0.081089422*self.max_normalize(dfpollution2["co1Mavg"]).max())
        + 0.083866782*self.max_normalize(dfpollution2['idx']).max() 
        + 0.5060676372*self.max_normalize(dfpollution2['hospiprevday']).max()
        + 0.290423065*self.max_normalize(dfpollution2["covidpostestprevday"]).max()
        )

    print(maxriskmap)
    times = [("0 days",0),('1 days', 24),('2 days', 48),("3 days", 72),('4 days',96)]
    counter = 0
    images = []
    riskMaps = []
    for (j,i) in tqdm(times):
      covidExtraToCom['1MMaxpm25'] = [dfpollution[(dfpollution['numero'] == depNum) &( dfpollution['leadtime_hour']== i)]["1MMaxpm25"].values.squeeze() for depNum in covidExtraToCom['dep']]
      covidExtraToCom['1MMaxpm10'] = [dfpollution[(dfpollution['numero'] == depNum) &( dfpollution['leadtime_hour']== i)]["1MMaxpm10"].values.squeeze() for depNum in covidExtraToCom['dep']]
      covidExtraToCom['1MMaxo3'] = [dfpollution[(dfpollution['numero'] == depNum) &( dfpollution['leadtime_hour']== i)]["1MMaxo3"].values.squeeze() for depNum in covidExtraToCom['dep']]
      covidExtraToCom['1MMaxno2'] = [dfpollution[(dfpollution['numero'] == depNum) &( dfpollution['leadtime_hour']== i)]["1MMaxno2"].values.squeeze() for depNum in covidExtraToCom['dep']]
      covidExtraToCom['1MMaxco'] = [dfpollution[(dfpollution['numero'] == depNum) &( dfpollution['leadtime_hour']== i)]["1MMaxco"].values.squeeze() for depNum in covidExtraToCom['dep']]
      covidExtraToCom['pm107davg'] = [dfpollution[(dfpollution['numero'] == depNum) &( dfpollution['leadtime_hour']== i)]["pm107davg"].values.squeeze() for depNum in covidExtraToCom['dep']]
      covidExtraToCom['pm257davg'] = [dfpollution[(dfpollution['numero'] == depNum) &( dfpollution['leadtime_hour']== i)]["pm257davg"].values.squeeze() for depNum in covidExtraToCom['dep']]
      covidExtraToCom['o37davg'] = [dfpollution[(dfpollution['numero'] == depNum) &( dfpollution['leadtime_hour']== i)]["o37davg"].values.squeeze() for depNum in covidExtraToCom['dep']]
      covidExtraToCom['no27davg'] = [dfpollution[(dfpollution['numero'] == depNum) &( dfpollution['leadtime_hour']== i)]["no27davg"].values.squeeze() for depNum in covidExtraToCom['dep']]
      covidExtraToCom['co7davg'] = [dfpollution[(dfpollution['numero'] == depNum) &( dfpollution['leadtime_hour']== i)]["co7davg"].values.squeeze() for depNum in covidExtraToCom['dep']]
      covidExtraToCom['pm101Mavg'] = [dfpollution[(dfpollution['numero'] == depNum) &( dfpollution['leadtime_hour']== i)]["pm101Mavg"].values.squeeze() for depNum in covidExtraToCom['dep']]
      covidExtraToCom['pm251Mavg'] = [dfpollution[(dfpollution['numero'] == depNum) &( dfpollution['leadtime_hour']== i)]["pm251Mavg"].values.squeeze() for depNum in covidExtraToCom['dep']]
      covidExtraToCom['o31Mavg'] = [dfpollution[(dfpollution['numero'] == depNum) &( dfpollution['leadtime_hour']== i)]["o31Mavg"].values.squeeze() for depNum in covidExtraToCom['dep']]
      covidExtraToCom['no21Mavg'] = [dfpollution[(dfpollution['numero'] == depNum) &( dfpollution['leadtime_hour']== i)]["no21Mavg"].values.squeeze() for depNum in covidExtraToCom['dep']]
      covidExtraToCom['co1Mavg'] = [dfpollution[(dfpollution['numero'] == depNum) &( dfpollution['leadtime_hour']== i)]["co1Mavg"].values.squeeze() for depNum in covidExtraToCom['dep']]
      covidExtraToCom['population'] = [dfpollution[(dfpollution['numero'] == depNum) &( dfpollution['leadtime_hour']== i)]["idx"].values.squeeze() for depNum in covidExtraToCom['dep']]
      covidExtraToCom['hospiprevday'] = [dfpollution[(dfpollution['numero'] == depNum) &( dfpollution['leadtime_hour']== i)]["hospiprevday"].values.squeeze() for depNum in covidExtraToCom['dep']]
      covidExtraToCom['covidpostestprevday'] = [dfpollution[(dfpollution['numero'] == depNum) &( dfpollution['leadtime_hour']== i)]["covidpostestprevday"].values.squeeze() for depNum in covidExtraToCom['dep']]
      print('OK', flush=True)

      filePath = '../data/train/cams/fr/forecast/'
      latestfiledatestring = self.findlatestdateofcamsdata(filePath)[1].strftime('%Y-%m-%d')
      fileName = "cams-forecast-"+latestfiledatestring +".nc"
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
      risk = (0.1283*(0.029469479*self.max_normalize(coInterpolated)\
              + 0.031129209*self.max_normalize(o3Interpolated)\
              + 0.025763024 * self.max_normalize(pm25Interpolated)\
              + 0.023641346* self.max_normalize(no2Interpolated) \
              + 0.021235887 * self.max_normalize(pm10Interpolated) \
              + 0.352199212 * self.max_normalize(covidExtraToCom["1MMaxo3"])\
              + 0.027367485*self.max_normalize(covidExtraToCom["1MMaxpm10"])\
              + 0.025778019*self.max_normalize(covidExtraToCom["1MMaxno2"])\
              + 0.048095527*self.max_normalize(covidExtraToCom["1MMaxpm25"])\
              + 0.026836554*self.max_normalize(covidExtraToCom["1MMaxco"])\
              + 0.06857573*self.max_normalize(covidExtraToCom["o37davg"])\
              + 0.023853625*self.max_normalize(covidExtraToCom["pm107davg"])\
              + 0.031856764*self.max_normalize(covidExtraToCom["no27davg"])\
              + 0.026716503*self.max_normalize(covidExtraToCom["pm257davg"])\
              + 0.056620756*self.max_normalize(covidExtraToCom["co7davg"])\
              + 0.033909774*self.max_normalize(covidExtraToCom["o31Mavg"])\
              + 0.022787624*self.max_normalize(covidExtraToCom["pm101Mavg"])\
              + 0.022010391*self.max_normalize(covidExtraToCom["no21Mavg"])\
              + 0.021063666*self.max_normalize(covidExtraToCom["pm251Mavg"])\
              + 0.081089422*self.max_normalize(covidExtraToCom["co1Mavg"]))\
              + 0.083866782*self.max_normalize(covidExtraToCom['population'])\
              + 0.5060676372*self.max_normalize(covidExtraToCom['hospiprevday'])\
              + 0.290423065*self.max_normalize(covidExtraToCom["covidpostestprevday"]))\

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
      cmap='RdYlGn_r', s=markersize*5, vmin=0, vmax=maxriskmap, zorder=4)
      cbar = fig.colorbar(cax, orientation='horizontal', pad=0, aspect=50,
      fraction=.01, extend='max', drawedges=False, ticks=[0, maxriskmap])
      cbar.ax.set_xticklabels(['low', 'high'])
      cbar.ax.xaxis.set_ticks_position('top')
      cbar.ax.xaxis.set_label_position('top')

      ax1.text(0,.0,'Data \nCAMS \ndata.gouv.fr', transform=ax1.transAxes,fontdict={'size':12})

      currentDateWD = datetime.strptime(currentDate, '%Y-%m-%d').strftime('%a, %d %b %Y')
      ax1.set_title('Risk of severe Covid-19 cases 12.83% of which is explained by features \nengineered from atmospheric pollutants:\n{:}\n'.format(currentDateWD),
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
    kargs = { 'duration': .2 }
    imageio.mimwrite(gifPath + gifName, images, 'GIF', **kargs)
    print('OK')
    print('Finished.')

    return None

if __name__ == '__main__':
  RiskLevelMaps = compute_covid_risk_heat_map()
  RiskLevelMaps.compute_map()
