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

class compute_maps:

  def __init__(self):

    self.status = None
    self.data = None
  
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

  def compute_maps(self):

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
    dfpollution3 = pd.read_csv("../data/train/all_data_merged/fr/traindfknnimputed.csv")
   
    dfpollution= dfpollution.dropna()
    dfpollution2 = dfpollution2.dropna()
    
    currentDate = pd.to_datetime(dfpollution3["date"].max()) + pd.Timedelta("1 Days")
    dfpollution=dfpollution[dfpollution["date"]==dfpollution["date"].max()]
    dfpollution3=dfpollution3[dfpollution3["date"]==dfpollution3["date"].max()]
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
    images1 = []
    images2 = []
    images3 = []
    images4 = []
    images5 = []
    images6 = []
    images7 = []

    risk1Maps = []
    risk2Maps = []
    risk3Maps = []
    risk4Maps = []
    risk5Maps = []
    risk6Maps = []
    newhospipredmaps = []
    currentDatestring = currentDate.strftime('%Y-%m-%d')

      # covidExtraToCom[['1MMaxpm25','1MMaxpm10','1MMaxo3','1MMaxno2','1MMaxco','pm107davg','pm257davg','o37davg','no27davg','co7davg','pm101Mavg',\
      #   'pm251Mavg','o31Mavg','no21Mavg','co1Mavg','population','hospi','CovidPosTest' ]] \
      #     = [dfpollution3[dfpollution3['numero'] == depNum].reindex(columns = ['1MMaxpm25','1MMaxpm10','1MMaxo3','1MMaxno2','1MMaxco','pm107davg','pm257davg','o37davg','no27davg','co7davg','pm101Mavg',\
      #   'pm251Mavg','o31Mavg','no21Mavg','co1Mavg','idx','hospi','CovidPosTest' ]).values.squeeze() for depNum in covidExtraToCom['dep']]

    for j in tqdm(times):
        print(j)
        filename = "../predictions/fr/" + currentDatestring + "_predictions_for_day_" + str(counter) +".csv"
        newhospipredictionsdf = pd.read_csv(filename)
        print(filename + " Read!")

        print("Interpolating newhospi predictions to commune level...")
        covidExtraToCom['newhospipred'] = [newhospipredictionsdf[newhospipredictionsdf['depnum'] == depNum]["newhospipred"].values.squeeze() for depNum in covidExtraToCom['dep']]
        print("newhospipred interpolated")
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
        print("Computing Atmospheric Pollutants maps ...")
        risk1 = pm25Interpolated
        risk2 = coInterpolated
        risk3 = o3Interpolated
        risk4 = no2Interpolated
        risk5 = pm10Interpolated
        risk6 = so2Interpolated
        newhospipredmap = covidExtraToCom['newhospipred']

        risk1 = np.array(risk1)
        risk1 = np.vstack((lons,lats,risk1))
        risk1 = risk1.T
        risk1 = pd.DataFrame(risk1, columns = ['lon', 'lat', 'idx'])
        risk1max = dfpollution2['pm25'].max()

        risk2 = np.array(risk2)
        risk2 = np.vstack((lons,lats,risk2))
        risk2 = risk2.T
        risk2 = pd.DataFrame(risk2, columns = ['lon', 'lat', 'idx'])
        risk2max = dfpollution2['co'].max()
    

        risk3 = np.array(risk3)
        risk3 = np.vstack((lons,lats,risk3))
        risk3 = risk3.T
        risk3 = pd.DataFrame(risk3, columns = ['lon', 'lat', 'idx'])
        risk3max = dfpollution2['o3'].max()

        risk4 = np.array(risk4)
        risk4 = np.vstack((lons,lats,risk4))
        risk4 = risk4.T
        risk4 = pd.DataFrame(risk4, columns = ['lon', 'lat', 'idx'])
        risk4max = dfpollution2['no2'].max()

        risk5 = np.array(risk5)
        risk5 = np.vstack((lons,lats,risk5))
        risk5 = risk5.T
        risk5 = pd.DataFrame(risk5, columns = ['lon', 'lat', 'idx'])
        risk5max = dfpollution2['pm10'].max()

        risk6 = np.array(risk6)
        risk6 = np.vstack((lons,lats,risk6))
        risk6 = risk6.T
        risk6 = pd.DataFrame(risk6, columns = ['lon', 'lat', 'idx'])
        risk6max = dfpollution2['so2'].max()

        newhospipredmap = np.array(newhospipredmap)
        newhospipredmap = np.vstack((lons,lats,newhospipredmap))
        newhospipredmap = newhospipredmap.T
        newhospipredmap = pd.DataFrame(newhospipredmap, columns = ['lon', 'lat', 'idx'])
        newhospipredmax = dfpollution2['newhospi'].max()

        risk1Maps.append(risk1)
        risk2Maps.append(risk2)
        risk3Maps.append(risk3)
        risk4Maps.append(risk4)
        risk5Maps.append(risk5)
        risk6Maps.append(risk6)
        newhospipredmaps.append(newhospipredmap)

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

        cax = ax1.scatter(risk1Maps[counter].lon,risk1Maps[counter].lat,c=risk1Maps[counter].idx,
        cmap='RdYlGn_r', s=markersize*5, vmin=0, vmax=risk1max, zorder=4)
        cbar = fig.colorbar(cax, orientation='horizontal', pad=0, aspect=50,
        fraction=.01, extend='max', drawedges=False, ticks=[0, risk1max])
        cbar.ax.set_xticklabels(['low', 'high'])
        cbar.ax.xaxis.set_ticks_position('top')
        cbar.ax.xaxis.set_label_position('top')

        ax1.text(0,.0,'Data \nCAMS \ndata.gouv.fr', transform=ax1.transAxes,fontdict={'size':12})

        currentDateWD = pd.to_datetime(dfpollution3["date"].max()).strftime('%a, %d %b %Y')
        ax1.set_title('PM2.5 concentrations: \n{:}\n'.format(currentDateWD + " + "+ str (counter) + " days"),
        loc='left', pad=-60)

        fig.subplots_adjust(bottom=.01, left=.01, right=.99, top=.99)
        plt.show()
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=70)
        buffer.seek(0)
        images1.append(imageio.imread(buffer))
        buffer.close()
        plt.close()

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

        cax = ax1.scatter(risk2Maps[counter].lon,risk2Maps[counter].lat,c=risk2Maps[counter].idx,
        cmap='RdYlGn_r', s=markersize*5, vmin=0, vmax=risk2max, zorder=4)
        cbar = fig.colorbar(cax, orientation='horizontal', pad=0, aspect=50,
        fraction=.01, extend='max', drawedges=False, ticks=[0, risk2max])
        cbar.ax.set_xticklabels(['low', 'high'])
        cbar.ax.xaxis.set_ticks_position('top')
        cbar.ax.xaxis.set_label_position('top')

        ax1.text(0,.0,'Data \nCAMS \ndata.gouv.fr', transform=ax1.transAxes,fontdict={'size':12})

        #currentDateWD = datetime.strptime(str(dfpollution3["date"].max())).strftime('%a, %d %b %Y')
        currentDateWD = pd.to_datetime(dfpollution3["date"].max()).strftime('%a, %d %b %Y')
        ax1.set_title('CO concentrations: \n{:}\n'.format(currentDateWD + " + "+ str (counter) + " days"),
        loc='left', pad=-60)

        fig.subplots_adjust(bottom=.01, left=.01, right=.99, top=.99)
        plt.show()
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=70)
        buffer.seek(0)
        images2.append(imageio.imread(buffer))
        buffer.close()
        plt.close()

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

        cax = ax1.scatter(risk3Maps[counter].lon,risk3Maps[counter].lat,c=risk3Maps[counter].idx,
        cmap='RdYlGn_r', s=markersize*5, vmin=0, vmax=risk3max, zorder=4)
        cbar = fig.colorbar(cax, orientation='horizontal', pad=0, aspect=50,
        fraction=.01, extend='max', drawedges=False, ticks=[0, risk3max])
        cbar.ax.set_xticklabels(['low', 'high'])
        cbar.ax.xaxis.set_ticks_position('top')
        cbar.ax.xaxis.set_label_position('top')

        ax1.text(0,.0,'Data \nCAMS \ndata.gouv.fr', transform=ax1.transAxes,fontdict={'size':12})

        currentDateWD = pd.to_datetime(dfpollution3["date"].max()).strftime('%a, %d %b %Y')
        ax1.set_title('O3 concentrations: \n{:}\n'.format(currentDateWD + " + "+ str (counter) + " days"),
        loc='left', pad=-60)

        fig.subplots_adjust(bottom=.01, left=.01, right=.99, top=.99)
        plt.show()
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=70)
        buffer.seek(0)
        images3.append(imageio.imread(buffer))
        buffer.close()
        plt.close()

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

        cax = ax1.scatter(risk4Maps[counter].lon,risk4Maps[counter].lat,c=risk4Maps[counter].idx,
        cmap='RdYlGn_r', s=markersize*5, vmin=0, vmax=risk4max, zorder=4)
        cbar = fig.colorbar(cax, orientation='horizontal', pad=0, aspect=50,
        fraction=.01, extend='max', drawedges=False, ticks=[0, risk4max])
        cbar.ax.set_xticklabels(['low', 'high'])
        cbar.ax.xaxis.set_ticks_position('top')
        cbar.ax.xaxis.set_label_position('top')

        ax1.text(0,.0,'Data \nCAMS \ndata.gouv.fr', transform=ax1.transAxes,fontdict={'size':12})

        currentDateWD = pd.to_datetime(dfpollution3["date"].max()).strftime('%a, %d %b %Y')
        ax1.set_title('NO2 concentrations: \n{:}\n'.format(currentDateWD + " + "+ str (counter) + " days"),
        loc='left', pad=-60)

        fig.subplots_adjust(bottom=.01, left=.01, right=.99, top=.99)
        plt.show()
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=70)
        buffer.seek(0)
        images4.append(imageio.imread(buffer))
        buffer.close()
        plt.close()

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

        cax = ax1.scatter(risk5Maps[counter].lon,risk5Maps[counter].lat,c=risk5Maps[counter].idx,
        cmap='RdYlGn_r', s=markersize*5, vmin=0, vmax=risk5max, zorder=4)
        cbar = fig.colorbar(cax, orientation='horizontal', pad=0, aspect=50,
        fraction=.01, extend='max', drawedges=False, ticks=[0, risk5max])
        cbar.ax.set_xticklabels(['low', 'high'])
        cbar.ax.xaxis.set_ticks_position('top')
        cbar.ax.xaxis.set_label_position('top')

        ax1.text(0,.0,'Data \nCAMS \ndata.gouv.fr', transform=ax1.transAxes,fontdict={'size':12})

        currentDateWD = pd.to_datetime(dfpollution3["date"].max()).strftime('%a, %d %b %Y')
        ax1.set_title('PM10 concentrations: \n{:}\n'.format(currentDateWD + " + "+ str (counter) + " days"),
        loc='left', pad=-60)

        fig.subplots_adjust(bottom=.01, left=.01, right=.99, top=.99)
        plt.show()
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=70)
        buffer.seek(0)
        images5.append(imageio.imread(buffer))
        buffer.close()
        plt.close()

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

        cax = ax1.scatter(risk6Maps[counter].lon,risk6Maps[counter].lat,c=risk6Maps[counter].idx,
        cmap='RdYlGn_r', s=markersize*5, vmin=0, vmax=risk6max, zorder=4)
        cbar = fig.colorbar(cax, orientation='horizontal', pad=0, aspect=50,
        fraction=.01, extend='max', drawedges=False, ticks=[0, risk6max])
        cbar.ax.set_xticklabels(['low', 'high'])
        cbar.ax.xaxis.set_ticks_position('top')
        cbar.ax.xaxis.set_label_position('top')

        ax1.text(0,.0,'Data \nCAMS \ndata.gouv.fr', transform=ax1.transAxes,fontdict={'size':12})

        currentDateWD = pd.to_datetime(dfpollution3["date"].max()).strftime('%a, %d %b %Y')
        ax1.set_title('SO2 concentrations: \n{:}\n'.format(currentDateWD + " + "+ str (counter) + " days"),
        loc='left', pad=-60)

        fig.subplots_adjust(bottom=.01, left=.01, right=.99, top=.99)
        plt.show()
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=70)
        buffer.seek(0)
        images6.append(imageio.imread(buffer))
        buffer.close()
        plt.close()

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

        cax = ax1.scatter(newhospipredmaps[counter].lon,newhospipredmaps[counter].lat,c=newhospipredmaps[counter].idx,
        cmap='RdYlGn_r', s=markersize*5, vmin=0, vmax=newhospipredmax, zorder=4)
        cbar = fig.colorbar(cax, orientation='horizontal', pad=0, aspect=50,
        fraction=.01, extend='max', drawedges=False, ticks=[0, newhospipredmax])
        cbar.ax.set_xticklabels(['low', 'high'])
        cbar.ax.xaxis.set_ticks_position('top')
        cbar.ax.xaxis.set_label_position('top')

        ax1.text(0,.0,'Data \nCAMS \ndata.gouv.fr', transform=ax1.transAxes,fontdict={'size':12})

        currentDateWD = pd.to_datetime(dfpollution3["date"].max()).strftime('%a, %d %b %Y')
        ax1.set_title('Predictions of severe Covid19 cases leading to hospitalizations: \n{:}\n'.format(currentDateWD + " + "+ str (counter) + " days"),
        loc='left', pad=-60)

        fig.subplots_adjust(bottom=.01, left=.01, right=.99, top=.99)
        plt.show()
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=70)
        buffer.seek(0)
        images7.append(imageio.imread(buffer))
        buffer.close()
        plt.close()
        counter += 1

    print('Create gif ...', flush=True, end='')
    gifPath = '../forecast/fr/'
    gifName = 'PM2.5-concentration-{:}.gif'.format(currentDatestring)
    kargs = { 'duration': 1 }
    imageio.mimwrite(gifPath + gifName, images1, 'GIF', **kargs)

    print('Create gif ...', flush=True, end='')
    gifPath = '../forecast/fr/'
    gifName = 'CO-concentration-{:}.gif'.format(currentDatestring)
    kargs = { 'duration': 1 }
    imageio.mimwrite(gifPath + gifName, images2, 'GIF', **kargs)

    print('Create gif ...', flush=True, end='')
    gifPath = '../forecast/fr/'
    gifName = 'O3-concentration-{:}.gif'.format(currentDatestring)
    kargs = { 'duration': 1 }
    imageio.mimwrite(gifPath + gifName, images3, 'GIF', **kargs)

    print('Create gif ...', flush=True, end='')
    gifPath = '../forecast/fr/'
    gifName = 'NO2-concentration-{:}.gif'.format(currentDatestring)
    kargs = { 'duration': 1 }
    imageio.mimwrite(gifPath + gifName, images4, 'GIF', **kargs)

    print('Create gif ...', flush=True, end='')
    gifPath = '../forecast/fr/'
    gifName = 'PM10-concentration-{:}.gif'.format(currentDatestring)
    kargs = { 'duration': 1 }
    imageio.mimwrite(gifPath + gifName, images5, 'GIF', **kargs)
    
    print('Create gif ...', flush=True, end='')
    gifPath = '../forecast/fr/'
    gifName = 'SO2-concentration-{:}.gif'.format(currentDatestring)
    kargs = { 'duration': 1 }
    imageio.mimwrite(gifPath + gifName, images6, 'GIF', **kargs)

    print('Create gif ...', flush=True, end='')
    gifPath = '../forecast/fr/'
    gifName = 'newhospi-{:}.gif'.format(currentDatestring)
    kargs = { 'duration': 1 }
    imageio.mimwrite(gifPath + gifName, images7, 'GIF', **kargs)

    print('OK')
    print('Finished.')

    return None

if __name__ == '__main__':
  Maps = compute_maps()
  Maps.compute_maps()
