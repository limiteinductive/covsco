from cartopy import crs
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import regionmask
import seaborn as sns
import datetime
from tqdm import tqdm
import regionmask
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import io
import imageio

class compute_new_hospi_map:
	def __init__(self):
		self.status = None

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
				#,'figure.subplot.left' 		: 0.1 # 0.125
				#,'figure.subplot.right' 	: 0.95 # 0.9
				#,'figure.subplot.top' 		: 0.90 # 0.88
				#,'figure.subplot.bottom' 	: 0.11 # 0.11
				#,'figure.subplot.wspace' 	: 0.2 # 0.2
				#,'figure.subplot.hspace' 	: 0.2 # 0.2
				#,'text.usetex':True
				#,'text.latex.preamble':[
				#	r'\usepackage{cmbright}', 	# set font; for Helvetica use r'\usepackage{helvet}'
				#	r'\usepackage{relsize}',
				#	r'\usepackage{upgreek}',
				#	r'\usepackage{amsmath}'
				#	r'\usepackage{siunitx}',
				#	r'\usepackage{physics}',
				#	r'\usepackage{wasysym}', 	# permil symbol in mathmode: \permil
				#	r'\usepackage{textcomp}', 	# permil symbol: \textperthousand
				#	r'\usepackage{mathtools}',
				#	r'\setlength\parindent{0pt}'
				#	]
				}
		)


		countryBorders = cfeature.NaturalEarthFeature(
			category    = 'cultural',
			name        = 'ne_admin_0_countries',
			scale       = '10m',
			facecolor   = 'none'
		)
		
		dfpollution3 = pd.read_csv("/home/ludo915/code/covsco/data/train/all_data_merged/fr/traindfknnimputed.csv")
		currentDate = pd.to_datetime(dfpollution3["date"].max()) + pd.Timedelta("1 Days")
		countryEU = regionmask.defined_regions.natural_earth.countries_50
		times = ["0 days","1 days","2 days", "3 days",'4 days']
		counter = 0

		currentDatestring = currentDate.strftime('%Y-%m-%d')
		currentDateWD = pd.to_datetime(dfpollution3["date"].max()).strftime('%a, %d %b %Y')
		sf = gpd.read_file('/home/ludo915/code/covsco/departements.geojson')
		images = []
		counter = 0
		for j in tqdm(times):
			print(j)
			filename = "/home/ludo915/code/covsco/predictions/fr/" + currentDatestring + "_predictions_for_day_" + str(counter) +".csv"
			df2 = pd.read_csv(filename)
			print(filename + " Read!")
			sf2 = gpd.read_file('/home/ludo915/code/covsco/departements.geojson')
			df2["date"]=pd.to_datetime(df2["date"])
			df3 = df2[df2["date"]==df2["date"].max()]
			df3 = df3.sort_values(by = "depnum").reset_index()
			sf2['code'] = sf['code'].replace({'2A':'201','2B':'202'}).astype(int)
			sf3 = sf2.merge(df3, left_on = "code", right_on = "depnum",suffixes=('','_y'))
			sf3 = sf3.merge(sf, on = "nom",suffixes=('_x','') )
			sf3 = sf3.drop(columns = ["code_x", "geometry_x", "index","depnum","date"])
			sf3['code'] = sf3['code'].replace({'2A':'201','2B':'202'}) 	
			
			fig, ax = plt.subplots(figsize = (15,15))
			gs = fig.add_gridspec(1, 1)
			ax = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
			ax.axis("off")
			ax.background_patch.set_fill(False)
			ax.add_geometries(countryEU['F'].polygon, ccrs.PlateCarree(),
			edgecolor=grayDark, lw=2, facecolor=grayDark, alpha=0.6, zorder=0)
			ax.set_extent([-5,10,41,52])
			ax.set_aspect('auto')
			ax.outline_patch.set_linewidth(0.)
			ax.set_title(('Predictions of severe Covid-19 cases leading to new hospitalizations\n{:}\n Day ' + str(counter)).format(currentDateWD), loc='left', pad=-60)
			ax.text(0,.0,'Data \nCAMS \ndata.gouv.fr', transform=ax.transAxes,fontdict={'size':12})
			sf3.plot(column='newhospipred', ax=ax, legend=True,cmap='RdYlGn_r', vmin = 0, vmax = 120)
			fig.subplots_adjust(bottom=.01, left=.01, right=.99, top=.99)
			plt.show()
			buffer = io.BytesIO()
			plt.savefig(buffer, format='png', dpi=70, pad_inches = 0)
			buffer.seek(0)
			images.append(imageio.imread(buffer))
			buffer.close()
			plt.close()
			counter += 1
		
		print('Create gif ...', flush=True, end='')
		gifPath = '/home/ludo915/code/covsco/forecast/fr/'
		gifName = 'newhospidepartementlevel-{:}.gif'.format(currentDatestring)
		kargs = { 'duration': 1 }
		imageio.mimwrite(gifPath + gifName, images, 'GIF', **kargs)

		print('OK')
		print('Finished.')

		return None

if __name__ == '__main__':
	NewHospiMap = compute_new_hospi_map()
	NewHospiMap.compute_map()

		
