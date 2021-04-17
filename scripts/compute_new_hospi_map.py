from cartopy import crs
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import regionmask
import seaborn as sns
import datetime
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

countryEU = regionmask.defined_regions.natural_earth.countries_50
sf = gpd.read_file('../departements.geojson')
df2 = pd.read_csv('../predictions/fr/predictions.csv')
sf2 = gpd.read_file('../departements.geojson')
df2["time"]=pd.to_datetime(df2["time"])
df3 = df2[df2["time"]==df2["time"].max()]
df3 = df3.sort_values(by = "depnum").reset_index()
sf2['code'] = sf['code'].replace({'2A':'201','2B':'202'}).astype(int)
sf3 = sf2.merge(df3, left_on = "code", right_on = "depnum",suffixes=('','_y'))
sf3 = sf3.merge(sf, on = "nom",suffixes=('_x','') )
sf3 = sf3.drop(columns = ["code_x", "geometry_x", "index","depnum","time"])
sf3['code'] = sf3['code'].replace({'2A':'201','2B':'202'}) 
fig, ax = plt.subplots(1, 0,figsize=(20, 10))
gs = fig.add_gridspec(1, 1)
ax = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
ax.background_patch.set_fill(False)
ax.add_geometries(countryEU['F'].polygon, ccrs.PlateCarree(),
edgecolor=grayDark, lw=2, facecolor=grayDark, alpha=0.6, zorder=0)
ax.set_extent([-5,10,41,52])
ax.set_aspect('auto')
ax.outline_patch.set_linewidth(0.)
currentDate = datetime.datetime.today().strftime('%Y-%m-%d')
currentDateWD = datetime.datetime.strptime(currentDate, '%Y-%m-%d').strftime('%a, %d %b %Y')
fig, ax = plt.subplots(1, 1,figsize=(20, 10))
ax.set_title('Forecasted number of severe Covid-19 cases leading to \nnew hospitalizations\n{:}\n'.format(currentDateWD), loc='left', pad=-60)
ax.text(0,.0,'Data \nCAMS \ndata.gouv.fr', transform=ax.transAxes,fontdict={'size':12})
sf3.plot(column='newhospipred', ax=ax, legend=True,cmap='RdYlGn_r')