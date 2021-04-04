import time, glob
import shutil
import xarray as xr
import datetime

def without_shutil_text(outfilename):
    with open(outfilename, 'wb') as outfile:
        dates=[]
        for filename in glob.glob('data/pm25/*.nc'):
            datefromfile = filename.replace("pm25-","").replace(".nc","").replace('data/pm25/',"")
            print(datefromfile)
            dates.append(datefromfile)
            pm25 = xr.open_dataset(filename)
            pm25 = pm25.assign_coords(date= dates)
            pm25.to_netcdf(filename)
            if filename == outfilename:
                # don't want to copy the output into the output
                continue
            with open(filename, 'rb') as readfile:
                outfile.write(readfile.read())

without_shutil_text("pm25_history.nc")