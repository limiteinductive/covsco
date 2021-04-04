# README

## Prepare
Create a correct environment e.g. using `conda`. We need quite specific versions, e.g. python 3.7 and some other specific versions of some packages. So save yourself some pain and do it the quick way ;)
```
conda create --name myenv --file package-list.txt
```
Make shell script executable variance
```
chmod +x update_data.sh
```

## Run analysis
```
./update_data.sh
```
## Download Historical Data
python download_covid_hist_data.py
python download_cams_reanalysis.py
python downlaod_era5_reanalysis.py

## Merge the data in one CSV for training the ML algorithm
python maintraindata.py

## Train the model/Search for the best model with the T-POT optimizer +exporting the tpot_covid_pipeline.py file + generating Feature importance reports
python maintrain.py

# Download Daily Data
download_covid_daily_data.py