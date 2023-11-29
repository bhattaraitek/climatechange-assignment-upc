import pathlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import calendar

MY_ROOT = pathlib.Path('My_Location_Data')
CLIMATE_VARIABLES = ['pr','tas']
SCENARIOS = ["historical","rcp85"]
MODELS = ["MOHC-HadGEM2-ES","MPI-M-MPI-ESM-LR"]

#Reading model data
def define_var(var = 'pr'):
    return var
    
for scenario in range(1):
    daily_raw_all_model = pd.DataFrame()
    for model in range(2):
        file_name = f"{define_var()}_{MODELS[model]}_{SCENARIOS[scenario]}.txt"
        daily_raw = pd.DataFrame(pd.read_csv(pathlib.Path(MY_ROOT/file_name)))
        daily_raw.columns = ['Date',f"{define_var()}"]
        daily_raw['Scenario'] = SCENARIOS[scenario]
        daily_raw['Model'] = MODELS[model]
        daily_raw_all_model = pd.concat([daily_raw_all_model,daily_raw],axis=0)

daily_raw_all_model['Date'] = pd.DatetimeIndex(daily_raw_all_model['Date'])
daily_raw_all_model['Date'] = pd.to_datetime(daily_raw_all_model['Date']).dt.normalize()
daily_raw_all_model['Year'] = pd.DatetimeIndex(daily_raw_all_model['Date']).year
daily_raw_all_model['Month'] = pd.DatetimeIndex(daily_raw_all_model['Date']).month

#Reading and preprocessing observed data
if define_var() == 'pr':
    '''Rainfall Data '''
    observed_pr = pd.DataFrame(pd.read_csv(f"{MY_ROOT}/pr_observed.csv"))
    observed_pr['Date'] = pd.DatetimeIndex(observed_pr['Date'])
    observed_pr['Year'] = pd.DatetimeIndex(observed_pr['Date']).year
    observed_pr['Month'] = pd.DatetimeIndex(observed_pr['Date']).month
else:
    '''Temperature Data'''
    observed_tas = pd.read_csv(f"{MY_ROOT}/tas_observed.csv")
    observed_tas['Date'] = pd.DatetimeIndex(observed_tas['Date'])
    observed_tas['Year'] = pd.DatetimeIndex(observed_tas['Date']).year
    observed_tas['Month'] = pd.DatetimeIndex(observed_tas['Date']).month

if define_var() == "pr":
    '''Grouping Rainfall Data by Year and Month for all historical period'''
    grouped_by_year_month_model_pr = daily_raw_all_model.groupby(by=['Year','Month','Model'])['pr'].sum().reset_index()
    grouped_by_year_month_model_pr['Year-Month'] = grouped_by_year_month_model_pr['Year'].astype(str)+'-'+grouped_by_year_month_model_pr['Month'].astype(str)
    #grouped_by_year_model_pr =grouped_by_year_month_model_pr.groupby(by = ['Year','Model'])['pr'].mean().reset_index()

    grouped_by_year_month_observed_pr = observed_pr.groupby(by=['Year','Month'])['s406'].sum().reset_index()
    grouped_by_year_month_observed_pr['Year-Month'] = grouped_by_year_month_observed_pr['Year'].astype(str)+'-'+grouped_by_year_month_observed_pr['Month'].astype(str)

    '''Grouping Rainfall Data by Year for all historical period'''
    grouped_by_year_model_pr = daily_raw_all_model.groupby(by=['Year','Model'])['pr'].sum().reset_index()
    grouped_by_month_model_pr = grouped_by_year_month_model_pr.groupby(by = ['Month','Model'])['pr'].mean().reset_index()
    long_term_average_model_pr = grouped_by_year_model_pr.groupby(by='Model')['pr'].mean().reset_index()

    grouped_by_year_obs_pr = observed_pr.groupby(by='Year')['s406'].sum().reset_index()
    grouped_by_year_obs_pr = grouped_by_year_obs_pr[(grouped_by_year_obs_pr['Year']>=1970) & (grouped_by_year_obs_pr['Year']<=2005)]
    grouped_by_month_obs_pr = grouped_by_year_month_observed_pr.groupby(by =['Month'])['s406'].mean().reset_index()
    grouped_by_year_obs_pr = pd.concat([grouped_by_year_obs_pr,grouped_by_year_obs_pr],ignore_index=True)
    long_term_average_obs_pr = grouped_by_year_obs_pr['s406'].mean()
else:
    '''Grouping Temperature Data by Year and Month for all historical period'''
    grouped_by_year_month_model_tas = daily_raw_all_model.groupby(by=['Year','Month','Model'])['tas'].mean().reset_index()
    grouped_by_year_month_model_tas['Year-Month'] = grouped_by_year_month_model_tas['Year'].astype(str)+'-'+grouped_by_year_month_model_tas['Month'].astype(str)
    
    grouped_by_year_month_observed_tas = observed_tas.groupby(by=['Year','Month'])['s406'].mean().reset_index()
    grouped_by_year_month_observed_tas['Year-Month'] = grouped_by_year_month_observed_tas['Year'].astype(str)+'-'+grouped_by_year_month_observed_tas['Month'].astype(str)
    grouped_by_month_observed_tas = grouped_by_year_month_observed_tas.groupby(by = 'Month')['s406'].mean().reset_index()


    '''Grouping Temperature Data by Year for all historical period'''
    grouped_by_year_model_tas = daily_raw_all_model.groupby(by=['Year','Model'])['tas'].mean().reset_index()
    long_term_average_model_tas = grouped_by_year_model_tas.groupby(by='Model')['tas'].mean().reset_index()
    grouped_by_month_model_tas = grouped_by_year_month_model_tas.groupby(by=['Month','Model'])['tas'].mean().reset_index()

    grouped_by_year_obs_tas = observed_tas.groupby(by='Year')['s406'].mean().reset_index()
    grouped_by_year_obs_tas = grouped_by_year_obs_tas[(grouped_by_year_obs_tas['Year']>=1970) & (grouped_by_year_obs_tas['Year']<=2005)]
    grouped_by_year_obs_tas = pd.concat([grouped_by_year_obs_tas,grouped_by_year_obs_tas],ignore_index=True)
    long_term_average_obs_tas = grouped_by_year_obs_tas['s406'].mean()

#Daily raiinfall hyetograph
def rainfall_hyetograph(model, observed= None):
    '''Args: Dataframe with daily data'''
    pivot_df = model.pivot(index='Date',columns='Model',values='pr')
    fig,ax = plt.subplots(figsize = (10,4))
    if observed is not None:
        ax.plot(np.array(observed['Date']),np.array(observed['s406']),color = 'cyan',label = 'observed')
    ax.plot(np.array(pivot_df.index),np.array(pivot_df['MPI-M-MPI-ESM-LR']), color ='blue',label = 'MPI-M-MPI-ESM-LR')
    ax.plot(np.array(pivot_df.index),np.array(pivot_df['MOHC-HadGEM2-ES']), color ='red',label = 'MOHC-HadGEM2-ES')
    ax.legend()
    plt.xlabel('Date',fontweight ='bold',color = 'black',fontsize =14)
    plt.ylabel('Rainfall(mm)',fontweight = 'bold',color = 'black',fontsize =14)
    plt.xticks(fontsize = 12,color = 'black')
    plt.yticks(fontsize = 12, color = 'black')
    plt.grid(False)


def plot_year_month_rainfall(model,observed= None):
    order = sorted(model['Year-Month'].unique())
    fig = plt.figure(figsize=(16,4))
    if observed is not None:
        sns.lineplot(observed, x = 'Year-Month',y= 's406',color = 'grey', label = "Observed")
    sns.barplot(model,x='Year-Month',y='pr',hue='Model',order = order, palette= ['green','blue'],dodge=True, linewidth=.1, edgecolor='0.1')
    plt.xticks(rotation=45, ha='right')
    plt.gca().set_xticks(plt.gca().get_xticks()[::30])
    plt.xlabel('Year-Month',fontweight ='bold',color = 'black',fontsize = 14)
    plt.ylabel('Accumulated rainfall (mm)',fontweight ='bold',color = 'black',fontsize = 14)
    plt.xticks(fontsize = 12,color = 'black')
    plt.yticks(fontsize = 12, color = 'black')
    plt.legend()


def plot_average_annual_rainfall(model,observed = None):
    yearly_plot,ax1 = plt.subplots(figsize=(10,4))
    if observed is not None:
        sns.lineplot(observed['s406'],color = 'grey',marker='*',markersize=10, label='Observed',ax=ax1)
        ax1.axhline(y=long_term_average_obs_pr,linestyle ="--",color = 'grey')
    
    sns.barplot(model,x='Year',y='pr',hue='Model',palette=['green','blue'], ax=ax1)
    ax1.axhline(y= long_term_average_model_pr.iloc[0][1],linestyle ='--',color = 'green')
    ax1.axhline(y= long_term_average_model_pr.iloc[1][1],linestyle ="--",color = 'blue')

    plt.gca().set_xticks(plt.gca().get_xticks()[::5])
    ax1.legend(loc='lower right')
    ax1.set_xlabel('Year',fontweight = 'bold',color = 'black',fontsize =14)
    ax1.set_ylabel('Annual rainfall (mm)',fontweight = 'bold',color = 'black',fontsize =14)
    plt.xticks(fontsize = 12,color = 'black')
    plt.yticks(fontsize = 12, color = 'black')
    

def plot_average_pr_monthly(model,observed=None):
    fig1,ax1 = plt.subplots(figsize=(10,4))
    if observed is not None:
        sns.lineplot(observed['s406'],color = 'grey',marker = '*', markersize = 10,label = 'Observed', ax = ax1)
    sns.barplot(model,x='Month',y = 'pr',hue = 'Model',palette=['green','blue'],ax=ax1)
    plt.legend(loc='upper right')
    plt.xlabel('Month',fontweight ='bold',color = 'black')
    plt.ylabel('Average Rainfall (mm)',fontweight = 'bold',color = 'black')
    plt.xticks(fontsize = 12,color = 'black')
    plt.yticks(fontsize = 12, color = 'black')

# For temperature plots
def plot_daily_tas(model,observed = None):
    daily_avg_plot,ax1 = plt.subplots(figsize = (10,4))
    sns.lineplot(model,x='Date', y='tas',palette=['orange','red'],hue = 'Model',linewidth = 0.25,ax=ax1)
    if observed is not None:
        sns.lineplot(observed['s406'],color='grey',linewidth = 0.25,ax = ax1)
    plt.xlabel('Date',fontweight = "bold", fontsize = 14, color = 'black')
    plt.ylabel('Avg. Surface Temperature (°C)',fontweight = "bold" ,fontsize = 14, color = 'black')
    plt.xticks(fontsize = 12,color = 'black')
    plt.yticks(fontsize = 12, color = 'black')
    plt.legend(loc = "upper right")

def plot_year_month_tas_series(model, observed= None):
    fig = plt.figure(figsize=(16,4))
    if observed is not None:
        sns.lineplot(observed, y = 'tas',color = 'grey')
    sns.lineplot(model,x='Year-Month',y='tas',hue='Model')
    plt.gca().set_xticks(plt.gca().get_xticks()[::24])
    plt.xticks(rotation = 45, ha = 'right')
    plt.legend(loc='upper right')
    plt.xlabel('Year-Month',fontweight ='bold',fontsize = 14,color = 'black')
    plt.ylabel('Avg. Surface Temperature (°C)',fontweight = 'bold',fontsize = 14,color ='black')
    plt.xticks(fontsize = 12,color = 'black')
    plt.yticks(fontsize = 12, color = 'black')

def plt_lt_monthly_avg_tas(model,observed=None):
    fig1,ax1 = plt.subplots(figsize=(8,5))
    if observed is not None:
        sns.lineplot(observed['s406'],color = 'grey',marker = '*', markersize = 10,label = 'Observed', ax = ax1)
    sns.barplot(model, x='Month',y = 'tas',hue = 'Model',palette=['orange','red'],ax=ax1)
    plt.legend(loc='upper right')
    plt.xlabel('Month',fontweight ='bold')
    plt.ylabel('Avg. Surface Temperature (°C)',fontweight = 'bold')

def plot_avg_annual_tas(model, observed=None):
    fig = plt.figure(figsize=(10,4))
    if observed is not None:
        sns.lineplot(observed, x = 'Year',y = 's406',color = 'grey',marker = '*',markersize = 8,label = 'Observed')
    sns.lineplot(model, x = 'Year',y = 'tas',hue='Model',marker = '*',markersize = 8,palette=['orange','red'])
    plt.legend(loc = 'upper center',ncol = 3,frameon = False)
    plt.xlabel('Year',fontweight = 'bold',fontsize = 13, color = 'black')
    plt.ylabel('Avg. Surface Temperature (°C)',fontweight = 'bold',fontsize = 13, color = 'black')
    plt.axhline(y=long_term_average_obs_tas,linestyle = '--',color = 'grey')
    plt.axhline(y=long_term_average_model_tas.iloc[0][1],linestyle = '--',color = 'orange',alpha = 0.6)
    plt.axhline(y=long_term_average_model_tas.iloc[1][1],linestyle ='--',color = 'red',alpha = 0.6)
    plt.xticks(fontsize = 12,color = 'black')
    plt.yticks(fontsize = 12, color = 'black')

    

