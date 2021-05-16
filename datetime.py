# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 18:39:55 2020

@author: Balaji
"""


def create_date_featues(df):

    df['Year'] = pd.to_datetime(df['DateTime']).dt.year

    df['Month'] = pd.to_datetime(df['DateTime']).dt.month

    df['Day'] = pd.to_datetime(df['DateTime']).dt.day

    df['Dayofweek'] = pd.to_datetime(df['DateTime']).dt.dayofweek

    df['DayOfyear'] = pd.to_datetime(df['DateTime']).dt.dayofyear

    df['Week'] = pd.to_datetime(df['DateTime']).dt.week

    df['Quarter'] = pd.to_datetime(df['DateTime']).dt.quarter 

    df['Is_month_start'] = pd.to_datetime(df['DateTime']).dt.is_month_start

    df['Is_month_end'] = pd.to_datetime(df['DateTime']).dt.is_month_end

    df['Is_quarter_start'] = pd.to_datetime(df['DateTime']).dt.is_quarter_start

    df['Is_quarter_end'] = pd.to_datetime(df['DateTime']).dt.is_quarter_end

    df['Is_year_start'] = pd.to_datetime(df['DateTime']).dt.is_year_start

    df['Is_year_end'] = pd.to_datetime(df['DateTime']).dt.is_year_end

    df['Semester'] = np.where(df['Quarter'].isin([1,2]),1,2)

    df['Is_weekend'] = np.where(df['Dayofweek'].isin([5,6]),1,0)

    df['Is_weekday'] = np.where(df['Dayofweek'].isin([0,1,2,3,4]),1,0)

    #df['Days_in_month'] = pd.to_datetime(df['DateTime']).dt.days_in_month
    
    df['Hour'] = pd.to_datetime(df['DateTime']).dt.hour
    
    #df['Time'] = [((date.hour*60+(date.minute))*60)+date.second for date in df.DateTime]

    return df