""" 
Utility functions
    - @author Ying Li
    - PRECONDITIONS: various
    - POSTCONDITIONS: various
    - PARAMETERS: various

"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf

def add_date_cols(baskets, date_col = "placed_at"):
    baskets['datetime'] = pd.to_datetime(baskets[date_col])
    baskets['year'] = baskets["datetime"].dt.year
    baskets['month'] = baskets["datetime"].dt.month
    baskets['date'] = baskets["datetime"].dt.date
    baskets['day'] = baskets["datetime"].dt.day
    baskets['hour'] = baskets["datetime"].dt.hour
    baskets['weekday'] = baskets["datetime"].dt.weekday
    baskets['year_month'] = baskets["datetime"].apply(lambda t: t.strftime("%Y-%m"))
    baskets['month_num'] = (baskets['year'] - 2021) * 12 + baskets['month']
    baskets['year_week'] = baskets["datetime"].apply(lambda t: t.strftime("%Y-%W")) # this makes the beginning of Jan 2022 as week 2022-00 , not 2022-52
    baskets['week_num'] = baskets["datetime"].apply(lambda t: int(t.strftime("%W"))) 
    baskets['iso_week_num'] = baskets["datetime"].dt.isocalendar().week # this returns week number 52 for Jan 1, 2021, not 0 which is what we want
    baskets['cum_week_num'] = (baskets['year'] - 2021) * 52 + baskets['week_num']
    return baskets

def get_merchant_attributes(baskets):
    merchant_attributes = baskets.groupby(['merchant_id']).agg(
        total_spent = ('spent', 'sum'), 
        num_orders = ('order_id', 'nunique'), 
        first_month = ('month_num', 'min'), 
        last_month = ('month_num', 'max'), 
        num_months = ('month_num', 'nunique'), 
        num_weeks = ('week_num', 'nunique'), 
        num_days = ('date', 'nunique'), 
        num_skus = ('sku_id','nunique'), 
        num_top_cats = ('top_cat','nunique'), 
        num_sub_cats = ('sub_cat','nunique'),
    ).reset_index()
    merchant_attributes['avg_spent_per_order'] = merchant_attributes.total_spent / merchant_attributes.num_orders
    merchant_attributes['tenure_month'] = merchant_attributes.last_month - merchant_attributes.first_month +1
    return merchant_attributes

def get_sku_attributes(baskets):
    sku_attributes = baskets.groupby(['sku_id']).agg(
        total_spent = ('spent', 'sum'), 
        num_orders = ('order_id', 'nunique'), 
        num_merchants = ('merchant_id', 'nunique'), 
        first_month = ('month_num', 'min'), 
        last_month = ('month_num', 'max'), 
        num_months = ('month_num', 'nunique'), 
        first_week = ('week_num', 'min'), 
        last_week = ('week_num', 'max'), 
        num_weeks = ('week_num', 'nunique'), 
        num_days = ('date', 'nunique'), 
    ).reset_index()
    sku_attributes['avg_spent_per_order'] = sku_attributes.total_spent / sku_attributes.num_orders
    sku_attributes['tenure_month'] = sku_attributes.last_month - sku_attributes.first_month +1
    return sku_attributes

def get_skus_by_week(baskets):
    skus_by_week = baskets.groupby(['sku_id','year_week']).agg(
        avg_price_by_week = ('price','mean'),
        total_spent_by_week = ('spent', 'sum'),
        num_order_by_week = ('order_id', 'nunique'), 
        num_merchants_by_week = ('merchant_id', 'nunique'),
    ).reset_index()
    return skus_by_week

def get_skus_by_day(baskets):
    skus_by_day = baskets.groupby(['sku_id','date']).agg(
        avg_price_by_day = ('price','mean'),
        total_spent_by_day = ('spent', 'sum'),
        num_order_by_day = ('order_id', 'nunique'), 
        num_merchants_by_day = ('merchant_id', 'nunique'),
    ).reset_index()
    return skus_by_day

def get_top_cat_attributes(baskets):
    top_cat_attributes = baskets.groupby(['top_cat']).agg(
        avg_price = ('price', 'mean'),
        total_spent = ('spent', 'sum'),
        total_quantity = ('qty' , 'sum'),
        num_orders = ('order_id', 'nunique'), 
        num_days = ('date' , 'nunique'),
        num_merchants = ('merchant_id', 'nunique')
        ).reset_index()
    return top_cat_attributes