import streamlit as st
import pandas as pd
import numpy as np
import re
import json
import swifter
import openai
from dotenv import load_dotenv, find_dotenv
import os
from joblib import Memory
from rapidfuzz import process, fuzz  #  Faster
# from fuzzywuzzy import fuzz   # 
from stqdm import stqdm
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import time
import requests
from multiprocessing import Pool
import boto3
import s3fs
from smart_open import open 
from datetime import datetime 



env_path = find_dotenv()
print("Loading .env from :", env_path)
load_dotenv(env_path)

AZURE_API_KEY = os.getenv("AZURE_API_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME")
API_VERSION = os.getenv("API_VERSION")
google_distance_api_key = os.getenv("Google_distance_api_key")
locate_one_api_key = os.getenv("locate_one_api_key")

client = openai.AzureOpenAI(
    api_key=AZURE_API_KEY,
    api_version=API_VERSION,
    azure_endpoint=AZURE_ENDPOINT
)


st.markdown(
    "<h1><span style='color:red;'>Delhivery</span> Data Genie</h1>",
    unsafe_allow_html=True
)

##### Data Processing functions - Starts ######
def get_llm_suggestions(column_names):
    """Ask Azure LLM to infer key logistics column names and return structured JSON."""
    prompt = (
        f'''Given these column names: {column_names}, identify columns related to 
        pincode, city,state, origin pincode, origin city, origin state, product_category ,product_code, product_desc,
        product_dimension, length, width, height, product_weight, pack_size. 
        Respond in strict JSON format:
        {{'pincode': 'col_name', 'city': 'col_name',,'state': 'col_name', 
        'pincode_origin': 'origin pincode column', 'city_origin': 'origin city column', 'state_origin': 'origin state column',
        'invoice_no':'invoice number column','order_id':'order id column','quantity':'order quantity column','date':'order/invoice date column',
        'product_category': 'col_name','product_code': 'col_name', 'product_desc': 'col_name',"
        'product_dimension': 'col_name',  'length': 'col_name', 'width': 'col_name',
         'height': 'col_name', 'product_weight': 'col_name', 'pack_size':'col_name'}} 
        If a column is missing, return null (not a string, but actual JSON null). No extra text, just JSON.'''
    )

    #st.write("🔹 **Prompt Sent to Azure:**", prompt)  # ✅ Debugging print

    response = client.chat.completions.create(
        model=DEPLOYMENT_NAME,
        messages=[{"role": "system", "content": "You are an AI assistant."}, {"role": "user", "content": prompt}],
        max_tokens=300
    )

    raw_llm_output = response.choices[0].message.content
    # st.write("🔹 **Raw Azure Response:**", raw_llm_output)  # ✅ Debugging print
    cleaned_output = re.sub(r"```(?:json)?\s*([\s\S]*?)\s*```", r"\1", raw_llm_output)

    try:
        parsed_response = json.loads(cleaned_output)
        # st.write("🔹 **Parsed Azure Response:**", parsed_response)  # ✅ Debugging print
        return parsed_response
    except json.JSONDecodeError:
        st.write("⚠️ **Error: Unable to parse LLM response**")
        return {}


def get_tms_columns_llm_suggestions(column_names):
    """Ask Azure LLM to infer key logistics column names and return structured JSON."""
    prompt = (
        f'''Given these column names: {column_names}, identify columns related to 
        'order_number', 'order_date', 'destination_address', 'destination_pincode',
        'destination_city', 'origin_address', 'origin_pincode', 'origin_city',
        'transaction_type', 'product_code', 'product_name', 'quantity', 'total_cost'

        Respond in strict JSON format:
        {{
        'order_number':'col_name', 'order_date':'col_name', 'destination_address':'col_name', 'destination_pincode':'col_name',
        'destination_city':'col_name', 'origin_address':'col_name', 'origin_pincode':'col_name', 'origin_city':'col_name',
        'transaction_type':'col_name', 'product_code':'col_name', 'product_name':'col_name', 'quantity':'col_name',
          'total_cost':'col_name'
         }} 
        If a column is missing, return null (not a string, but actual JSON null). No extra text, just JSON.'''
    )

    #st.write("🔹 **Prompt Sent to Azure:**", prompt)  # ✅ Debugging print

    response = client.chat.completions.create(
        model=DEPLOYMENT_NAME,
        messages=[{"role": "system", "content": "You are an AI assistant."}, {"role": "user", "content": prompt}],
        max_tokens=400
    )

    raw_llm_output = response.choices[0].message.content
    # st.write("🔹 **Raw Azure Response:**", raw_llm_output)  # ✅ Debugging print
    cleaned_output = re.sub(r"```(?:json)?\s*([\s\S]*?)\s*```", r"\1", raw_llm_output)

    try:
        parsed_response = json.loads(cleaned_output)
        # st.write("🔹 **Parsed Azure Response:**", parsed_response)  # ✅ Debugging print
        return parsed_response
    except json.JSONDecodeError:
        st.write("⚠️ **Error: Unable to parse LLM response**")
        return {}


def parallel_apply_with_progress(func, data):
   
    with ThreadPoolExecutor() as executor:
        # Use tqdm to wrap the executor's map, enabling progress tracking
        result = list(stqdm(executor.map(func, data), total=len(data), desc="Processing"))
    return result 


def extract_dimensions(text):
    """
    Extracts length, width, and height from a given text.
    """
    match = re.findall(r'(\d+(\.\d+)?)\s*[*Xx,]\s*(\d+(\.\d+)?)\s*[*Xx,]\s*(\d+(\.\d+)?)', text)
    if match:
        dimensions = [float(match[0][0]), float(match[0][2]), float(match[0][4])]
        if dimensions == [1.0, 1.0, 1.0]:  # Ignore junk dimensions
            return [np.nan, np.nan, np.nan]
        return dimensions
    
    match = re.findall(r'L[-_]?\s*(\d+(\.\d+)?)\s*[Mm]*,\s*W[-_]?\s*(\d+(\.\d+)?)\s*[Mm]*,\s*H[-_]?\s*(\d+(\.\d+)?)', text, re.IGNORECASE)
    if match:
        dimensions = [float(match[0][0]), float(match[0][2]), float(match[0][4])]
        if dimensions == [1.0, 1.0, 1.0]:  # Ignore junk dimensions
            return [np.nan, np.nan, np.nan]
        return dimensions
    
    return [np.nan, np.nan, np.nan]

def extract_numeric(val):
    match = re.findall(r'\d+\.?\d*', str(val)) 
    if match:
        return float(match[0])
    else:
        return np.nan

def check_columns(row):
    if row[['length', 'width', 'height']].notnull().any():
        return True
    elif row[['length', 'width', 'height']].isnull().any():
        return False

def handle_pack_size(df):
    if all(col in df.columns for col in ['pack_size']):
       
        df['pack_size'] = df['pack_size'].astype(float)
        return df
        
    else: 
        df['pack_size'] = 1 
        return df
    

def process_dataframe(df):
        raw_df = df.copy()
        if 'product_weight' not in df.columns:
            df['product_weight'] = ''

        if all(col in df.columns for col in ['length', 'width', 'height','product_weight']):
            df = df.replace('#REF!', None)
            extract_l = df['length'].apply(lambda x: extract_numeric(x))
            df['length'] = extract_l
            extract_w = df['width'].apply(lambda x: extract_numeric(x))
            df['width'] = extract_w
            extract_h = df['height'].apply(lambda x: extract_numeric(x))
            df['height'] = extract_h
            extract_wt  = df['product_weight'].apply(lambda x: extract_numeric(x))
            df['product_weight'] = extract_wt
            df['Values_Added'] = df.apply(check_columns, axis=1)
        elif all(col in df.columns for col in ['product_dimension','product_weight']):
            df = df.replace('#REF!', None)
            extracted = df['product_dimension'].apply(lambda x: pd.Series(extract_dimensions(str(x))))
            extracted.columns = ['length', 'width', 'height']
            df = df.join(extracted)
            extract_wt  = df['product_weight'].apply(lambda x: extract_numeric(x))
            df['product_weight'] = extract_wt
            df['Values_Added'] = extracted.notna().any(axis=1) 

        if 'length' not in df.columns:
            df['length'] = ''
        if 'width' not in df.columns:
            df['width'] = ''
        if 'height' not in df.columns:
            df['height'] = ''

        df['Values_Added'] = df.apply(check_columns, axis=1)

        df = df.fillna(0)
        df = df.replace('',0)
        df = handle_pack_size(df) 

        df['product_weight'] = df['product_weight'].astype(str)
        df['length'] = df['length'].astype(str)
        df['width'] = df['width'].astype(str)
        df['height'] = df['height'].astype(str)
        # df['product_desc']  = df['product_desc'].replace(0,'not available')
        if 'product_desc' in df.columns:
            df['product_desc'] = df['product_desc'].replace(0, 'not available')
        # if 'product_code' not in df.columns:
        #     df['product_code'] = ''
        else:
            df['product_desc'] = ''

        return df
    
    # else:
    #     return df

def mapping_and_missing_df_create(df):
    if 'product_code' not in df.columns:
        df['product_code'] = ''
    df['sku_volume'] = df['length'].astype(float) * df['width'].astype(float) * df['height'].astype(float)
    mapping_df = df[(df['Values_Added'] == True) & (df['sku_volume'] > 0) & (df['product_weight'] != '0.0') & (df['product_weight']!='#N/A ()')].reset_index(drop=True)
    to_be_mapped_df = df[~((df['Values_Added'] == True) & (df['sku_volume'] > 0) & (df['product_weight'] != '0.0') & (df['product_weight']!='#N/A ()'))].reset_index(drop=True)
    # if 'product_code' in df.columns:
    #     to_be_mapped_df = df[~df['product_code'].isin(mapping_df['product_code'])].reset_index(drop=True)


        # to_be_mapped_df = df[~df['product_desc'].isin(mapping_df['product_desc'])].reset_index(drop=True)

    return mapping_df, to_be_mapped_df

##### Data Processing functions - Ends ######



###### EDA Agent Functions ######

def check_value_counts(df):
    summary = []
    col_consider = ['pincode', 'city', 'state', 'product_category', 'product_code', 'product_desc','length', 'width', 'height', 'pack_size', 'product_weight']
    for col in df.columns:
        if col in col_consider:
            result = {"column": col}
            result["null_count"] = df[col].isnull().sum()
            result["non_null_count"] = df[col].notnull().sum()
            top_vals = df[col].value_counts(dropna=True).head(5).to_dict()
            result["top_5_values"] = top_vals
            if col in ['product_weight','length','width','height']:
                df[col] = df[col].astype(float)
                result["negative_count"] = (df[col] < 0).sum()
                result["zero_count"] = (df[col] == 0).sum()
                result["positive_count"] = (df[col] > 0).sum()

                desc = df[col].describe(percentiles=[0.25, 0.5, 0.75, 0.95])
                result['min'] = desc['min']
                result['25%'] = desc['25%']
                result['50%'] = desc['50%']
                result['75%'] = desc['75%']
                result['95%'] = desc['95%']
                result['max'] = desc['max']
                result['mean'] = desc['mean']
                result['std_dev'] = desc['std']
        
            summary.append(result)
    return pd.DataFrame(summary)

def check_value_counts_by_category(df, custom_col, top_n):
    summary = pd.DataFrame()
    
    top_categories = df[custom_col].value_counts().head(top_n).index
    for category in top_categories:
            df_subset = df[df[custom_col] == category].copy()
            cat_summary = check_value_counts(df_subset)
            # cat_summary1  = cat_summary.copy()
            col_name = custom_col + '_group'
            cat_summary[col_name]= category

            summary = pd.concat([summary, cat_summary], ignore_index=True)

    return summary

import matplotlib.pyplot as plt
import seaborn as sns

def plot_histogram_subplots(df, bins):

    numeric_cols = ['product_weight', 'length', 'width', 'height']
    categorical_cols = ['product_category', 'product_desc']

    total_plots = len(numeric_cols) + len(categorical_cols)
    n_cols = 2
    n_rows = (total_plots + 1) // n_cols

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
    axs = axs.flatten()

    plot_idx = 0
    for col in numeric_cols:
        if col in df.columns:
            data = df[col].astype(float).fillna(0)
            sns.histplot(data, bins=bins, ax=axs[plot_idx], kde=False, color='skyblue', edgecolor='black')

            axs[plot_idx].set_title(f'{col} Distribution')
            axs[plot_idx].set_xlabel(col)
            axs[plot_idx].set_ylabel('Frequency')
            plot_idx += 1

    for col in categorical_cols:
        if col in df.columns:
            top_categories = df[col].value_counts().nlargest(10).index
            filtered_data = df[df[col].isin(top_categories)]
            sns.countplot(y=col, data=filtered_data, ax=axs[plot_idx], palette='Set2', order=top_categories)
            axs[plot_idx].set_title(f'{col} (Top 10 Values)')
            axs[plot_idx].set_xlabel('Count')
            axs[plot_idx].set_ylabel(col)
            plot_idx += 1

    # Hide any unused axes
    for j in range(plot_idx, len(axs)):
        fig.delaxes(axs[j])

    # plt.tight_layout()
    st.pyplot(fig)

def eda_summary_agent(eda_json):
    prompt = f''' You are a Lead data analysis assistant. Below is the EDA summary of a dataset,
    
         {eda_json}

    If product_category_group is present that indicates each category has its own EDA Data .
    Please provide a concise summary covering:
    - Columns with missing and negative values
    - Total records 
    - Numeric columns with statistics (mean, std, min, max, percentiles) in tabular form
    - Most frequent values in all columns including product_category, product_desc, weight, lenght, width and height columns
    - Any outliers or interesting observations
    - Category wise obervation

    just reply with the above pointers nothing else

    '''

    # return prompt
   
    response = client.chat.completions.create(
        model=DEPLOYMENT_NAME,  
        messages=[
            {"role": "system", "content": "You are Lead Data analysis assistant, you are expert in summarising the EDA table, making inenteresting obersations, and category wise observation"},
            {"role": "user", "content": prompt}
        ],
        max_tokens = 2000
    )
    llm_json_output =  response.choices[0].message.content
    return llm_json_output


def order_summ_fun(df_input):

    avg_skus_per_order = None
    avg_qty_per_order = None
    avg_qty_per_sku_per_order = None
    avg_order_lines = None
    avg_orders_per_day = None
    avg_qty_per_day = None
    city_demand = None
    top_80_per_city = None
    state_demand = None
    top_80_per_state = None
    velocity_agg = None
    orderline_per_day = None
    orderline_per_day_first_5 = None
    sku_velocity_tag = None

    if 'order_id' in df_input.columns and 'product_code' in df_input.columns:
        skus_per_order = df_input.groupby('order_id')['product_code'].nunique().reset_index(name='skus_per_order')
        avg_skus_per_order = skus_per_order['skus_per_order'].mean()


    if 'order_id' in df_input.columns and 'quantity' in df_input.columns:
        qty_per_order = df_input.groupby('order_id')['quantity'].sum().reset_index(name='qty_per_order')
        avg_qty_per_order = qty_per_order['qty_per_order'].mean()


    if {'order_id', 'product_code', 'quantity'}.issubset(df_input.columns):
        qty_per_sku_per_order = (df_input.groupby(['order_id', 'product_code'])['quantity'].sum().reset_index(name='qty_per_order_per_sku'))
        avg_qty_per_sku_per_order = qty_per_sku_per_order['qty_per_order_per_sku'].mean()


    if {'order_id', 'product_code', 'date'}.issubset(df_input.columns):

        df_input['date'] = pd.to_datetime(df_input['date']) 
        sku_per_order = df_input.groupby(['date', 'order_id'])['product_code'].nunique().reset_index(name='sku_count')
        avg_sku_per_order_per_day = sku_per_order.groupby('date')['sku_count'].mean().reset_index(name='avg_skus_per_order')
        orders_per_day = df_input.groupby('date')['order_id'].nunique().reset_index(name='num_orders')
        orderline_per_day = pd.merge(avg_sku_per_order_per_day, orders_per_day, on='date')
        orderline_per_day['orderlines_per_day'] = orderline_per_day['avg_skus_per_order'] * orderline_per_day['num_orders']
        orderline_per_day = orderline_per_day[['date', 'orderlines_per_day']] #return this df for day wise orderline 
        orderline_per_day_first_5 = orderline_per_day.head(5)

    if 'quantity' in df_input.columns and 'product_code' in df_input.columns:
        skus_quantity_df = df_input.groupby(['product_code'])['quantity'].sum().reset_index().sort_values(by = 'quantity',ascending=False)
        total_qty = skus_quantity_df['quantity'].sum()
        skus_quantity_df['qty_percent'] = 100 * skus_quantity_df['quantity'] / total_qty
        skus_quantity_df['cum_quantity'] = skus_quantity_df['quantity'].cumsum()
        skus_quantity_df['cum_percent'] = 100 * skus_quantity_df['cum_quantity'] / total_qty
        skus_quantity_df['velocity'] = skus_quantity_df['cum_percent'].apply(lambda x: 'Fast' if x <= 80 else ('Medium' if x <= 96 else 'Slow'))
        sku_velocity_tag = skus_quantity_df[['product_code','velocity']] # return this df to tag products velocity
        
        velocity_agg = skus_quantity_df.groupby('velocity')['product_code'].nunique().reset_index()
        velocity_agg = velocity_agg.rename(columns={'product_code': 'unique_sku_count'})
        
        df_input = df_input.merge(sku_velocity_tag, on = 'product_code', how = 'left')


    if 'date' in df_input.columns and 'order_id' in df_input.columns:
        df_input['date'] = pd.to_datetime(df_input['date'], errors='coerce')
        # Orders per day
        orders_per_day = df_input.groupby(df_input['date'].dt.date)['order_id'].nunique()
        avg_orders_per_day = orders_per_day.mean()


    if 'date' in df_input.columns and 'quantity' in df_input.columns:
        df_input['date'] = pd.to_datetime(df_input['date'], errors='coerce')
        # Quantity per day
        qty_per_day = df_input.groupby(df_input['date'].dt.date)['quantity'].sum()
        avg_qty_per_day = qty_per_day.mean()


    if 'city' in df_input.columns and 'quantity' in df_input.columns:
        city_demand = df_input.groupby('city')['quantity'].sum().reset_index()
        city_demand = city_demand.sort_values(by='quantity', ascending=False)
        total_qty = city_demand['quantity'].sum()
        city_demand['qty_percent'] = 100 * city_demand['quantity'] / total_qty
        city_demand['cum_quantity'] = city_demand['quantity'].cumsum()
        city_demand['cum_percent'] = 100 * city_demand['cum_quantity'] / total_qty
        city_demand['pareto_class'] = city_demand['cum_percent'].apply(lambda x: 'Top 80%' if x <= 80 else 'Bottom 20%')
        city_demand.reset_index(drop=True, inplace=True)
        top_80_per_city = city_demand[city_demand['pareto_class'] == 'Top 80%']


    if 'state' in df_input.columns and 'quantity' in df_input.columns:
        state_demand = df_input.groupby('state')['quantity'].sum().reset_index()
        state_demand = state_demand.sort_values(by='quantity', ascending=False)
        total_qty = state_demand['quantity'].sum()
        state_demand['qty_percent'] = 100 * state_demand['quantity'] / total_qty
        state_demand['cum_quantity'] = state_demand['quantity'].cumsum()
        state_demand['cum_percent'] = 100 * state_demand['cum_quantity'] / total_qty
        state_demand['pareto_class'] = state_demand['cum_percent'].apply(lambda x: 'Top 80%' if x <= 80 else 'Bottom 20%')
        state_demand.reset_index(drop=True, inplace=True)
        top_80_per_state = state_demand[state_demand['pareto_class'] == 'Top 80%']

    order_summary_dict = {
        "avg_skus_per_order": round(avg_skus_per_order, 2) if avg_skus_per_order is not None else None,
        "avg_quantity_per_order": round(avg_qty_per_order, 2) if avg_qty_per_order is not None else None,
        "avg_quantity_per_sku_per_order": round(avg_qty_per_sku_per_order, 2) if avg_qty_per_sku_per_order is not None else None,
        "orderline_per_day": orderline_per_day_first_5[['date', 'orderlines_per_day']].to_dict(orient='records') if orderline_per_day_first_5 is not None else None,
        "sku_velocity_agg_data": velocity_agg.to_dict(orient='records') if velocity_agg is not None else None,
        "avg_orders_per_day":round(avg_orders_per_day, 2) if avg_orders_per_day is not None else None,
        "avg_qty_per_day":round(avg_qty_per_day, 2) if avg_qty_per_day is not None else None,
        "city_demand_summary": {
            "total_cities": len(city_demand) if city_demand is not None else None,
            "80_per_demand_cities_count": len(top_80_per_city) if top_80_per_city is not None else None,
            "80_per_demand_cities_list": top_80_per_city[['city', 'qty_percent']].to_dict(orient='records') if top_80_per_city is not None else None
        },
        "state_demand_summary": {
            "total_states": len(state_demand) if state_demand is not None else None,
            "80_per_demand_states_count": len(top_80_per_state) if top_80_per_state is not None else None,
            "80_per_demand_states_list": top_80_per_state[['state', 'qty_percent']].to_dict(orient='records') if top_80_per_state is not None else None
        }
    }

    return df_input, order_summary_dict, orderline_per_day

def order_summary_agent(eda_json):
    prompt = f''' 
    You are a Lead sales & order data analysis assistant. Below is the summary of a order related metrics and demand summary city or state wise,
    
         {eda_json}

    Please provide a concise summary with good representation covering:
    - average sku's per order, average quantity per order, average quantity per sku per order,  order lines per day (day wise list),
      sku velocity summary, avg orders per day,  avg qty per day (**give each metrics in separate line**)
    - For demand pareto principle,  total X cities/state with 80% demand demand concentrated in Y cities/state (**only for Pareto principle**)
    - City wise demand summary with total cities count, 80 percentage demand cities count, cities list with  quantity contribution 
    - State wise demand summary with total State count, 80 percentage demand State count, State list with  quantity contribution 
    - If any metrics is Null then don't inlcude that in response
    just reply with the above pointers nothing else

    '''

    # return prompt
   
    response = client.chat.completions.create(
        model=DEPLOYMENT_NAME,  
        messages=[
            {"role": "system", "content": "You are Lead Sales Data analysis assistant, you are expert in analyzing the sales/order table, location wise demand, and making interesting obersations"},
            {"role": "user", "content": prompt}
        ],
        max_tokens = 2000
    )
    llm_json_output =  response.choices[0].message.content
    return llm_json_output

def sales_data_plots(df_input, orderline_per_day):
        
    active_plots = []

    if {'quantity', 'date'}.issubset(df_input.columns):

        date_wise_qty = df_input.groupby('date')['quantity'].sum().rename('date_wise_order').reset_index()
        date_wise_qty['date'] = pd.to_datetime(date_wise_qty['date'])
        date_wise_qty = date_wise_qty.sort_values('date')
        date_wise_qty['month_year'] = date_wise_qty['date'].dt.to_period('M').astype(str)

        active_plots.append("plot1")
        active_plots.append("plot2")

    if {'order_id', 'product_code', 'date'}.issubset(df_input.columns):
        active_plots.append("plot3")



    if active_plots:
        fig, axs = plt.subplots(nrows=len(active_plots), figsize=(14, 5 * len(active_plots)))

        if len(active_plots) == 1:
            axs = [axs]  # make iterable

        plot_index = 0

        if "plot1" in active_plots:
            sns.lineplot(
                x='date', y='date_wise_order', data=date_wise_qty, marker='o', ax=axs[plot_index]
            )
            axs[plot_index].set_title('Orders per Day')
            axs[plot_index].tick_params(axis='x', rotation=45)
            axs[plot_index].set_xlabel('Date')
            axs[plot_index].set_ylabel('Orders')
            plot_index += 1

        if "plot2" in active_plots:
            sns.lineplot(
                x='month_year', y='date_wise_order', data=date_wise_qty, marker='o', ax=axs[plot_index]
            )
            axs[plot_index].set_title('Orders per Month')
            axs[plot_index].tick_params(axis='x', rotation=45)
            axs[plot_index].set_xlabel('Date')
            axs[plot_index].set_ylabel('Orders')
            plot_index += 1

        if "plot3" in active_plots:
            sns.lineplot(
                x='date', y='orderlines_per_day', data=orderline_per_day, marker='o', ax=axs[plot_index]
            )
            axs[plot_index].set_title('Orderlines per Day')
            axs[plot_index].set_xlabel('Date')
            axs[plot_index].set_ylabel('Orderlines')

        # plt.tight_layout()
        # plt.show()
        
        st.pyplot(fig)

    else:
        print("No sufficient columns available to generate plots.")

def eda_agent(df_input):
    order_columns = ['invoice_no','order_id','quantity','date']
    sku_columns_list = ["product_code", "product_desc", "product_dimension", "length", "width", "height", "product_weight","pack_size"]
    sku_col_available = [col for col in df_input.columns if col in sku_columns_list]

    if any(col in df_input.columns for col in order_columns):
        st.write("Input data rows count : ", df_input.shape[0])
        df_input, order_summary_dict, orderline_per_day = order_summ_fun(df_input)
        # st.write('after order eda df_input',df_input)
        
        order_summary_agent_output = order_summary_agent(order_summary_dict)
        # print(order_summary_agent_output)
        st.write("EDA Agent output", order_summary_agent_output)
        st.markdown("### **Order Trends**")
        sales_data_plots(df_input, orderline_per_day)

        st.session_state.order_eda = True
        st.session_state['order_df_input'] = df_input
        
    if sku_col_available == ['product_code']:
        print('only product_code columns is present')
    else:
        df = process_dataframe(df_input)
        true_df, false_df = mapping_and_missing_df_create(df)
        # print('Mapping dataset ', true_df.shape)
        # print('Data to map', false_df.shape)
        st.write("Input data rows count : ", df_input.shape[0])
        st.write("Filtered data rows with available weight and dimension : ", true_df.shape[0])
        summary_df = check_value_counts(true_df)
        # st.write("### EDA Summary", summary_df)
        eda_df_json = summary_df.to_dict(orient="records")
        eda_summary_agent_output = eda_summary_agent(eda_df_json)
        print(eda_summary_agent_output)
        st.write("EDA Agent output", eda_summary_agent_output)

        st.markdown("### **Frequency distributions of the columns**")
        plot_histogram_subplots(true_df, 20)

        
        st.session_state.true_df = true_df
        st.session_state.eda_ran = True
    
def eda_agent_custom_column(df_input):  
    df = process_dataframe(df_input)
    true_df, false_df = mapping_and_missing_df_create(df)

    # print('Mapping dataset ', true_df.shape)
    # print('Data to map', false_df.shape)

    st.write("Input data rows count : ", df_input.shape[0])
    st.write("Filtered data rows with available weight and dimension : ", true_df.shape[0])

    prod_desc_columns = ["Select a column..."] + list(true_df.columns)

    default_value = "product_category"
    if default_value in prod_desc_columns:
        default_index = prod_desc_columns.index(default_value)
    else:
        default_index = 0

    cust_col_name= st.selectbox(f"Select column name to run EDA agent on", options = prod_desc_columns, index = default_index )

    # cust_col_name = 'product_category'
    # run_cat_eda = st.button("Run EDA Agent for detailed EDA")
    # if run_cat_eda:
    category_df = check_value_counts_by_category(true_df, cust_col_name, top_n = 5) # change  here to get top n values EDA summary
    eda_df_json = category_df.to_dict(orient="records")
    eda_summary_agent_output = eda_summary_agent(eda_df_json)
    # print(eda_summary_agent_output)
    st.write("EDA Summary Agent", eda_summary_agent_output)

    st.markdown("### **Frequency distributions of the columns**")
    plot_histogram_subplots(true_df, 20)
            
###### EDA function stops ######
    

###### Pincode Mapping Functions  - Starts #####
def fuzzy_match_on_pincode(df):
    # Check if required columns exist
    required_cols = {"mapped_pincode", "mapped_city_db"}
    if not required_cols.issubset(df.columns):
        print("Missing required columns. Skipping fuzzy match on pincode.")
        return df  # Return unchanged dataframe if columns are missing

    df["mapped_pincode"] = df["mapped_pincode"].astype(str).str.zfill(6)  # Ensure pincode is 6 digits
    df["pincode_prefix"] = df["mapped_pincode"].str[:3]  # Extract first 3 digits

    # Filter only rows where mapped_city_tat is missing
    missing_tat_df = df[df["mapped_city_tat"].isna()].copy()

    # If no rows need filling, return early
    if missing_tat_df.empty:
        return df

    # Create a lookup dictionary for pincode-prefix based matching
    lookup_df = df.dropna(subset=["mapped_city_tat"]).copy()
    lookup_df["mapped_pincode"] = lookup_df["mapped_pincode"].astype(int)  # Convert to int for numeric comparison
    lookup_dict = {}

    for (prefix, mapped_city_db), group in lookup_df.groupby(["pincode_prefix", "mapped_city_db"]):
        # Find the closest pincode numerically
        closest_match = group.loc[(group["mapped_pincode"] - group["mapped_pincode"].min()).idxmin()]
        lookup_dict[(prefix, mapped_city_db)] = {
            "mapped_city_tat": closest_match["mapped_city_tat"],
            "mapped_state_db": closest_match["mapped_state_db"]
        }

    # Apply fuzzy matching to fill missing mapped_city_tat
    def apply_fuzzy_match(row):
        key = (row["pincode_prefix"], row["mapped_city_db"])
        if key in lookup_dict:
            match = lookup_dict[key]
            row["mapped_city_tat"] = match["mapped_city_tat"]
            row["mapped_state_db"] = match["mapped_state_db"]
            row["mapping_type"] = "fuzzy match on pincode"
        return row

    df.update(missing_tat_df.apply(apply_fuzzy_match, axis=1))
    df.drop(columns=["pincode_prefix"], inplace=True, errors="ignore")  # Remove helper column
    return df

def validate_pincode(pincode):
    """Check if a pincode is a valid 6-digit Indian pincode."""
    return bool(re.fullmatch(r"[1-9][0-9]{5}", str(pincode)))

def read_pincode_csv_s3():
    # Set up
    bucket = 'abis3'
    prefix = 'Sheetal/pincode_city_mapping/Pincode_city_recent_mapping_with_Lat_Long'

    # List all CSV part files in the folder
    s3 = boto3.client('s3')
    objects = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)

    # Filter only part CSV files
    csv_keys = [obj['Key'] for obj in objects.get('Contents', []) if obj['Key'].endswith('.csv')]

    # Read and concat all parts
    dfs = []
    for key in csv_keys:
        s3_path = f's3://{bucket}/{key}'
        with open(s3_path, 'r') as f:
            df = pd.read_csv(f)
            dfs.append(df)
    # Combine into a single DataFrame
    full_df = pd.concat(dfs, ignore_index=True)
    return full_df

def get_internal_pincode_db(refresh_time=datetime.now()):
    """Fetch the internal pincode database from a local CSV file."""
    try:
        # df_pincode = pd.read_csv("Pincode_Mapping.csv")
        if  '/Users/sheetal.suwalka/Documents/Python_codes' in os.getcwd():
            df_pincode = pd.read_csv('/Users/sheetal.suwalka/Documents/Python_codes/Missing_dimensions/Pincode_Mapping.csv')
        else:
            df_pincode  = read_pincode_csv_s3()

        print('pincode mapping data shape:', df_pincode.shape)
        print('pincode data last records',df_pincode.tail())

        df_pincode.columns = df_pincode.columns.str.strip()
        df_pincode["mapped_pincode"] = df_pincode["mapped_pincode"].astype(str)
        df_pincode["Latitude"] = df_pincode["Latitude"].astype(str)
        df_pincode["Longitude"] = df_pincode["Longitude"].astype(str)
        df_pincode["mapped_city_db"] = df_pincode["mapped_city_db"].str.lower().fillna("Unknown")
        df_pincode["mapped_state_db"].fillna("Unknown", inplace=True)
        return df_pincode
    except FileNotFoundError:
        st.error("Error: Pincode_Mapping.csv file not found.")
        return pd.DataFrame()
    
def get_most_probable_pincode(row):
    """Use Azure LLM to get the most probable pincode for a given Indian city."""
    city = row['city']
    state = row['state'] if 'state' in row and pd.notna(row['state']) else None

    if state:
        state = row['state']
        prompt = (
            f"Given the city '{city}' of the {state} state in India, return the most commonly used 6-digit Indian pincode for this city. "
            "Try to give the pinocde as centrally possible for the given city. "
            "The response MUST follow strict JSON format with double quotes: "
            "{\"mapped_pincode\": \"valid_pincode\"}. "
            "You MUST always return a valid 6-digit pincode—never leave it empty or return 'Unknown'."
        )
    else:
        prompt = (
            f"Given the city '{city}' in India, return the most commonly used 6-digit Indian pincode for this city."
             "Try to give the pinocde as centrally possible for the given city."
            "The response MUST follow strict JSON format with double quotes: "
            "{\"mapped_pincode\": \"valid_pincode\"}. "
            "You MUST always return a valid 6-digit pincode—never leave it empty or return 'Unknown'."
        )

    
    #st.write(f"🔹 Sending to Azure for missing pincode: {prompt}")  # ✅ Debugging print

    try:
        response = client.chat.completions.create(
            model=DEPLOYMENT_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100
        )

        response_text = response.choices[0].message.content.strip()
        #st.write(f"🔹 Azure Response: {response_text}")  # ✅ Debugging print

        response_text = response_text.replace("```json", "").replace("```", "").strip()
        parsed_response = json.loads(response_text) if response_text.startswith("{") else {}

        probable_pincode = parsed_response.get("mapped_pincode", "000000")  # Default fallback

        return probable_pincode if re.fullmatch(r"[1-9][0-9]{5}", probable_pincode) else "000000"  # Validate
    except (json.JSONDecodeError, openai.BadRequestError):
        return "000000"  # Fallback pincode if error occurs
    
def get_correct_pincode_or_city(pincode):
    """Use Azure LLM to correct pincode and find mapped city."""
    
    prompt = (
        f"The given Indian pincode is '{pincode}', please correct it."
        "If this pincode is too long (more than 6 digits), use only the first 6 digits. "
        "If the pincode is too short (less than 6 digits), try adding a suitable last digit to make it valid. "
        "If the last digit seems incorrect, try changing it to the closest valid number. "
        "Always return the closest correct 6-digit pincode based on known patterns. "
        "If the corrected pincode has a mapped big city, include that city name. "
        "Only Return blank if given pincode doesnot contain numbers"
        "Your response MUST be in strict JSON format with no extra text: "
        "{\"mapped_pincode\": \"correct_pincode\", \"mapped_city\": \"correct_city\"}. "
        "Ensure the pincode is valid and exists. Do NOT return 'unknown' or an empty value."
    )

    #st.write(f"🔹 Sending to Azure: {prompt}")  # ✅ Debugging print

    try:
        response = client.chat.completions.create(
            model=DEPLOYMENT_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100
        )

        response_text = response.choices[0].message.content.strip()
        #st.write(f"🔹 Azure Response: {response_text}")  # ✅ Debugging print

        # Clean and parse JSON response
        response_text = response_text.replace("```json", "").replace("```", "").strip()
        parsed_response = json.loads(response_text)

        if isinstance(parsed_response, dict):
            return {
                "mapped_pincode": parsed_response.get("mapped_pincode", "Unknown"),
                "mapped_city": parsed_response.get("mapped_city", "Unknown")
            }
        else:
            return {"mapped_pincode": "Unknown", "mapped_city": "Unknown"}

    except Exception as e:
        #st.write(f"⚠️ Azure API Error: {e}")
        return {"mapped_pincode": "Unknown", "mapped_city": "Unknown"}
    
def get_lat_long(row):
    """Use Azure LLM to get the most probable lat long for a given Indian pincode and city."""
    pincode = row["mapped_pincode"]
    tat_city = row["mapped_city_tat"]
    state_code = row["mapped_state_db"]
    prompt = (
        f'''Given the pincode: '{pincode}', city: '{tat_city}', state code: '{state_code}' in India,
         return the most nearest Latitude and Longitude for the given indian pincode, city, and state code combination.
         prioritize the pincode first, then city and state, as I have seen you gives same lat long for different pincodes
        The response MUST follow strict JSON format with double quotes: 
       {{'Latitude':'latitude,
         'Longitude':'longitude'}}
        You MUST always return a latitude and longitude—never leave it empty or return 'Unknown'.'''
    )

    # return prompt

    response = client.chat.completions.create(
            model=DEPLOYMENT_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100
        )

    
    llm_json_output =  response.choices[0].message.content
    try:
        json_match = re.search(r"```json\n([\s\S]+?)\n```", llm_json_output)

        if json_match:
            json_data = json_match.group(1) 
            data = json.loads(json_data) 
            return pd.Series([data['Latitude'], data['Longitude']])

        else:
            return pd.Series([None, None])
        
    except:
        return pd.Series([None , None,])

#### Function to get Lat Long using LocateOne API
def get_lat_lng_locateoneapi(row):
    url = "https://api.getos1.com/locateone/v1/geocode"
    payload = {"data": {
        "address":  f"{row['mapped_pincode']}, {row['mapped_city_db']}, {row['mapped_state_db']}, India",
        "pincode": row['mapped_pincode'],
        "city": row['mapped_city_db'],
        "state": row['mapped_state_db']}}

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "x-api-key": locate_one_api_key
    }

    time.sleep(0.53)  # ~100 calls per min

    try:
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            data = response.json()
            return pd.Series([row['mapped_pincode'],
                              data['result']['geocode']['lat'],
                              data['result']['geocode']['lng']])
        else:
            return pd.Series([row['mapped_pincode'], 'Error', response.text])
    except Exception as e:
        return pd.Series([row['mapped_pincode'], 'Error', str(e)])    

#### Function to get Lat Long using Google Geocoding API
def get_lat_lng_from_geocode_api(row):
    GEOCODE_URL = "https://maps.googleapis.com/maps/api/geocode/json"
    input_pincode = row['mapped_pincode']
    address = f"{row['mapped_pincode']}, {row['mapped_city_db']}, {row['mapped_state_db']}, India"
    # address = f"{row['mapped_city_db']}, {row['mapped_state_db']}, India"

    params = {"address": address, "region": "in", "key": google_distance_api_key}

    params = {
    # "components": f"postal_code:{input_pincode}|country:IN",
    "address": address,
    "key": google_distance_api_key
}
    # return params
    try:
        response = requests.get(GEOCODE_URL, params=params, timeout=(3, 5))
        data = response.json()
        return data
        if data.get("status") == "OK" and data["results"]:
            result = data["results"][0]
            formatted_address = result["formatted_address"]
            lat = result["geometry"]["location"]["lat"]
            lng = result["geometry"]["location"]["lng"]
            return pd.Series([input_pincode, formatted_address, lat, lng])
        else:
            return pd.Series([input_pincode, None, None, None])
    except Exception:
        return pd.Series([input_pincode, None, None, None])

# function to create one df for pin, city and state
def unique_pin_city_state(df_input):

    for col in ['pincode', 'city', 'state', 'pincode_origin', 'city_origin', 'state_origin']:
        if not col in df_input.columns:
            df_input[col] = ""


    initial_raw_df = df_input.copy()
    unique_od_pair_df = df_input[['pincode', 'city', 'state', 'pincode_origin', 'city_origin', 'state_origin']].drop_duplicates().reset_index(drop=True) 

    pincodes_1 = df_input[['pincode', 'city', 'state']].drop_duplicates()
    pincodes_2 = (df_input[['pincode_origin', 'city_origin', 'state_origin']].drop_duplicates()
                  .rename(columns={'pincode_origin': 'pincode','city_origin':'city','state_origin':'state'}))
    unique_pin_city_state = pd.concat([pincodes_1, pincodes_2], ignore_index=True).drop_duplicates()
    
    return initial_raw_df, unique_od_pair_df, unique_pin_city_state

#  Function to get pincode mapping from internal db and LLM
def pincode_mapping_agent(df_input):
    print('Entered pincode_mapping_agent function')
    df_pincode = get_internal_pincode_db()

    st.write('Mapping Input data to pincode, city and state from Internal Database ...')

    rename_map = {"pincode_origin": "pincode",
    "city_origin": "city",
    "state_origin": "state"}

    for old_col, new_col in rename_map.items():
        if old_col in df_input.columns:
            df_input = df_input.rename(columns={old_col: new_col})


    for col in ["pincode", "city", "state"]:
        if not col in df_input.columns:
            df_input[col] = ""  
            # column_map["pincode"] = "pincode"

    
    # print("Column Map:", column_map)

    progress_bar = st.progress(0)
    progress_text = st.empty()
    total_rows = len(df_input)

    required_columns_set_1 = {'length', 'width', 'height', 'product_weight'}
    required_columns_set_2 = {'product_dimension', 'product_weight'}
    if required_columns_set_1.issubset(df_input.columns) or required_columns_set_2.issubset(df_input.columns):
        df_input = process_dataframe(df_input)

    df_input['pincode'] = df_input['pincode'].replace('', np.nan)

    df_input['pincode'] = df_input['pincode'].astype('Int64')
    df_input['pincode'] = df_input['pincode'].replace(0, None)
    # return df_input
    df_input["pincode"] = df_input["pincode"].astype(str).str.strip()

    df_input["city"] = df_input['city'].astype(str)
    df_input["state"] = df_input['state'].astype(str)

    df_input = df_input.replace('nan','')

    df_input['pincode'] = df_input['pincode'].replace('<NA>', None)
    join_cols = [col for col in ['pincode', 'city', 'state'] if col in df_input.columns]
    # , 'product_code','product_desc'
    df = df_input[join_cols].drop_duplicates()

    # return df

    df["pincode_flag"] = df["pincode"].swifter.apply(validate_pincode)

    df = df.merge(df_pincode, left_on = "pincode", right_on = "mapped_pincode", how="left")
    # return df
    for index, row in df.iterrows():
        # return row
        import pandas as pd
        if (pd.isna(row["pincode"]) or row["pincode"].strip() == "" or row["pincode"] in ["", "nan", "None"]) and (pd.isna(row["city"]) or row["city"].strip() == ""):
            df.at[index, "mapping_type"] = "No Data Provided"
            continue  # Skip further processing for this row


        if not row["pincode_flag"] or pd.isna(row["mapped_city_db"]):  
            # unmapped pincode with internal db
            if pd.isna(row["pincode"]) or row["pincode"] in ["", "nan", "None"] and pd.notna(row['city']):  
                
                    probable_pincode = get_most_probable_pincode(row) 
                    df.at[index, "mapped_pincode"] = probable_pincode  
                    df.at[index, "mapping_type"] = "Azure OpenAI"

                    # Try mapping the new pincode to internal DB
                    matched_pincode_row = df_pincode[df_pincode["mapped_pincode"] == probable_pincode]

                    if not matched_pincode_row.empty:
                        df.at[index, "mapped_city_db"] = matched_pincode_row["mapped_city_db"].values[0]
                        df.at[index, "mapped_state_db"] = matched_pincode_row["mapped_state_db"].values[0]
                        df.at[index, "mapped_city_tat"] = matched_pincode_row["mapped_city_tat"].values[0]
                        df.at[index, "mapping_type"] = "LLM City-based Pincode Match"
                    else:
                        df.at[index, "mapping_type"] = "LLM Pincode - No Match"

            else:  #pincode present but didn't mapped to internal pincode db
                # **Correct invalid pincode using LLM**
                correction = get_correct_pincode_or_city(row["pincode"])
                df.at[index, "mapped_pincode"] = correction["mapped_pincode"]
                df.at[index, "mapped_city_db"] = correction["mapped_city"]

                if pd.notna(df.at[index, "mapped_city_db"]) and df.at[index, "mapped_city_db"] != "Unknown":
                    city_match = df_pincode[df_pincode["mapped_city_db"].str.lower() == df.at[index, "mapped_city_db"].lower()]
                    
                    if not city_match.empty:
                        df.at[index, "mapped_pincode"] = city_match["mapped_pincode"].values[0]  
                        df.at[index, "mapped_state_db"] = city_match["mapped_state_db"].values[0]  
                        df.at[index, "mapped_city_tat"] = city_match["mapped_city_tat"].values[0]  
                        df.at[index, "mapping_type"] = "City-based Mapping from Internal DB"

                # **Check if the corrected pincode exists in Internal DB**
                matched_corrected_pincode_row = df_pincode[df_pincode["mapped_pincode"] == correction["mapped_pincode"]]

                if not matched_corrected_pincode_row.empty:
                    df.at[index, "mapped_state_db"] = matched_corrected_pincode_row["mapped_state_db"].values[0]
                    df.at[index, "mapped_city_tat"] = matched_corrected_pincode_row["mapped_city_tat"].values[0]
                    df.at[index, "mapping_type"] = "LLM Corrected Pincode Match"
                else:
                    df.at[index, "mapping_type"] = "LLM Pincode - No Match"

        # **Update progress bar**
        progress_percent = int(((index + 1) / total_rows) * 100)
        progress_bar.progress(progress_percent)

        progress_text.text(f"Processing... {progress_percent}% completed ({index} rows done) ")

    progress_bar.progress(80)


    df.columns = pd.Series(df.columns).where(~pd.Series(df.columns).duplicated(), 
                                        pd.Series(df.columns) + "_" + pd.Series(df.columns).duplicated().cumsum().astype(str))

    # df = df.swifter.apply(fuzzy_match_on_pincode)
    df =  fuzzy_match_on_pincode(df) # Apply fuzzy matching on pincode

    df1 = df[df['Latitude'].isna()]
    df2 = df[df['Latitude'].notna()]
    # Fetch Latitude and Longitude from LLM

    print("Fetching Latitude and Longitude through Delhivery's LocateOne API...")
        
    rows = [row for index, row in df1.iterrows()]

        # Use parallel_apply_with_progress function instead of swifter.apply
    # lat_long_output =  parallel_apply_with_progress(lambda row: get_lat_long(row), rows)

    import pandas as pd

    locate_one_results = []

    for _, row in stqdm(df1.iterrows(), total=len(df1), desc="Processing"):
        res = get_lat_lng_locateoneapi(row)
        locate_one_results.append(res)

    locate_one_api_output_df = pd.DataFrame(
        [item.tolist() if item is not None else [None] * 3 for item in locate_one_results],
        columns=['mapped_pincode','locate_one_latitudes', 'locate_one_longitudes'])

    df1  = df1.drop(columns = ['Latitude','Longitude'])
    df1 = df1.merge(locate_one_api_output_df, how='left', on = ['mapped_pincode'])
    df1 = df1.rename(columns = {'locate_one_latitudes':'Latitude','locate_one_longitudes':'Longitude'})

    # st.write("llm Response output", lat_long_output)

    # latitudes = []
    # longitudes = []

    # # Iterate through the results array and extract the values
    # for res in lat_long_output:
    #     latitudes.append(res[0])  # Extract latitude
    #     longitudes.append(res[1]) # Extract longitude

    # # # Assign the latitudes and longitudes to the DataFrame columns
    # df1['Latitude'] = latitudes
    # df1['Longitude'] = longitudes

    # st.write("Added Lat Long to Unmapped", df1)

    df = pd.concat([df1, df2], ignore_index=True )

    progress_bar.progress(100)
    progress_text.text("✅ Processing Complete!")

    if "mapping_type" not in df.columns:
        df["mapping_type"] = ""
    df["mapping_type"] = df["mapping_type"].replace("", pd.NA).fillna("Direct Mapping")
    df = df.fillna("")
    df = df.astype(str).replace("nan", "").replace("None", "").replace("NaN", "")
    df.drop(columns=["pincode_prefix"], errors="ignore", inplace=True)

    # st.write("Processed after lat long:", df.head(10))

    import pandas as pd
    import io
    df_input[['pincode', 'city', 'state']] = df_input[['pincode', 'city', 'state']].replace(r'^\s*$', np.nan, regex=True)
    df[['pincode', 'city', 'state']] = df[['pincode', 'city', 'state']].replace(r'^\s*$', np.nan, regex=True)

    df['state'] = df['state'].astype(object)
    df['city'] = df['city'].astype(object)
    # df = clean_df(df, join_cols)
    # df_input = clean_df(df_input, join_cols)
    df_input['state'] = df_input['state'].astype(object)
    df_input['city'] = df_input['city'].astype(object)

    df_input = df_input.fillna('0')
    df  = df.fillna('0')
    df_input_processed = df_input.merge(df, how='left', on = join_cols)

    df_input_processed = df_input_processed.fillna('0')

    return df_input_processed

# Separate pincodwe mapping at origin & destination level 
def origin_destination_mapping(df_input_processed, raw_od_pair_df):

    raw_od_pair_df = raw_od_pair_df.fillna(0)
    raw_od_pair_df = raw_od_pair_df.replace('','0')
    
    for col in ['pincode','pincode_origin']:

        raw_od_pair_df[col] = raw_od_pair_df[col].astype('Int64')
        raw_od_pair_df[col] = raw_od_pair_df[col].astype(object)

    for col in ['pincode', 'city', 'state', 'pincode_origin', 'city_origin', 'state_origin']:
        raw_od_pair_df[col] = raw_od_pair_df[col].astype(str).str.strip()

    for col in ['pincode', 'city', 'state']:
        df_input_processed[col] = df_input_processed[col].astype(str).str.strip()

    raw_od_pair_df = raw_od_pair_df.merge(df_input_processed, how='left', 
                            left_on=['pincode', 'city', 'state'], right_on = ['pincode', 'city', 'state'])

    raw_od_pair_df = raw_od_pair_df.merge(df_input_processed, how='left', 
                            left_on=['pincode_origin', 'city_origin', 'state_origin'], right_on = ['pincode', 'city', 'state'], suffixes=('', '_origin_1'))
    
    raw_od_pair_df = raw_od_pair_df.drop(columns=['pincode_origin_1', 'city_origin_1', 'state_origin_1'])
    
    return raw_od_pair_df

# Creating a custom parallel process function for OSRM API
def parallel_apply_with_progress_osrm(func, data):
    with ThreadPoolExecutor(max_workers=5) as executor:
        # Use tqdm to wrap the executor's map, enabling progress tracking
        result = list(stqdm(executor.map(func, data), total=len(data), desc="Processing"))
    return result 

# Fetching distance from OSRM API using OD Lat and Long 
def get_distance_from_osrm(row, mode='driving'):
    try:
        origin_lng = row['Longitude_origin_1']
        origin_lat = row['Latitude_origin_1']

        destination_lng = row['Longitude']
        destination_lat = row['Latitude']

        # print(abs(float(origin_lng) -  float(destination_lng)))
        # print(abs(float(origin_lat) -  float(origin_lat)))

        if abs(float(origin_lng) -  float(destination_lng)) < 0.05 and abs(float(origin_lat) -  float(origin_lat)) <0.05:
            return pd.Series([20])
        
        else:

            # """
            # Function to get distance from OSRM API.
            # """
            base_url = 'http://router.project-osrm.org/route/v1'

            profile = mode  # can be 'driving', 'walking', 'cycling'
            coordinates = f"{origin_lng},{origin_lat};{destination_lng},{destination_lat}"

            url = f"{base_url}/{profile}/{coordinates}?overview=false"

            # print(url)
            response = requests.get(url)
            # return response
            try: 
                data = response.json()  # This line could raise a JSONDecodeError
            # return data
            except json.JSONDecodeError:
                return pd.Series([distance_in_km])

                # Check if the response contains valid data
            if 'routes' in data and len(data['routes']) > 0:
                    # Distance is in meters
                    distance_in_meters = data['routes'][0]['distance']

                    # Convert meters to kilometers
                    distance_in_km = distance_in_meters / 1000
                    return pd.Series([distance_in_km])

            else:
                return pd.Series([None])
            
    except Exception as e:
        return pd.Series([None])
    

#### Function to get google distances using Lat Long from LocateOne API
def get_distance_from_google(row, mode='driving'):

    # Create origin and destination distance using pincode, city state
    # origin = f"{row['mapped_pincode_origin_1']}, {row['mapped_city_db_origin_1']}, India"
    # destination = f"{row['mapped_pincode']}, {row['mapped_city_db']}, India"

    # Create origin and destination distance using Lat Long
    origin = f"{row['Latitude_origin_1']}, {row['Longitude_origin_1']}"
    destination = f"{row['Latitude']}, {row['Longitude']}"

    base_url = "https://maps.googleapis.com/maps/api/distancematrix/json"

    params = {
        "origins": origin,
        "destinations": destination,
        "mode": mode,           # driving, walking, bicycling, transit
        "units": "metric",
        "region": "in",         # bias towards India
        "key": google_distance_api_key
    }

    try:
        response = requests.get(base_url, params=params, timeout=10)
        data = response.json()
    except Exception as e:
        return pd.Series([None, None, None])

    if data.get("status") == "OK":
        element = data["rows"][0]["elements"][0]
        if element.get("status") == "OK":
            origin_address = data.get("origin_addresses", [None])[0]
            destination_address = data.get("destination_addresses", [None])[0]
            distance_km = element["distance"]["value"] / 1000  # meters to km
            return pd.Series([origin_address, destination_address, distance_km])
        else:
            return pd.Series([None, None, None])
    else:
        return pd.Series([None, None, None])

# Joining OD distace data with initial raw df
def join_raw_df(raw_df, initial_data_OD_lat_long):
    
    raw_df = raw_df.fillna(0)
    raw_df = raw_df.replace('','0')

    for col in ['pincode','pincode_origin']:

        raw_df[col] = raw_df[col].astype('Int64')
        raw_df[col] = raw_df[col].astype(object)

    # raw_df['pincode_origin'] = raw_df['pincode_origin'].astype('Int64')
    # raw_df['pincode_origin'] = raw_df['pincode_origin'].astype(object)

    for col in ['pincode', 'city', 'state', 'pincode_origin', 'city_origin', 'state_origin']:
        raw_df[col] = raw_df[col].astype(str).str.strip()
        initial_data_OD_lat_long[col] = initial_data_OD_lat_long[col].astype(str).str.strip()

    final_df = raw_df.merge(initial_data_OD_lat_long, how='left', on = ['pincode', 'city', 'state', 'pincode_origin', 'city_origin', 'state_origin'])
    return final_df

###### Pincode functions - Ends ######



###### Missing Dimension Functions - Starts ######

# memory = Memory("cache_directory", verbose=0)
# @memory.cache

@st.cache_data
def azure_llm_density_threshold(row, client_name, product_types):
    if row['product_desc'] in ['', 'not available']:
        print('entered this loop in azure_llm_density_threshold fun')
        prod_desc = row['product_desc'] 
        den_lower_threshold = 1
        den_upper_threshold = 50
        density_unit = 'kg/ft3'
        return pd.Series([prod_desc, None, None, density_unit,den_lower_threshold, den_upper_threshold, None, None])

    else:
        prod_desc = row['product_desc'] 
        prompt = f''' Given the product description: '{prod_desc}' for the '{client_name}' company which manufactures '{product_types}' products,
        Write a concise product description of 100 characters based on your understanding. This should be formatted exactly as it would appear in a product catalog database (no extra descriptive details).
        Based on the detailed product description, estimate the product category and for that category, provide the following:
        
        1. Lower and upper thresholds for the weight in kg
        2. Lower and upper density thresholds for a closed-box product (in kg/ft^3)

        Ensure that the weight threshold range has a minimum difference of 10 kg between the lower and upper bounds, and the density threshold range has at least a 10 kg/ft^3 difference. 
        Provide appropriate ranges based on typical materials used for products in this category (do not use null values).

        Output format: The output must be strictly in the following JSON format (no extra explanations):

        # Sample Output :
        ```json
            {{'detailed_product_description)': 'product description'
            'product_category': 'Category'
            'weight_lower_threshold': 'lower threshold of weight in kg'
            'weight_upper_threshold': 'upper threshold of weight in kg'
            'density_unit' : 'Density unit kg/ft^3'
                'density_lower_threshold' : 'lower threshold of Density in kg/ft^3'
                'density_upper_threshold' : 'upper threshold of Density in kg/ft^3 '
                }}
        ```

        Please re-evaluate and ensure the density thresholds are correct and within a reasonable range for typical product categories.   
        '''

        response = client.chat.completions.create(
            model=DEPLOYMENT_NAME,  
            messages=[
                {"role": "system", "content": "You are a product catalog expert specializing in providing accurate density and weight thresholds based on product category and description."},
                {"role": "user", "content": prompt}
            ],
            max_tokens = 400,
            temperature=0  # Set temperature to 0 for deterministic output
        )
        llm_json_output =  response.choices[0].message.content
        # return llm_json_output

        try:
            json_match = re.search(r"```json\n([\s\S]+?)\n```", llm_json_output)
            # json_match = re.search(r"```json\s*(\{[\s\S]+?\})\s*```", llm_json_output)
            # return json_match
        

            if json_match:
                json_data = json_match.group(1) 
                data = json.loads(json_data) 
                # return data
                den_lower_threshold = data['density_lower_threshold'] - 20
                den_upper_threshold = data['density_upper_threshold'] + 5

                wt_lower_threshold = data['weight_lower_threshold'] - 2
                wt_upper_threshold = data['weight_upper_threshold'] + 2

                if  den_lower_threshold <= 0 or den_lower_threshold > 50 :
                    den_lower_threshold = 1
                
                if den_upper_threshold > 50:
                    den_upper_threshold = 50

                if wt_lower_threshold <=0:
                    wt_lower_threshold = data['weight_lower_threshold']/2

                return pd.Series([prod_desc, data['detailed_product_description'],data['product_category'],data['density_unit']
                                ,den_lower_threshold, den_upper_threshold, wt_lower_threshold, wt_upper_threshold])

            else:
                pd.Series([None, None, None, None, None, None, None, None])

        except:
            return pd.Series([None, None, None, None, None, None, None, None])
        


conversion_factors = {
    'mm': 0.00328084,
    'cm': 0.0328084,
    'm': 3.28084,
    'inch': 1/12,
    'ft': 1 }


# Get density as per dimension unit and check if lies in density threshold
def compute_density(df, unit):
    factor = conversion_factors[unit]
    df['length'] = df['length'].astype(float)
    df['width'] = df['width'].astype(float)
    df['height'] = df['height'].astype(float)
    df['product_weight'] = df['product_weight'].astype(float)

    volume_ft3 = (df['length'] * factor) * (df['width'] * factor) * (df['height'] * factor)
    
    density = df['product_weight'] / volume_ft3
    
    return density


def best_guess_unit(row):
    if row['product_desc'] in ['', 'not available']:
        for unit in conversion_factors:
            density = row[f'density_{unit}']
            if row['density_lower_threshold'] <= density <= row['density_upper_threshold']: # Comparing density with each product threshold
                return unit
        return 'unknown'

    else:
        for unit in conversion_factors:
            density = row[f'density_{unit}']
            # if row['density_lower_threshold'] <= density <= row['density_upper_threshold']: # Comparing density with each product threshold
            if (row['density_lower_threshold'] <= density <= row['density_upper_threshold']) & (row['weight_lower_threshold'] <= row['product_weight'] <= row['weight_upper_threshold']): # Comparing density with each product threshold
                return unit
        return 'unknown'
        


def azure_llm_suggested_product_dimensions(row, client_name, product_types, weight_unit, dimension_unit):

    matched_from = None
    matched_material = None  # Store matched Material Code
    matched_description = None  # Store matched Material Description
    values_added = 'Not Mapped'   # Retain existing value from df_false
    gross_weight = row['product_weight']  # Retain existing Gross weight from df_false
    user_prod_desc = row['product_desc']
    length = row['length']
    width = row['width']
    height = row['height']
    pack_size = row['pack_size']
    lower_density_threshold = row['density_lower_threshold']
    upper_density_threshold = row['density_upper_threshold']



    if pd.isna(lower_density_threshold) or lower_density_threshold == 0 or lower_density_threshold == '0.0':
        lower_density_threshold = 1

    if pd.isna(upper_density_threshold) or upper_density_threshold == 0 or upper_density_threshold == '0.0':
        upper_density_threshold = 50
    
    if pd.isna(pack_size) or pack_size == 0 or pack_size == '0.0':  # Only update if packsize is missing or 0
        pack_size = 1
    
    prompt = f'''Given the product description {user_prod_desc} for the {client_name} company which makes {product_types} kinds of products,
        return  the  matched product description in more detail, product weight, length, width, height, source of information and how confident are you on output as confidence percentage.
        Consider there are total {pack_size} items in the closed box.
        You should give dimensions and weight so that the density of the product is in range of {lower_density_threshold} kg/ft3 to {upper_density_threshold} kg/ft3.
        You must look at all the information available you have on which you were trained, and provide the output.

        ## Dummy Output format: 
        Output format: The response MUST follow strict JSON format with double quotes and weight unit should be {weight_unit},
        and length, width and height unit should be {dimension_unit}. Don't include unit in the response, just the float value.
        {{'matched_product_description':'product description',
        'product_weight':'2', 
        'length':'10',
        'width':'10',
        'height':'10',
        'dimension_unit':unit of dimension,
        'confidence_percentage': '75%',
        'source_of_information': 'source of information'}}

        You MUST always return some json output and never leave it empty or return 'Unknown'
    '''


    # return prompt
   
    response = client.chat.completions.create(
        model=DEPLOYMENT_NAME,  
        messages=[
            {"role": "system", "content": "You are a product catalog expert and you are master in providing the product dimensions "
            "and product weight informtion by just looking at the product description"},
            {"role": "user", "content": prompt}
        ],
        max_tokens = 200
    )
    llm_json_output =  response.choices[0].message.content
    # return llm_json_output
    try:
        json_match = re.search(r"```json\n([\s\S]+?)\n```", llm_json_output)
    

        if json_match:
            json_data = json_match.group(1) 
            data = json.loads(json_data) 
            matched_from = 'LLM generated response'
            # return data
           
            values_added_1 = 'Added Dimension and Weight'
            pack_size = 1
            return pd.Series([data['length'], data['width'], data['height'], data['dimension_unit'] , data['product_weight'], pack_size, values_added_1,
                        matched_from, matched_material, data['matched_product_description'] ,data['confidence_percentage']])
            
        else:
            return pd.Series([length , width, height, None,gross_weight, pack_size, values_added, matched_from, matched_material, matched_description, None]) 
            # return 'no output'

    except:
        return pd.Series([length , width, height,None, gross_weight, pack_size, values_added, matched_from, matched_material, matched_description, None]) 

def find_best_match(row, df_true):
    best_match = None
    best_score = 0
    matched_from = None
    matched_material = None  # Store matched Material Code
    matched_description = None  # Store matched Material Description
    values_added = 'Not Mapped'  # Retain existing value from df_false
    gross_weight = row['product_weight']  # Retain existing Gross weight from df_false
    length = row['length']
    width = row['width']
    height = row['height']
    pack_size = row['pack_size']

    if len(df_true) == 0:
        return pd.Series([None, None, None,None, None, None, values_added, None, None, None, None])

    
    if row['product_desc'] in ['', 'not available']:
        for index, true_row in df_true.iterrows():
            if str(row['product_code'])[:3] == str(true_row['product_code'])[:3]:  # First 3 chars must match
                score_code = fuzz.ratio(str(row['product_code']), str(true_row['product_code'])) 
                if score_code > 90:
                    best_match = true_row
                    best_score = score_code
                    confidence  = '>90%'
                    matched_from = "product_code"
                    dimension_unit = true_row['guessed_unit']
                    matched_material = true_row['product_code']  # Store matched Material Code
                    # matched_description = true_row['product_desc']  # Store matched Material Description
                    if values_added is None:  # Only update if it was missing
                        values_added = true_row['Values_Added']

                    # values_added_1 = 'Added Dimension'
                    # if pd.isna(gross_weight) or gross_weight == '1.0' or gross_weight == '0.0':  # Only update if Gross weight is missing or 1
                    gross_weight = true_row['product_weight']
                    pack_size = true_row['pack_size']
                    values_added_1 = 'Added Dimension and Weight'

                    return pd.Series([
                        best_match['length'], best_match['width'], best_match['height'], dimension_unit,
                        gross_weight, pack_size, values_added_1, matched_from, 
                        matched_material, None, confidence ])
                else:
                    return pd.Series([None, None, None,None, None, None, values_added, None, None, None, None])

    # Try matching on 'Material' (First 3 characters must match)

    else:
        for index, true_row in df_true.iterrows():
            if str(row['product_code'])[:3] == str(true_row['product_code'])[:3]:  # First 3 chars must match
                score_code = fuzz.ratio(str(row['product_code']), str(true_row['product_code']))  #
                score_desc = fuzz.ratio(str(row['product_desc']), str(true_row['product_desc']))
                # return score_code, score_desc
                if score_code > 80 and score_desc > 80: 
                    best_match = true_row
                    best_score = score_code
                    confidence  = '>90%'
                    matched_from = "product_code_and_desc"
                    dimension_unit = true_row['guessed_unit']
                    matched_material = true_row['product_code']  # Store matched Material Code
                    matched_description = true_row['product_desc']  # Store matched Material Description
                    if values_added is None:  # Only update if it was missing
                        values_added = true_row['Values_Added']

                    # values_added_1 = 'Added Dimension'
                    # if pd.isna(gross_weight) or gross_weight == '1.0' or gross_weight == '0.0':  # Only update if Gross weight is missing or 1
                    gross_weight = true_row['product_weight']
                    pack_size = true_row['pack_size']
                    values_added_1 = 'Added Dimension and Weight'

                    return pd.Series([
                        best_match['length'], best_match['width'], best_match['height'], dimension_unit,
                        gross_weight, pack_size, values_added_1, matched_from, 
                        matched_material, matched_description, confidence ])
                
                
                elif score_code < 80 and score_desc >= 80: 
                    best_match = true_row
                    best_score = score_desc
                    confidence  = '>70%'
                    matched_from = "product_desc"
                    dimension_unit = true_row['guessed_unit']
                    matched_material = true_row['product_code']  # Store matched Material Code
                    matched_description = true_row['product_desc']  # Store matched Material Description
                    if values_added is None:  # Only update if it was missing
                        values_added = true_row['Values_Added']

                    # values_added_1 = 'Added Dimension'
                    # if pd.isna(gross_weight) or gross_weight == '1.0' or gross_weight == '0.0':  # Only update if Gross weight is missing or 1
                    gross_weight = true_row['product_weight']
                    pack_size = true_row['pack_size']
                    values_added_1 = 'Added Dimension and Weight'

                    return pd.Series([
                        best_match['length'], best_match['width'], best_match['height'], dimension_unit,
                        gross_weight, pack_size, values_added_1, matched_from, 
                        matched_material, matched_description, confidence ])

            
            else:
                score_desc = fuzz.ratio(str(row['product_desc']), str(true_row['product_desc']))

                if score_desc > 85:
                    best_match = true_row
                    best_score = score_desc
                    confidence  = '>85%'
                    matched_from = "product_desc"
                    dimension_unit = true_row['guessed_unit']
                    matched_material = true_row['product_code']  # Store matched Material Code
                    matched_description = true_row['product_desc']  # Store matched Material Description
                    if values_added is None:  # Only update if it was missing
                        values_added = true_row['Values_Added']

                    # values_added_1 = 'Added Dimension'
                    # if pd.isna(gross_weight) or gross_weight == '1.0' or gross_weight == '0.0':  # Only update if Gross weight is missing or 1
                    gross_weight = true_row['product_weight']
                    pack_size = true_row['pack_size']
                    values_added_1 = 'Added Dimension and Weight'

                    return pd.Series([
                        best_match['length'], best_match['width'], best_match['height'], dimension_unit,
                        gross_weight, pack_size, values_added_1, matched_from, 
                        matched_material, matched_description ,confidence ])

        return pd.Series([None, None, None,None, None, None, values_added, None, None, None, None])  # No match found, retain original values

def weight_anomaly_detection(df_input):
    # Initialize progress bar and text
    progress_bar = st.progress(0)
    status_text = st.empty()  
    print('entered this weight_anomaly_detection function')
    # Process DataFrame
    df = process_dataframe(df_input)
    status_text.text("Dataframe Created for Processing...")
    progress_bar.progress(10)

    status_text.text("Web searching the density threshold for each product.. this can take a while...")
    progress_bar.progress(20)

    client_name = st.session_state.client_name,
    product_types = st.session_state.product_types

    df_unique_prod = df[['product_desc']].drop_duplicates().reset_index(drop=True)

    rows = [row for index, row in df_unique_prod.iterrows()]

    ###### Calling LLM to get weight and density thresholds
    density_threshold_output = parallel_apply_with_progress(lambda row: azure_llm_density_threshold(row,
                                            client_name, product_types), rows)

    density_df = pd.DataFrame([item.tolist() if item is not None else [None] * 8
        for item in density_threshold_output], columns=['input_prod_desc','detailed_product_description', 'llm_product_category', 'density_unit',
        'density_lower_threshold', 'density_upper_threshold','weight_lower_threshold', 'weight_upper_threshold'])

    df = df.merge(density_df, left_on='product_desc', right_on='input_prod_desc', how='left').drop(columns =['input_prod_desc'])
    
    ###### bifurcating data in available and missing values
    true_df, false_df = mapping_and_missing_df_create(df)
    print('Mapping dataset ', true_df.shape)
    print('Data to map', false_df.shape)

    progress_bar.progress(80)

    # Creating the column for density by assuming dimension unit as (mm, cm, m, inch, ft)
    for unit in conversion_factors:
        true_df[f'density_{unit}'] = compute_density(true_df, unit)

    true_df['guessed_unit'] = true_df.apply(best_guess_unit, axis=1) #guessing the dimension unit based on density


    true_df_true = true_df[true_df['guessed_unit']!= 'unknown']
    true_df_false = true_df[true_df['guessed_unit'] == 'unknown']

    true_df_true['Values_Added'] = 'Dimensions and Weight is in Material density range'
    true_df_false['Values_Added'] = 'Anomaly - Dimensions and Weight are out of Material density range'
    false_df['Values_Added'] = 'Dimension or Weight is Missing'
    df_combined = pd.concat([true_df_true, true_df_false, false_df], ignore_index=True)

    df=df_combined

    status_text.text("Process completed successfully!")
    progress_bar.progress(100)
    
    return df

def missing_weight_dimensions_estimator_agent(df_input):
    # Initialize progress bar and text
    progress_bar = st.progress(0)
    status_text = st.empty()   
    print('entered this missing_weight_dimensions_estimator_agent function')
    ##### Cleaning input data
    # if st.session_state.get('eda_ran', False):
    #     print('eda already ran')
        
    # else: 
    #     print('eda not ran')
    # st.write('before process',df_input)
    df = process_dataframe(df_input)
    # st.write('after process df_input',df_input)
    # st.write('after process df',df)
    status_text.text("Dataframe Created for Processing...")
    progress_bar.progress(20)

    status_text.text("Web searching the density threshold for each product.. this can take a while...")
    progress_bar.progress(40)

    ##### Create df with unique product description
    df_unique_prod = df[['product_desc']].drop_duplicates().reset_index(drop=True)

    rows = [row for index, row in df_unique_prod.iterrows()]

    client_name = st.session_state.client_name,
    product_types = st.session_state.product_types

    ##### Calling LLM to get density & weight threshold for unique products
    density_threshold_output = parallel_apply_with_progress(lambda row: azure_llm_density_threshold(row,
                                            client_name, product_types), rows)

    density_df = pd.DataFrame([
        item.tolist() if item is not None else [None] * 8
        for item in density_threshold_output], columns=[
        'input_prod_desc','detailed_product_description', 'llm_product_category', 'density_unit',
        'density_lower_threshold', 'density_upper_threshold','weight_lower_threshold', 'weight_upper_threshold'])

    df = df.merge(density_df, left_on='product_desc', right_on='input_prod_desc', how='left').drop(columns =['input_prod_desc'])

    # st.write('llm density thresholds', df)
    
    true_df, false_df = mapping_and_missing_df_create(df)
    print('input df', df.shape)
    print('Mapping dataset ', true_df.shape)
    print('Data to map', false_df.shape)
    # st.write('true_df',true_df)
    # st.write('false_df',false_df)

    # Creating the column for density by assuming dimension unit as (mm, cm, m, inch, ft)
    for unit in conversion_factors:
        true_df[f'density_{unit}'] = compute_density(true_df, unit)

    true_df['guessed_unit'] = true_df.apply(best_guess_unit, axis=1) #guessing the dimension unit based on density

    true_df_true = true_df[true_df['guessed_unit']!= 'unknown']
    true_df_false = true_df[true_df['guessed_unit'] == 'unknown'] #Move the rows which are not in the density range to False

    print('true_df within density range',true_df_true.shape)
    print('true_df outside density range',true_df_false.shape)

    false_df_combined = pd.concat([false_df, true_df_false], ignore_index=True) # combined outside density range true_df to false_df

    print('overall false_df_combined rows',false_df_combined.shape)
    status_text.text("Finding best match in mapping file this can take a while...")
    progress_bar.progress(30)

    rows = [row for index, row in false_df_combined.iterrows()]
    k1 = false_df_combined.copy()

    # st.write('true_df_true', true_df_true, true_df_true.shape)
    # st.write('false_df_combined', false_df_combined)
    ##### 1. Finding best match for anomolous values within golden dataset
    best_match_output = parallel_apply_with_progress(lambda row: find_best_match(row,true_df_true), rows)

    # st.write('best_match_output', best_match_output)

    false_df_combined[['predicted_length', 'predicted_width', 'predicted_height', 'predicted_unit','predicted_weight',
                    'predicted_pack_size', 'Values_Added','Matched_From', 'Matched_Material', 'Matched_Description', 'Match_confidence']] = best_match_output
    
    
    false_df_mapped = false_df_combined[false_df_combined['Matched_From'].notna()]
    false_df_2 = false_df_combined[false_df_combined['Matched_From'].isna()].reset_index(drop=True)
    print('Unmapped dataset from mapping file', false_df_2.shape)
    print('Mapped dataset from mapping file',false_df_mapped.shape) 


    false_df_2_process = false_df_2[false_df_2['product_desc']!='not available']
    false_df_2_skip = false_df_2[false_df_2['product_desc']=='not available']
    status_text.text("Mapping done!! Sending unmapped data to LLM for processing")
    progress_bar.progress(60)

    ##### 2. Calling LLM to get dimensions for unmatched values from best match function
    
    rows = [row for index, row in false_df_2_process.iterrows()]
    # Use parallel_apply_with_progress function instead of swifter.apply
    client_name = st.session_state.client_name,
    product_types = st.session_state.product_types
    weight_unit = st.session_state.weight_unit
    dimension_unit = st.session_state.dimension_unit 
    llm_output_lbh = pd.DataFrame(parallel_apply_with_progress(lambda row: azure_llm_suggested_product_dimensions(row,
                        client_name, product_types, weight_unit , dimension_unit ), rows))

    false_df_2_process[['predicted_length', 'predicted_width', 'predicted_height', 'predicted_unit','predicted_weight', 'predicted_pack_size',
                        'Values_Added','Matched_From', 'Matched_Material', 'Matched_Description', 'Match_confidence']] = llm_output_lbh

    true_df_true['Values_Added'] = 'Dimensions and Weight present' ## Golden dataset
    false_df_2_skip['Values_Added']= 'NA' ## Unmapped dataset by LLM and internal mapping

    # st.write('true_df_true',true_df_true)
    # st.write('false_df_mapped',false_df_mapped)
    # st.write('false_df_2_process',false_df_2_process)
    # st.write('false_df_2_skip',false_df_2_skip)
    df_combined = pd.concat([true_df_true, false_df_mapped, false_df_2_process, false_df_2_skip], ignore_index=True)

    df=df_combined

    # columns_to_fill = ['Matched_From', 'Matched_Material', 'Matched_Description', 'Match_confidence']
    # df[columns_to_fill] = df[columns_to_fill].replace('', 'NA').fillna('NA')
    
    status_text.text("Process completed successfully!")
    progress_bar.progress(100)

    return df

###### Missing Dimension Functions  - Ends ######



###### Data cleaning functions - Starts ######
def convert_df_to_excel(df, uploaded_file, sheet_name):
    with pd.ExcelWriter('processed_data.xlsx', engine='openpyxl') as writer:
        for sheet in pd.ExcelFile(uploaded_file).sheet_names:
            if sheet == sheet_name:
                df.to_excel(writer, sheet_name=sheet, index=False)
            else:
                pd.read_excel(uploaded_file, sheet_name=sheet).to_excel(writer, sheet_name=sheet, index=False)
    with open('processed_data.xlsx', 'rb') as f:
        return f.read()

def clean_df(df, join_cols):
        df = df.copy()
        for col in join_cols:
            if col in ['pincode', 'pincode_origin','length','width','height','pack_size','product_weight']:
                df[col] = df[col].apply(lambda x: int(float(x)) if pd.notna(x) and str(x).strip() not in ["", "nan", "NaN", "<NA>"] else str(x))
                # df[col] = df[col].astype(object)
                df[col] = df[col].astype(str).str.strip()

            else:
            
                df[col] = df[col].apply(lambda x: x if pd.isna(x) else str(x))
                # df[col] = df[col].astype(object)
                df[col] = df[col].astype(str).str.strip()
                df[col] = df[col].apply(lambda x: x.upper() if isinstance(x, str) else x)

        return df
  

###### Initializing sessions
if 'client_name' not in st.session_state:
    st.session_state.client_name = "Halonix"

if 'product_types' not in st.session_state:
    st.session_state.product_types = "LED Bulbs, battens and Fans"

if "dimension_unit" not in st.session_state:
    st.session_state.dimension_unit = "inch" 

if "weight_unit" not in st.session_state:
    st.session_state.weight_unit = "kg" 

if 'missing_value_button' not in st.session_state:
    st.session_state['missing_value_button'] = None
if 'anomaly_button' not in st.session_state:
    st.session_state['anomaly_button'] = None
if 'pincode_map_button' not in st.session_state:
    st.session_state['pincode_map_button'] = None

# File uploader
uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx", "xls","csv"])

if 'column_names' not in st.session_state:
    st.session_state.column_names = []
if 'column_map' not in st.session_state:
    st.session_state.column_map = {}

# if 'llm_response' not in st.session_state:
#     st.session_state.llm_response = None
# if 'current_file' not in st.session_state:
#     st.session_state.current_file = None

stqdm.pandas()

if uploaded_file:
    file_name = uploaded_file.name.lower()

    if file_name.endswith(".csv"):
        # Preview without header
        df_preview = pd.read_csv(uploaded_file, header=None)
        sheet_names =  os.path.splitext(uploaded_file.name)[0]
        # st.write('sheet_names',sheet_names)
        sheet_name = st.selectbox("Select Sheet", sheet_names)
        header_row = st.number_input("Select Header Row (0-indexed)", min_value=0, max_value=len(df_preview)-1, value=0)
        # Re-read with header
        uploaded_file.seek(0)  # reset pointer
        df_input = pd.read_csv(uploaded_file, header=header_row)

    elif file_name.endswith((".xlsx", ".xls")):
        # Read Excel and get sheet names
        xls = pd.ExcelFile(uploaded_file)
        sheet_name = st.selectbox("Select Sheet", xls.sheet_names)
        # Preview without header
        df_preview = pd.read_excel(xls, sheet_name=sheet_name, header=None)
        header_row = st.number_input("Select Header Row (0-indexed)", min_value=0, max_value=len(df_preview)-1, value=0)
        # Read with proper header
        df_input = pd.read_excel(xls, sheet_name=sheet_name, header=header_row)

    st.write("Data with Selected Header:")
    st.write(df_input.head(10))

    print('Started new session')

    st.write("### Enter Client details and Verify/Update column mapping")

    st.session_state.client_name = st.text_input("Enter Client Name", value=st.session_state.client_name)

    st.session_state.product_types = st.text_input("General product types of the client", value=st.session_state.product_types)

    st.session_state.dimension_unit = st.selectbox("Select Dimension Unit",
                            options=["inch", "cm", "mm", "m", "ft"],
                            index=["inch", "cm", "mm", "m", "ft"].index(st.session_state.dimension_unit))

    st.session_state.weight_unit = st.selectbox(
                                "Select Weight Unit",
                                options=["kg", "gm"],
                                index=["kg", "gm",].index(st.session_state.weight_unit))
    
    # column_names = list(df_input.columns)
    # column_map = {}

    st.session_state.column_names = list(df_input.columns)
    # st.session_state.column_map = {} 

    column_names = st.session_state.column_names
    # st.write("Identified columns:", column_names)
    st.write("Identified columns:", st.session_state.column_names)

    # if st.button("Proceed to map columns"):
    st.markdown("""
        <style>
        div.stButton > button:first-child {
            background-color: #007BFF;  /* Bootstrap Primary Blue */
            color: white;
            padding: 0.75em 1.5em;
            font-size: 1.2em;
            font-weight: bold;
            border-radius: 0.5em;
            border: none;
            transition: background-color 0.3s;
        }
        div.stButton > button:first-child:hover {
            background-color: #0056b3;
            cursor: pointer;}
        </style>""", unsafe_allow_html=True)

# Your button logic
    if st.button("Auto-Map Columns Using AI"):
        st.success("Proceeding to map columns...")
        st.session_state.llm_response = get_llm_suggestions(column_names)
        st.session_state.column_map = {} 
        st.rerun()

    
    if "llm_response" in st.session_state:
        llm_response = st.session_state.llm_response
        # st.write("LLM Suggested Columns:", llm_response)
        
        expected_columns = ["pincode","city","state","pincode_origin","city_origin","state_origin",
                            "invoice_no","order_id","quantity","date", "product_category","product_code",
                                "product_desc","product_dimension", "length", "width", "height", "product_weight","pack_size"]
        
        # for col in expected_columns:
        #     suggested_col = llm_response.get(col, "Not found")
        #     st.session_state.column_map[col] = st.selectbox(f"Select column for {col}", [None] + column_names, 
        #             index=(column_names.index(suggested_col) + 1) if suggested_col in column_names else 0, key=f"alt_{col}")

        num_cols_per_row = 3
        for i in range(0, len(expected_columns), num_cols_per_row):
            row_cols = expected_columns[i:i + num_cols_per_row]
            cols = st.columns(len(row_cols))
            for j, col in enumerate(row_cols):
                suggested_col = llm_response.get(col, "Not found")
                st.session_state.column_map[col] = cols[j].selectbox(f"Select column for {col}", [None] + column_names, 
                    index=(column_names.index(suggested_col) + 1) if suggested_col in column_names else 0, key=f"alt_{col}")
                

    st.write("## Column Mapping Summary")
    column_map = st.session_state.column_map
    st.write(column_map) 


    clean_mapping = {v: k for k, v in column_map.items() if v is not None}

    df_input = df_input.rename(columns=clean_mapping)

    st.write("mapped columns preview", df_input.head(10))
          
    df_pincode_processed = None
    df_anomaly_result  = None
    df_missing_values_result = None

    skus_columns = ["product_category","product_code", "product_desc",
                    "product_dimension", "length", "width", "height", "product_weight","pack_size"]
    
    pincode_columns = ["pincode","city","state","pincode_origin","city_origin","state_origin"]

    order_columns = ['invoice_no','order_id','quantity','date']

    pin_1 = {'pincode', 'city', 'state'}
    pin_2 = {'pincode_origin', 'city_origin', 'state_origin'}

    def missing_col_null_check(df_input,expected_columns):
        existing_columns = list(df_input.columns)
        missing_columns = [col for col in expected_columns if col not in existing_columns]
        present_columns = [col for col in expected_columns if col in existing_columns]
        stats = []

        for col in present_columns:
            null_count = int(df_input[col].isna().sum())
            stats.append((col,  null_count))

        return {"columns not available": missing_columns, "null count in the columns": stats}
    
    ### input df has order data
    if (any(col in df_input.columns for col in order_columns)):
        
        nulls_detail = missing_col_null_check(df_input,order_columns)
        with st.status("Plan of Action", expanded=True):
            st.markdown(f"""
                    - ✅ Found columns for Order data
                    - **Details:** `{nulls_detail}`  
                    - 🤖 EDA Agent will be activated 
                    - 📊  EDA Agent will show order Summary, order pareto analysis and date level trend
                    - 🤖 Super cleaning Agent will be activated 
                    - 🗺️ Agent will map clean the pincode, find anomalies and predict missing weight and dimension based on data 
                    """)

    ### input df has only pincode columns
    elif (any(col in df_input.columns for col in pin_1) and not any(col in df_input.columns for col in pin_2) 
                            and not any(col in df_input.columns for col in skus_columns)):
        
        nulls_detail = missing_col_null_check(df_input,pin_1)
        with st.status("Plan of Action", expanded=True):
            st.markdown(f"""
                    - ✅ Found columns for Pincode/City/State  
                    - **Details:** `{nulls_detail}`  
                    - 🤖 Pincode Agent will be Activated  
                    - 🗺️ Agent will map the data to Delhivery internal data: pincode, city, state, and TAT city  
                    - 📍 Agent will fetch Latitude and Longitude using LLM  
                    """)
            
    ### input df has origin and destination pincode columns
    elif (any(col in df_input.columns for col in pin_1) and  any(col in df_input.columns for col in pin_2)
                        and not any(col in df_input.columns for col in skus_columns)):
        nulls_detail = missing_col_null_check(df_input,pincode_columns)
        with st.status("Plan of Action", expanded=True):
            st.markdown(f"""
                - ✅ Found columns for Origin and destination Pincode/City/State  
                - **Details:** `{nulls_detail}` 
                - 🤖 Pincode Distance Agent will be Activated
                - 🗺️ Agent will map the origin and destination data to delhivery 
                        internal data: pincode, city, state and TAT city  
                - 📍 Pincode Agent will fetch Latitude and Longitude using LLM  
                - 📏 Agent will calculate distance between origin and destination pincode using OSRM API
                """)
            
    ### input df has only product weight/dimension columns
    elif (not any(col in df_input.columns for col in pin_1) and not any(col in df_input.columns for col in pin_2) 
                            and any(col in df_input.columns for col in skus_columns)):
        
        sku_cols_1 = ["product_dimension", "product_weight"]
        sku_cols_2 = ["length", "width", "height", "product_weight"]
        sku_base = ["product_category","product_code", "product_desc", "pack_size"]
        
        if any(col in df_input.columns for col in skus_columns):
            has_cols_1 = all(col in df_input.columns for col in sku_cols_1)
            has_cols_2 = all(col in df_input.columns for col in sku_cols_2)

            # has sku_cols_1 but all values are present
            if has_cols_1 and not (df_input[sku_cols_1].isnull().any() | (df_input[sku_cols_1] == 0).any()).any():
                req_cols = sku_base + sku_cols_1
                nulls_detail = missing_col_null_check(df_input,req_cols)

                with st.status("Plan of Action", expanded=True):
                    st.markdown(f"""
                            - ✅ Found columns for Product 📦 Weights and Dimension
                            - **Details:** `{nulls_detail}` 
                            - 🤖 EDA and Anomaly Agent will be Activated
                            - 📊  EDA Agent will show Data Summary and Frequency distributions of the columns
                            - 🧹  Dimension/Weight Analyzer Agent will clean the data, fetch Weight & Density threshold from LLM 
                                and creates golden/true dataset
                            - 🔍 Agent finds the anomalous weight/dimension products based on Weight and Density thresholds 
                                """)

            # has sku_cols_2 but all values are present
            elif has_cols_2 and not (df_input[sku_cols_2].isnull().any() | (df_input[sku_cols_2] == 0).any()).any():
                req_cols = sku_base + sku_cols_2
                nulls_detail = missing_col_null_check(df_input,req_cols)

                with st.status("Plan of Action", expanded=True):
                    st.markdown(f"""
                            - ✅ Found columns for Product 📦 Weights and Dimension
                            - **Details:** `{nulls_detail}` 
                            - 🤖 EDA and Anomaly Agent will be Activated
                            - 📊  EDA Agent will show Data Summary and Frequency distributions of the columns
                            - 🧹  Dimension/Weight Analyzer Agent will clean the data, fetch Weight & Density threshold from LLM 
                                and creates golden/true dataset
                            - 🔍 Agent finds the anomalous weight/dimension products based on Weight and Density thresholds 
                                """)

            # has missing values
            else:
                if has_cols_1:
                    req_cols = sku_base + sku_cols_1
                    nulls_detail = missing_col_null_check(df_input,req_cols)
                elif has_cols_2:
                    req_cols = sku_base + sku_cols_2
                    nulls_detail = missing_col_null_check(df_input,req_cols)
                else:
                    nulls_detail = None

               
                with st.status("Plan of Action", expanded=True):
                    st.markdown(f"""
                            - ✅ Found columns for Product 📦 Weights and Dimension
                            - **Details:** `{nulls_detail}`
                            - 🤖 EDA and Dimension/Weight Analyzer Agent will be Activated
                            - 📊  EDA Agent will show Data Summary and Frequency distributions of the columns
                            - 🧹  Dimension/Weight Analyzer Agent will clean the data, fetch Weight & Density threshold from LLM 
                                and creates golden/true dataset
                            - 🔍 Agent finds best match for missing/anomalous values within true dataset on product code and description
                            - 🧠 Agent calls LLM to predict for unmapped products 
                                """)
                    
    ### input df has pincode and product weight/dimension columns
    elif (any(col in df_input.columns for col in pin_1) and not any(col in df_input.columns for col in pin_2) 
                            and any(col in df_input.columns for col in skus_columns)):
        
        sku_cols_1 = ["product_dimension", "product_weight"]
        sku_cols_2 = ["length", "width", "height", "product_weight"]
        sku_base = ["product_category","product_code", "product_desc", "pack_size"]
        
        if any(col in df_input.columns for col in skus_columns):
            has_cols_1 = all(col in df_input.columns for col in sku_cols_1)
            has_cols_2 = all(col in df_input.columns for col in sku_cols_2)

            if has_cols_1 and not (df_input[sku_cols_1].isnull().any() | (df_input[sku_cols_1] == 0).any()).any():
                req_cols = list(pin_1) + sku_base + sku_cols_1
                nulls_detail = missing_col_null_check(df_input,req_cols)
                
                with st.status("Plan of Action", expanded=True):
                    st.markdown(f"""
                            - ✅ Found columns for Pincode/City/State  
                            - **Details:** `{nulls_detail}`
                            - 🤖 Pincode Agent will be Activated  
                            - 🗺️ Agent will map the data to Delhivery internal data: pincode, city, state, and TAT city  
                            - 📍 Agent will fetch Latitude and Longitude using LLM  
                            - ✅ Found columns for Product 📦 Weights and Dimension
                            - 🤖 EDA and Anomaly Agent will be Activated
                            - 📊  EDA Agent will show Data Summary and Frequency distributions of the columns
                            - 🧹  Dimension/Weight Analyzer Agent will clean the data, fetch Weight & Density threshold from LLM 
                                and creates golden/true dataset
                            - 🔍 Agent finds the anomalous weight/dimension products based on Weight and Density thresholds 
                                """)

            elif has_cols_2 and not (df_input[sku_cols_2].isnull().any() | (df_input[sku_cols_2] == 0).any()).any():
                req_cols = list(pin_1) + sku_base + sku_cols_2
                nulls_detail = missing_col_null_check(df_input,req_cols)

                with st.status("Plan of Action", expanded=True):
                    st.markdown(f"""
                            - ✅ Found columns for Pincode/City/State  
                            - **Details:** `{nulls_detail}`
                            - 🤖 Pincode Agent will be Activated  
                            - 🗺️ Agent will map the data to Delhivery internal data: pincode, city, state, and TAT city  
                            - 📍 Agent will fetch Latitude and Longitude using LLM  
                            - ✅ Found columns for Product 📦 Weights and Dimension
                            - 🤖 EDA and Anomaly Agent will be Activated
                            - 📊  EDA Agent will show Data Summary and Frequency distributions of the columns
                            - 🧹  Dimension/Weight Analyzer Agent will clean the data, fetch Weight & Density threshold from LLM 
                                and creates golden/true dataset
                            - 🔍 Agent finds the anomalous weight/dimension products based on Weight and Density thresholds 
                                """)

            else:
                if has_cols_1:
                    req_cols = list(pin_1)+ sku_base + sku_cols_1
                    nulls_detail = missing_col_null_check(df_input,req_cols)

                    with st.status("Plan of Action", expanded=True):
                        st.markdown(f"""
                                - ✅ Found columns for Pincode/City/State  
                                - **Details:** `{nulls_detail}`
                                - 🤖 Pincode Agent will be Activated  
                                - 🗺️ Agent will map the data to Delhivery internal data: pincode, city, state, and TAT city  
                                - 📍 Agent will fetch Latitude and Longitude using LLM  
                                - ✅ Found columns for Product 📦 Weights and Dimension
                                - 🤖 EDA and Dimension/Weight Analyzer Agent will be Activated
                                - 📊  EDA Agent will show Data Summary and Frequency distributions of the columns
                                - 🧹  Dimension/Weight Analyzer Agent will clean the data, fetch Weight & Density threshold from LLM 
                                    and creates golden/true dataset
                                - 🔍 Agent finds best match for missing/anomalous values within true dataset on product code and description
                                - 🧠 Agent calls LLM to predict for unmapped products 
                                    """)
                elif has_cols_2:
                    req_cols = list(pin_1)+  sku_base + sku_cols_2
                    nulls_detail = missing_col_null_check(df_input,req_cols)

                    with st.status("Plan of Action", expanded=True):
                        st.markdown(f"""
                                - ✅ Found columns for Pincode/City/State  
                                - **Details:** `{nulls_detail}`
                                - 🤖 Pincode Agent will be Activated  
                                - 🗺️ Agent will map the data to Delhivery internal data: pincode, city, state, and TAT city  
                                - 📍 Agent will fetch Latitude and Longitude using LLM  
                                - ✅ Found columns for Product 📦 Weights and Dimension
                                - 🤖 EDA and Dimension/Weight Analyzer Agent will be Activated
                                - 📊  EDA Agent will show Data Summary and Frequency distributions of the columns
                                - 🧹  Dimension/Weight Analyzer Agent will clean the data, fetch Weight & Density threshold from LLM 
                                    and creates golden/true dataset
                                - 🔍 Agent finds best match for missing/anomalous values within true dataset on product code and description
                                - 🧠 Agent calls LLM to predict for unmapped products 
                                    """)
       
                    
     ### input df has pincode/city columns for origin/destination and product weight/dimension columns
    elif (any(col in df_input.columns for col in pin_1) and any(col in df_input.columns for col in pin_2) 
                            and any(col in df_input.columns for col in skus_columns)):
        
        sku_cols_1 = ["product_dimension", "product_weight"]
        sku_cols_2 = ["length", "width", "height", "product_weight"]
        sku_base = ["product_category","product_code", "product_desc", "pack_size"]
        
        if any(col in df_input.columns for col in skus_columns):
            has_cols_1 = all(col in df_input.columns for col in sku_cols_1)
            has_cols_2 = all(col in df_input.columns for col in sku_cols_2)

            if has_cols_1 and not (df_input[sku_cols_1].isnull().any() | (df_input[sku_cols_1] == 0).any()).any():
                
                req_cols = pincode_columns + sku_base + sku_cols_1
                nulls_detail = missing_col_null_check(df_input,req_cols)

                with st.status("Plan of Action", expanded=True):
                    st.markdown(f"""
                            - ✅ Found columns for Origin and destination Pincode/City/State  
                            - **Details:** `{nulls_detail}`
                            - 🤖 Pincode Distance Agent will be Activated
                            - 🗺️ Agent will map the origin and destination data to delhivery 
                                    internal data: pincode, city, state and TAT city  
                            - 📍 Pincode Agent will fetch Latitude and Longitude using LLM  
                            - 📏 Agent will calculate distance between origin and destination pincode using OSRM API
                            - ✅ Found columns for Product 📦 Weights and Dimension
                            - 🤖 EDA and Anomaly Agent will be Activated
                            - 📊  EDA Agent will show Data Summary and Frequency distributions of the columns
                            - 🧹  Dimension/Weight Analyzer Agent will clean the data, fetch Weight & Density threshold from LLM 
                                and creates golden/true dataset
                            - 🔍 Agent finds the anomalous weight/dimension products based on Weight and Density thresholds 
                                """)

            elif has_cols_2 and not (df_input[sku_cols_2].isnull().any() | (df_input[sku_cols_2] == 0).any()).any():
                req_cols = pincode_columns + sku_base + sku_cols_2
                nulls_detail = missing_col_null_check(df_input,req_cols)

                with st.status("Plan of Action", expanded=True):
                    st.markdown(f"""
                            - ✅ Found columns for Origin and destination Pincode/City/State  
                            - **Details:** `{nulls_detail}`
                            - 🤖 Pincode Distance Agent will be Activated
                            - 🗺️ Agent will map the origin and destination data to delhivery 
                                    internal data: pincode, city, state and TAT city  
                            - 📍 Pincode Agent will fetch Latitude and Longitude using LLM  
                            - 📏 Agent will calculate distance between origin and destination pincode using OSRM API 
                            - ✅ Found columns for Product 📦 Weights and Dimension
                            - 🤖 EDA and Anomaly Agent will be Activated
                            - 📊  EDA Agent will show Data Summary and Frequency distributions of the columns
                            - 🧹  Dimension/Weight Analyzer Agent will clean the data, fetch Weight & Density threshold from LLM 
                                and creates golden/true dataset
                            - 🔍 Agent finds the anomalous weight/dimension products based on Weight and Density thresholds 
                                """)

            else:
                if has_cols_1:
                    req_cols = pincode_columns +sku_base + sku_cols_1
                    nulls_detail = missing_col_null_check(df_input,req_cols)

                    with st.status("Plan of Action", expanded=True):
                        st.markdown(f"""
                                - ✅ Found columns for Origin and destination Pincode/City/State  
                                - **Details:** `{nulls_detail}`
                                - 🤖 Pincode Distance Agent will be Activated
                                - 🗺️ Agent will map the origin and destination data to delhivery 
                                        internal data: pincode, city, state and TAT city  
                                - 📍 Pincode Agent will fetch Latitude and Longitude using LLM  
                                - 📏 Agent will calculate distance between origin and destination pincode using OSRM API
                                - ✅ Found columns for Product 📦 Weights and Dimension
                                - 🤖 EDA and Dimension/Weight Analyzer Agent will be Activated
                                - 📊  EDA Agent will show Data Summary and Frequency distributions of the columns
                                - 🧹  Dimension/Weight Analyzer Agent will clean the data, fetch Weight & Density threshold from LLM 
                                    and creates golden/true dataset
                                - 🔍 Agent finds best match for missing/anomalous values within true dataset on product code and description
                                - 🧠 Agent calls LLM to predict for unmapped products 
                                    """)
                        
                elif has_cols_2:
                    req_cols = pincode_columns +  sku_base + sku_cols_2
                    nulls_detail = missing_col_null_check(df_input,req_cols)

                    with st.status("Plan of Action", expanded=True):
                        st.markdown(f"""
                                - ✅ Found columns for Origin and destination Pincode/City/State  
                                - **Details:** `{nulls_detail}`
                                - 🤖 Pincode Distance Agent will be Activated
                                - 🗺️ Agent will map the origin and destination data to delhivery 
                                        internal data: pincode, city, state and TAT city  
                                - 📍 Pincode Agent will fetch Latitude and Longitude using LLM  
                                - 📏 Agent will calculate distance between origin and destination pincode using OSRM API
                                - ✅ Found columns for Product 📦 Weights and Dimension
                                - 🤖 EDA and Dimension/Weight Analyzer Agent will be Activated
                                - 📊  EDA Agent will show Data Summary and Frequency distributions of the columns
                                - 🧹  Dimension/Weight Analyzer Agent will clean the data, fetch Weight & Density threshold from LLM 
                                    and creates golden/true dataset
                                - 🔍 Agent finds best match for missing/anomalous values within true dataset on product code and description
                                - 🧠 Agent calls LLM to predict for unmapped products 
                                    """)


    if st.button("Run SCS Agent"):
        st.success("Proceeding to Clean the data...")
        #### Activating EDA Agent #### 
        if any(col in df_input.columns for col in skus_columns) | any(col in df_input.columns for col in order_columns):
        #     #### functions for EDA agent
            # st.write('before eda df_input', df_input)
            eda_agent(df_input)
            print('EDA Function ends')
            
            if 'order_df_input' in st.session_state:
                df_input = st.session_state.get('order_df_input')
                # st.write('after eda order_df_input', df_input)

        
        #### Activating Pincode Agent ####
        if any(col in df_input.columns for col in pincode_columns):
            if any(col in df_input.columns for col in pin_1) and not any(col in df_input.columns for col in pin_2):
                
                ### function to get lat long and pincode mapping for pincodes
                final_df = pincode_mapping_agent(df_input)
                print('Pincode Agent ends')
                # st.write('pincode mapped df', final_df.head(50))

            elif not any(col in df_input.columns for col in pin_1) and  any(col in df_input.columns for col in pin_2):
                final_df = pincode_mapping_agent(df_input)
                print('Pincode Agent ends')

            elif any(col in df_input.columns for col in pin_1) and  any(col in df_input.columns for col in pin_2):

                ### function flow to get distance betwwen given sets of pincodes
                initial_raw_df, unique_od_pair_df, unique_pin_city_state_df = unique_pin_city_state(df_input)

                df_input_processed = pincode_mapping_agent(unique_pin_city_state_df)

                # st.write('pincode mapped df', df_input_processed.head(50))
                # st.write('unique OD mapped df', unique_od_pair_df.head(50))

                initial_data_OD_lat_long = origin_destination_mapping(df_input_processed, unique_od_pair_df)

                # st.write('OD mapped df', initial_data_OD_lat_long.head(50))

                # st.write("Fetching distance between origin and destination through OSRM API. Please wait...")
                # rows = [row for index, row in initial_data_OD_lat_long.iterrows()]
                # distance_km = parallel_apply_with_progress_osrm(lambda row: get_distance_from_osrm(row), rows)
                # # If there are nulls in distance_km then check for nulls in lat and long and then try to change parallel_apply_with_progress_osrm function 
                # distance_df = pd.DataFrame([item.tolist() if item is not None else [None] * 1
                #             for item in distance_km], columns=['distance_km'])
                # initial_data_OD_lat_long['distance_km'] = distance_df

                ###### Calculating distance using Google API
                st.write("Fetching distance between origin and destination through Google API. Please wait...")
                rows = [row for index, row in initial_data_OD_lat_long.iterrows()]
                google_distance_km = parallel_apply_with_progress_osrm(lambda row: get_distance_from_google(row), rows)
                google_distance_df = pd.DataFrame([item.tolist() if item is not None else [None] * 1
                            for item in google_distance_km], columns=['origin_address','destination_addresses', 'google_distance_km'])
                initial_data_OD_lat_long[['origin_address','destination_addresses', 'google_distance_km']] = google_distance_df

                final_df = join_raw_df(initial_raw_df, initial_data_OD_lat_long)
                final_df = final_df.fillna('0')
                final_df = final_df.replace('<NA>','0')
                final_df = final_df.replace('nan','0')
                # st.write('distance mapped df', final_df.head(50))
                print('Pincode Agent ends')

            else:
                st.markdown("<span style='color:red; font-weight:bold; font-size:20px'>Select appropriate columns for pincode/city/state</span>",
                    unsafe_allow_html=True)
            st.session_state['pincode_map_button'] = final_df


        #### Activating Anomaly Detection Agent #### 

        sku_cols_1 = ["product_dimension", "product_weight"]
        sku_cols_2 = ["length", "width", "height", "product_weight"]
        # if st.button("Run Dimensions/Weight Anomaly Detection Agent"):
        skus_columns_1 = [ "product_desc", "product_dimension", "length", "width", "height", "product_weight","pack_size"]
        # st.write('before product cleaning df', df_input)
        if any(col in df_input.columns for col in skus_columns_1):
            has_cols_1 = all(col in df_input.columns for col in sku_cols_1)
            has_cols_2 = all(col in df_input.columns for col in sku_cols_2)

            if has_cols_1 and not (df_input[sku_cols_1].isnull().any() |  (df_input[sku_cols_1].isin([0, 0.0, '0', '0.0',''])).any() ).any():
                
                df = weight_anomaly_detection(df_input)
                st.session_state['anomaly_button'] = df
                print('Anomaly Agent ends')

            elif has_cols_2 and not (df_input[sku_cols_2].isnull().any() |  (df_input[sku_cols_2].isin([0, 0.0, '0', '0.0',''])).any() ).any():

                df = weight_anomaly_detection(df_input)
                st.session_state['anomaly_button'] = df
                print('Anomaly Agent ends')

        #### Activating Missing Weights/Dimension Analyzer Agent #### 
            else:
        # if st.button("Run Missing Weight/Dimensions Estimator Agent "): 
                # print('entered this loop')
                has_cols_1 = all(col in df_input.columns for col in sku_cols_1)
                has_cols_2 = all(col in df_input.columns for col in sku_cols_2)
                # if has_cols_1:
                df = missing_weight_dimensions_estimator_agent(df_input)
                # df_missing_values_result = df
                st.session_state['missing_value_button'] = df
                print('Missing values estimator Agent ends')

                # elif has_cols_2:
                #     df = missing_weight_dimensions_estimator_agent(df_input)
                #     # df_missing_values_result = df
                #     st.session_state['missing_value_button'] = df

    ##### Joining all the dataframes ####

    def join_all_df(df1, df2, df3):

        dfs = {'df_missing_values_result': df1, 'df_anomaly_result': df2, 'df_pincode_processed': df3}

        dataframes = [df for df in [df1, df2, df3] if df is not None]

        if len(dataframes) == 0:
            return None  
        elif len(dataframes) == 1:
            return dataframes[0]  
        else:     
            result = dataframes[0]
            result = result.fillna(0)
            
            # joininig_cols = [col for col in ['pincode', 'city', 'state', 'product_code', 'product_desc'] if col in result.columns]
            joininig_cols = list(df_input.columns)
            # st.write('df_input',df_input)
            result = clean_df(result, joininig_cols)
            # st.write('result',result)
            

            for df in dataframes[1:]:
                df = df.fillna(0)
                # st.write(joininig_cols, df.head(20))
                df = clean_df(df, joininig_cols)

                # print('df',df[joininig_cols][df['length']=='0'].head(10))
                # st.write('missing df',  result[joininig_cols].head(50), result[joininig_cols].dtypes)
                # st.write('pincode df',  df[joininig_cols].head(50), df[joininig_cols].dtypes)
                # st.write('Debugging',result[joininig_cols].head(50), df[joininig_cols].head(50) )
                result = pd.merge(result, df, on=joininig_cols, how='left') 
            return result

    # joininig_cols = list(df_input.columns)
    df1 = st.session_state.get('missing_value_button')
    # st.write('missing_value_ df', df1.head(50))
    df2 = st.session_state.get('anomaly_button')
    # st.write('anomaly df',  df2[joininig_cols].head(50), df2[joininig_cols].dtypes)
    df3 = st.session_state.get('pincode_map_button')
    # st.write('pincode df', df3.head(50))

    # st.write('input df columns', list(df_input.columns))
    
    final_all_joined = join_all_df(df1, df2, df3)

    if final_all_joined is not None:
        print('Processing Complete')
        st.write("### Final Processed DataFrame")
        st.write(final_all_joined.head(10))

        if file_name.endswith(".csv"):
            csv_data = final_all_joined.to_csv(index=False)
            st.download_button(
                label="Download Processed Data as CSV",
                data=csv_data,
                file_name='processed_data.csv',
                mime='text/csv', )


        elif file_name.endswith((".xlsx", ".xls")):
            excel_data = convert_df_to_excel(final_all_joined, uploaded_file, sheet_name)
            st.download_button(
                label="Download Processed Data as Excel",
                data=excel_data,
                file_name='processed_data.xlsx',
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', )
        
        # try:
        #     output_df_columns = list(final_all_joined.columns)
        #     st.session_state.llm_response = get_tms_columns_llm_suggestions(output_df_columns)


