# ==============================================================================
# 1. IMPORTS
# ==============================================================================
import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import re
import time
from io import BytesIO
import swifter
import openai
from openai import AzureOpenAI
from dotenv import load_dotenv, find_dotenv
from joblib import Memory
from rapidfuzz import process, fuzz
from datetime import datetime
import boto3
from smart_open import open
from stqdm import stqdm
from concurrent.futures import ThreadPoolExecutor
import requests
import matplotlib.pyplot as plt
import seaborn as sns

# --- Dependency Check ---
# These checks ensure necessary libraries are installed.
try:
    import openpyxl
except ImportError:
    st.error("ERROR: The 'openpyxl' library is required to generate Excel files. Please install it by running: pip install openpyxl")
    st.stop()

try:
    import boto3
    from botocore.config import Config
    from botocore.exceptions import NoCredentialsError
except ImportError:
    st.error("ERROR: The 'boto3' library is required for AWS S3 functionality. Please install it by running: pip install boto3")
    st.stop()

env_path = find_dotenv()
print("Loading .env from :", env_path)
load_dotenv(env_path)


google_distance_api_key = os.getenv("Google_distance_api_key")
locate_one_api_key = os.getenv("locate_one_api_key")

# ==============================================================================
# 2. SETUP & CONFIGURATION
# ==============================================================================

# --- Load Environment Variables ---
# @st.cache_resource
def load_env_vars():
    """Looks for a .env file and loads variables from it."""
    if find_dotenv():
        load_dotenv(find_dotenv())
        
        st.sidebar.success("✅ Environment variables loaded.")
    else:
        st.sidebar.warning("⚠️ No .env file found. Using Streamlit secrets.")

# --- Initialize LLM Client ---
# @st.cache_resource
def initialize_llm_client():
    """Initializes and returns the AzureOpenAI client if credentials are available."""
    azure_api_key = os.getenv("AZURE_API_KEY")
    azure_endpoint = os.getenv("AZURE_ENDPOINT")
    deployment_name = os.getenv("DEPLOYMENT_NAME")
    api_version = os.getenv("API_VERSION")
    

    if all([azure_api_key, azure_endpoint, deployment_name, api_version]):
        try:
            client = openai.AzureOpenAI(
                api_key=azure_api_key,
                api_version=api_version,
                azure_endpoint=azure_endpoint
            )
            return client, True, deployment_name
        except Exception as e:
            st.error(f"Failed to initialize Azure OpenAI client: {e}")
            return None, False, None
    st.sidebar.warning("Azure OpenAI credentials not found. LLM features will be disabled.")
    return None, False, None


# --- Central Application Configuration ---
def get_app_config():
    """Returns a dictionary containing all application configurations."""
    config = {
        "paths": {
            "vehicles_types": "/home/ubuntu/standard_vehicle_types.csv",
            "origin_ref_repo": "/home/ubuntu/origin_ref_repo.csv",
            "pincodes_repo": "/home/ubuntu/picodes_repo.csv",
            "pricing_ref_id": "/home/ubuntu/ref_id_file.csv",
            "ltl_prices": "/home/ubuntu/default_DLV_ltl_prices.csv",
            "pincode_mapping_db": "/home/ubuntu/Pincode_Mapping.csv",
        },

        "api": {
            "tms_url": os.getenv("TMS_API_URL"),
            "s3_bucket": os.getenv("S3_BUCKET"),
            "s3_region": os.getenv("S3_REGION"),
        },
        "order_cleaner": {
            "standard_output_columns": [
                'Origin Facility ID', 'Origin address', 'Origin address PIN code', 'origin_city',
                'Destination Facility ID', 'Destination address', 'Destination address PIN code', 'destination_city',
                'Client ID', 'Order Number', 'Order Date', 'Expected Delivery Date', 'Sold To Facility ID',
                'Movement', 'Customer Reference Number', 'Order Priority', 'Order Item No.', 'Product Code',
                'Product Name', 'Quantity', 'Unit of measurement', 'Weight', 'Volume', 'Weight Unit',
                'Volume Unit', 'Product MRP', 'Product Category', 'Invoice Number', 'Invoice Amount',
                'Invoice Date', 'Invoice Description', 'E-Way Bill Number', 'E-Way Bill Expiry Date',
                'IRN', 'Buyer GST', 'Seller GST', 'Transaction Type', 'Total_cost'
            ],
            "columns_always_null": [
                'Destination Facility ID', 'Client ID', 'Expected Delivery Date', 'Sold To Facility ID',
                'Customer Reference Number', 'Order Priority', 'Invoice Number', 'Invoice Amount',
                'Invoice Date', 'Invoice Description', 'E-Way Bill Number', 'E-Way Bill Expiry Date',
                'IRN', 'Buyer GST', 'Seller GST'
            ],
            "columns_calculated_or_hardcoded": ['Origin Facility ID', 'Movement', 'Weight Unit', 'Volume Unit'],
            "initial_critical_columns": [
                'Order Number', 'Order Date', 'Destination address', 'Destination address PIN code',
                'destination_city', 'Origin address', 'Origin address PIN code', 'origin_city',
                'Transaction Type', 'Product Code', 'Product Name', 'Quantity', 'Total_cost'
            ],
            "predefined_clients": ['BlueStar', 'FTPL', 'AJIO', 'Havells', 'Voltas', 'Other client'],
            "default_uom_options": ["-- Select Default (Optional) --", "BOX", "BUNDLE", "PCS"],
            "default_product_category_options": ["-- Select Default (Optional) --", "DRY", "WET"],
        },
        "pricing_cleaner": {
            "truck_types": [
                'Open 6 Tyre', 'Open 10 Tyre', 'Open 12 Tyre', 'Open 14 Tyre', 'Closed 34FT SXL',
                'Closed 34FT MXL', 'Closed 32FT SXL HQ', 'Closed 32FT MXL HQ', 'Closed 32FT SXL',
                'Closed 32FT MXL', 'Closed 24FT MXL', 'Closed 24FT SXL', 'Closed 22FT SXL',
                'Closed 22FT MXL', 'Closed 20FT SXL', 'Closed 19FT SXL', 'Closed 17FT SXL',
                'Closed 14FT SXL', 'Open 12FT SXL', 'Closed 28FT SXL', 'Tata Ace', 'Tata 407',
                'Bolero pickup', 'Flatbed 20FT Semi', 'Flatbed 40FT Semi'
            ],
            "generic_truck_keywords": ['truck type', 'vehicle type', 'type', 'vehicle name', 'vehicle model', 'equipment'],
            "standard_output_columns": [
                'Origin Ref ID', 'Origin Type', 'origin_city', 'origin_state',
                'Destination Ref ID', 'Destination Type', 'destination_city', 'destination_state',
                'Truck Type', 'Service Type', 'Rate Type', 'Rate Value', 'Transit Days'
            ],
            "llm_mapping_target_keys": {
                'origin_city': 'origin_city_Source', 'origin_state': 'origin_state_Source',
                'origin_pincode': 'origin_pincode_Source', 'destination_pincode': 'destination_pincode_Source',
                'destination_city': 'destination_city_Source', 'destination_state': 'destination_state_Source',
                'Truck Type': 'Truck Type_Source', 'Rate Value': 'Rate Value_Source',
                'Service Type': 'Service Type_Source'
            },
            "standard_states": [
                'andaman & nicobar', 'andhra pradesh', 'arunachal pradesh', 'assam', 'bihar',
                'chhattisgarh', 'chandigarh', 'daman & diu', 'delhi', 'dadra and nagar haveli',
                'goa', 'gujarat', 'himachal pradesh', 'haryana', 'jharkhand', 'jammu & kashmir',
                'karnataka', 'kerala', 'ladakh', 'lakshadweep', 'maharashtra', 'meghalaya',
                'manipur', 'madhya pradesh', 'mizoram', 'nagaland', 'orissa', 'punjab',
                'pondicherry', 'rajasthan', 'sikkim', 'tamil nadu', 'tripura', 'telangana',
                'uttarakhand', 'uttar pradesh', 'west bengal'
            ]
        },
        "product_cleaner": {
            "standard_output_columns": [
                'product_code', 'product_name', 'product_category', 'product_weight',
                'product_volume', 'max_stackable_quantities', 'max_stackable_weight',
                'product_length', 'product_breadth', 'product_height', 'product_orientation'
            ],
            "required_columns": [
                'product_code', 'product_name', 'product_length', 'product_breadth',
                'product_height', 'product_weight'
            ],
            "orientation_options": ["all", "upright"],
            "default_orientation": "all"
        }
    }
    oc_config = config["order_cleaner"]
    oc_config["standard_direct_mappable_columns"] = [
        col for col in oc_config["standard_output_columns"]
        if col not in oc_config["columns_always_null"] and col not in oc_config["columns_calculated_or_hardcoded"]
    ]
    oc_config["initial_critical_columns"] = [
        col for col in oc_config["initial_critical_columns"]
        if col in oc_config["standard_direct_mappable_columns"]
    ]
    return config

# --- Session State Initialization ---
def initialize_session_state():
    """Initializes all required session state variables for the combined app."""
    # General App State
    if 'page' not in st.session_state:
        st.session_state.page = "SCS - Standardiser"
    if 'config' not in st.session_state:
        st.session_state.config = get_app_config()

    # SCS State Variables
    for key, default_value in {
        'scs_client_name': "Halonix",
        'scs_product_types': "LED Bulbs, battens and Fans",
        "scs_dimension_unit": "inch",
        "scs_weight_unit": "kg",
        'scs_column_map': {},
        'scs_llm_response': None,
        'scs_processed_df': None,
        'uploader_key': 0,
        'scs_original_filename': None,
        'eda_ran': False,
        'pincode_map_button': None,
        'anomaly_button': None,
        'missing_value_button': None,
        'true_df': None,
    }.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

    # TMS State Variables
    for key, default_value in {
        'processed_orders_df': None,
        'processed_pricing_df': None,
        'processed_product_df': None,
        'initial_orders_df_for_reports': None,
        'tms_triggered': False,
        'req_id': None,
        'sub_ids_df': pd.DataFrame(),
        'plan_summaries': [],
        'route_summaries': [],
        'route_sequences': [],
        'failed_sub_ids': [],
        'direct_trigger_source_url': None,
        'tms_source_is_scs': False,
        'tms_input_from_scs': None,
        'pc_ref_id_df': None
    }.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

# ==============================================================================
# 3. SHARED UTILITY & HELPER FUNCTIONS
# ==============================================================================

# @st.cache_data
def load_csv_from_upload(uploaded_file):
    """Loads an uploaded CSV into a pandas DataFrame."""
    if uploaded_file is None:
        return None
    try:
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"❌ Error reading CSV file '{uploaded_file.name}': {e}")
        return None

# @st.cache_data
def load_csv_from_path(file_path):
    """Loads a CSV from a local file path."""
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"Error: The file `{file_path}` was not found. Please ensure it exists in the same directory.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading '{file_path}': {e}")
        return pd.DataFrame()

def parallel_apply_with_progress(func, data, desc="Processing..."):
    """Applies a function to data in parallel with a progress bar."""
    if not data:
        return []
    with ThreadPoolExecutor() as executor:
        results = list(stqdm(executor.map(func, data), total=len(data), desc=desc))
    return results

def missing_col_null_check(df_input,expected_columns):
        existing_columns = list(df_input.columns)
        missing_columns = [col for col in expected_columns if col not in existing_columns]
        present_columns = [col for col in expected_columns if col in existing_columns]
        stats = []

        for col in present_columns:
            null_count = int(df_input[col].isna().sum())
            stats.append((col,  null_count))

        return {"columns not available": missing_columns, "null count in the columns": stats}

def add_velocity_column(df):
    """Calculates and adds the SKU velocity column to the DataFrame."""
    if 'quantity' in df.columns and 'product_code' in df.columns:
        # Create a copy to avoid modifying the original DataFrame unexpectedly
        df_processed = df.copy()

        # Ensure 'quantity' is a numeric type for calculation
        df_processed['quantity'] = pd.to_numeric(df_processed['quantity'], errors='coerce').fillna(0)

        # Group by product code and calculate sum of quantities
        skus_quantity_df = df_processed.groupby('product_code')['quantity'].sum().reset_index()

        # Calculate velocity only if there's a total quantity to analyze
        total_qty = skus_quantity_df['quantity'].sum()
        if total_qty > 0:
            skus_quantity_df = skus_quantity_df.sort_values(by='quantity', ascending=False)
            skus_quantity_df['cum_percent'] = 100 * skus_quantity_df['quantity'].cumsum() / total_qty
            skus_quantity_df['velocity'] = skus_quantity_df['cum_percent'].apply(
                lambda x: 'Fast' if x <= 80 else ('Medium' if x <= 96 else 'Slow')
            )
            
            # Merge the calculated velocity back into the main DataFrame
            sku_velocity_tag = skus_quantity_df[['product_code', 'velocity']]
            df_processed = df_processed.merge(sku_velocity_tag, on='product_code', how='left')
            return df_processed
    
    # Return the original DataFrame if required columns are not present
    return df


# ==============================================================================
# 4. SCS - STANDARDISER: FUNCTION DEFINITIONS
# ==============================================================================

# --- LLM & Data Processing Functions ---
def get_llm_suggestions(client, DEPLOYMENT_NAME, column_names):
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


def get_tms_columns_llm_suggestions(client, DEPLOYMENT_NAME, column_names):
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
    else:
        df['product_desc'] = ''

    return df
    
    # else:
    # return df

def mapping_and_missing_df_create(df):
    if 'product_code' not in df.columns:
        df['product_code'] = ''
    df['sku_volume'] = df['length'].astype(float) * df['width'].astype(float) * df['height'].astype(float)
    # df['sku_volume'] = df['length'].astype(float) * df['width'].astype(float) * df['height'].astype(float)
    mapping_df = df[(df['Values_Added'] == True) & (df['sku_volume'] > 0) & (df['product_weight'] != '0.0') & (df['product_weight']!='#N/A ()')].reset_index(drop=True)
    to_be_mapped_df = df[~((df['Values_Added'] == True) & (df['sku_volume'] > 0) & (df['product_weight'] != '0.0') & (df['product_weight']!='#N/A ()'))].reset_index(drop=True)
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

def eda_summary_agent(client, DEPLOYMENT_NAME, eda_json):
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
        
        df_input['quantity'] = pd.to_numeric(df_input['quantity'], errors='coerce')
        df_input['quantity'].fillna(0, inplace=True) # Recommended if missing quantities should be zero

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
        
        # --- MODIFIED LOGIC FOR SINGLE CITY/STATE ---
        if len(city_demand) == 1:
            top_80_per_city = city_demand.copy() # If only one city, it's always "top 80%" (100%)
            city_demand['pareto_class'] = 'Top 80%' # Assign Pareto class explicitly
        else:
            city_demand['pareto_class'] = city_demand['cum_percent'].apply(lambda x: 'Top 80%' if x <= 80 else 'Bottom 20%')
            top_80_per_city = city_demand[city_demand['pareto_class'] == 'Top 80%']
        # --- END MODIFIED LOGIC ---

        city_demand.reset_index(drop=True, inplace=True)


    if 'state' in df_input.columns and 'quantity' in df_input.columns:
        state_demand = df_input.groupby('state')['quantity'].sum().reset_index()
        state_demand = state_demand.sort_values(by='quantity', ascending=False)
        total_qty = state_demand['quantity'].sum()
        state_demand['qty_percent'] = 100 * state_demand['quantity'] / total_qty
        state_demand['cum_quantity'] = state_demand['quantity'].cumsum()
        state_demand['cum_percent'] = 100 * state_demand['cum_quantity'] / total_qty

        # --- MODIFIED LOGIC FOR SINGLE CITY/STATE ---
        if len(state_demand) == 1:
            top_80_per_state = state_demand.copy() # If only one state, it's always "top 80%" (100%)
            state_demand['pareto_class'] = 'Top 80%' # Assign Pareto class explicitly
        else:
            state_demand['pareto_class'] = state_demand['cum_percent'].apply(lambda x: 'Top 80%' if x <= 80 else 'Bottom 20%')
            top_80_per_state = state_demand[state_demand['pareto_class'] == 'Top 80%']
        # --- END MODIFIED LOGIC ---

        state_demand.reset_index(drop=True, inplace=True)

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

def order_summary_agent(client, DEPLOYMENT_NAME, eda_json):
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

def eda_agent(client, DEPLOYMENT_NAME, df_input):
    order_columns = ['invoice_no','order_id','quantity','date']
    skus_columns = ['product_code', 'product_desc','product_dimension', 'length', 'width', 'height', 'pack_size', 'product_weight']
    sku_col_available = [col for col in df_input.columns if col in skus_columns]

    eda_summary_output = None
    order_summary_output = None

    # Check for order-related columns and generate order summary if applicable
    if any(col in df_input.columns for col in order_columns):
        st.write("Input data rows count : ", df_input.shape[0])
        # Pass a copy of df_input to order_summ_fun to avoid modifying the original DataFrame
        df_for_order_summary, order_summary_dict, orderline_per_day = order_summ_fun(df_input.copy())
        order_summary_output = order_summary_agent(client, DEPLOYMENT_NAME, order_summary_dict)
        st.subheader("Order Summary Agent Output")
        st.write(order_summary_output) 
        st.markdown("### **Order Trends**")
        sales_data_plots(df_for_order_summary, orderline_per_day)
    # No else block here, as we want to check for SKU columns independently.

    # Check for SKU-related columns and generate EDA summary if applicable
    if sku_col_available == ['product_code']:
        print('only product_code columns is present')
    else:
        df_for_sku_eda = process_dataframe(df_input.copy()) # Process a copy
        true_df, false_df = mapping_and_missing_df_create(df_for_sku_eda)
        st.write("Input data rows count : ", df_input.shape[0])
        st.write("Filtered data rows with available weight and dimension : ", true_df.shape[0])
        summary_df = check_value_counts(true_df)
        eda_df_json = summary_df.to_dict(orient="records")
        eda_summary_output = eda_summary_agent(client, DEPLOYMENT_NAME, eda_df_json)
        print(eda_summary_output) # For debugging

        st.subheader("EDA Agent Output")
        st.write(eda_summary_output)
        st.markdown("### **Frequency distributions of the columns**")
        plot_histogram_subplots(true_df, 20)
        st.session_state.true_df = true_df # This might be used elsewhere, keep it
    # No else block here, as we want to check for Order columns independently.

    st.session_state.eda_ran = True
    return eda_summary_output, order_summary_output

    
def eda_agent_custom_column(client, DEPLOYMENT_NAME, df_input):  
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
    eda_summary_agent_output = eda_summary_agent(client, DEPLOYMENT_NAME, eda_df_json)
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

def get_internal_pincode_db():
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
        df_pincode = df_pincode[df_pincode['mapped_pincode'].notna()]
        df_pincode["mapped_pincode"] = df_pincode["mapped_pincode"].astype(int).astype(str)
        df_pincode["Latitude"] = df_pincode["Latitude"].astype(str)
        df_pincode["Longitude"] = df_pincode["Longitude"].astype(str)
        df_pincode["mapped_city_db"] = df_pincode["mapped_city_db"].str.lower().fillna("Unknown")
        df_pincode["mapped_state_db"].fillna("Unknown", inplace=True)
        return df_pincode
    except FileNotFoundError:
        st.error("Error: Pincode_Mapping.csv file not found.")
        return pd.DataFrame()

    
def get_most_probable_pincode(client, DEPLOYMENT_NAME, row):
    """Use Azure LLM to get the most probable pincode for a given Indian city."""
    city = row['city']
    state = row['state'] if 'state' in row and pd.notna(row['state']) else None

    if state:
        state = row['state']
        prompt = (
            f"Given the city '{city}' of the {state} state in India, return the most commonly used 6-digit Indian pincode for this city. "
            "The response MUST follow strict JSON format with double quotes: "
            "{\"mapped_pincode\": \"valid_pincode\"}. "
            "You MUST always return a valid 6-digit pincode—never leave it empty or return 'Unknown'."
        )
    else:
        prompt = (
            f"Given the city '{city}' in India, return the most commonly used 6-digit Indian pincode for this city. "
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
    
def get_correct_pincode_or_city(client, DEPLOYMENT_NAME, pincode):
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
    
def get_lat_long(client, DEPLOYMENT_NAME, row):
    """Use Azure LLM to get the most probable lat long for a given Indian pincode and city."""
    pincode = row["mapped_pincode"]
    tat_city = row["mapped_city_tat"]
    state_code = row["mapped_state_db"]
    prompt = (
        f'''Given the pincode: '{pincode}', city: '{tat_city}', state code: '{state_code}' in India,
         return the most nearest Latitude and Longitude for the given indian pincode, city, and state code combination.
         prioritize the pincode first, then city and state, as I have seen you gives same lat long for different pincodes
        The response MUST follow strict JSON format with double quotes: 
       {{'Latitude':'latitude',
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
import requests
import json
import time
import pandas as pd

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
def pincode_mapping_agent(client, DEPLOYMENT_NAME, df_input):
    import pandas as pd
    df_pincode = get_internal_pincode_db()

    st.write('Mapping Input data to pincode, city and state from Internal Database ...')

    rename_map = {
    "pincode_origin": "pincode",
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

    # st.write('df_input', df_input)
    # st.write('df_input dtype', df_input.dtypes)

    # df_input['pincode'] = df_input['pincode'].astype('Int64')

    df_input['pincode'] = (pd.to_numeric(df_input['pincode'], errors='coerce').fillna(0).astype('Int64'))

    # st.write('df_input dtype after ', df_input.dtypes)

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
                
                    probable_pincode = get_most_probable_pincode(client, DEPLOYMENT_NAME, row) 
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
                correction = get_correct_pincode_or_city(client, DEPLOYMENT_NAME, row["pincode"])
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

    # print("Fetching Latitude and Longitude through LLM...")
        
    # rows = [row for index, row in df1.iterrows()]

    #     # Use parallel_apply_with_progress function instead of swifter.apply
    # lat_long_output =  parallel_apply_with_progress(lambda row: get_lat_long(client, DEPLOYMENT_NAME, row), rows)

    # # st.write("llm Response output", lat_long_output)

    # latitudes = []
    # longitudes = []

    # # Iterate through the results array and extract the values
    # for res in lat_long_output:
    #     latitudes.append(res[0])  # Extract latitude
    #     longitudes.append(res[1]) # Extract longitude


    # # # Assign the latitudes and longitudes to the DataFrame columns
    # df1['Latitude'] = latitudes
    # df1['Longitude'] = longitudes

    st.write("Fetching Latitude and Longitude through Delhivery's LocateOne API...")
        
    rows = [row for index, row in df1.iterrows()]

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

    origin_pincode = row['mapped_pincode_origin_1']
    destination_pincode = row['mapped_pincode']

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
            return pd.Series([origin_pincode, destination_pincode, origin_address, destination_address, distance_km])
        else:
            return pd.Series([origin_pincode, destination_pincode, None, None, None])
    else:
        return pd.Series([origin_pincode, destination_pincode, None, None, None])

    
import requests
import pandas as pd
from math import ceil


def get_distance_batched(df,api_key, mode="driving", batch_size=10):

    base_url = "https://maps.googleapis.com/maps/api/distancematrix/json"
    results = []

    total_rows = len(df)
    num_batches = ceil(len(df) / batch_size)

    #### Streamlit UI elements
    progress_bar = st.progress(0)
    progress_text = st.empty()

    rows_processed = 0

    for b in range(num_batches):

        chunk = df.iloc[b*batch_size : (b+1)*batch_size]

        # Build coordinate lists
        origins = [f"{row['Latitude_origin_1']},{row['Longitude_origin_1']}" 
                   for _, row in chunk.iterrows()]
        destinations = [f"{row['Latitude']},{row['Longitude']}" 
                        for _, row in chunk.iterrows()]

        # Keep pincode identifiers aligned with same indexes
        origin_pins = chunk["mapped_pincode_origin_1"].tolist()
        destination_pins = chunk["mapped_pincode"].tolist()

        params = {
            "origins": "|".join(origins),
            "destinations": "|".join(destinations),
            "mode": mode,
            "units": "metric",
            "region": "in",
            "key": api_key        }


        try:
            response = requests.get(base_url, params=params, timeout=10)
            data = response.json()
            
        except:
            # Return None for all rows in this chunk
            for i in range(len(chunk)):
                results.append([
                    origin_pins[i],
                    destination_pins[i],
                    None, None, None
                ])

            rows_processed += len(chunk)
            progress_bar.progress(rows_processed / total_rows)
            progress_text.text(f"Fetched distance for {rows_processed} rows  out of {total_rows} rows")
            continue

        # If Google API fails for the whole request
        if data.get("status") != "OK":
            for i in range(len(chunk)):
                results.append([
                    origin_pins[i],
                    destination_pins[i],
                    None, None, None
                ])
            rows_processed += len(chunk)
            progress_bar.progress(rows_processed / total_rows)
            progress_text.text(f"Fetched distance for {rows_processed} rows  out of {total_rows} rows")
            continue

        rows = data.get("rows", [])
        origin_addr_list = data.get("origin_addresses", [])
        dest_addr_list = data.get("destination_addresses", [])

        # Extract diagonal elements (row i → destination i)
        for i in range(len(chunk)):

            try:
                element = rows[i]["elements"][i]
            except:
                results.append([origin_pins[i], destination_pins[i], None, None, None])
                continue

            if element.get("status") == "OK":
                distance_km = element["distance"]["value"] / 1000

                results.append([
                    origin_pins[i],
                    destination_pins[i],
                    origin_addr_list[i],
                    dest_addr_list[i],
                    distance_km
                ])
            else:
                results.append([
                    origin_pins[i],
                    destination_pins[i],
                    None, None, None
                ])

        ### Row counter update
        rows_processed += len(chunk)
        progress_bar.progress(rows_processed / total_rows)
        progress_text.text(f"Fetched distance for {rows_processed} rows out of {total_rows} rows")

    progress_text.text("Completed.")

    # Final output dataframe
    return pd.DataFrame(results, columns=[
        'mapped_pincode_origin_1','mapped_pincode','origin_address','destination_addresses', 'google_distance_km'])



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
def azure_llm_density_threshold(client, DEPLOYMENT_NAME, row, client_name, product_types):
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
            {{'detailed_product_description': 'product description'
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
                return pd.Series([None, None, None, None, None, None, None, None])

        except Exception as e:
            print(f"Error in azure_llm_density_threshold: {str(e)}")
            print(f"LLM output was: {llm_json_output}")
            return pd.Series([None, None, None, None, None, None, None, None])
        


conversion_factors = { #are these global variables?
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
    
    


def azure_llm_suggested_product_dimensions(client, DEPLOYMENT_NAME, row, client_name, product_types, weight_unit, dimension_unit):

    matched_from = None
    matched_material = None  # Store matched Material Code
    matched_description = None  # Store matched Material Description
    values_added = 'Not Mapped'   # Retain existing value from df_false
    gross_weight = row['product_weight']  # Retain existing Gross weight from df_false
    length = row['length']
    width = row['width']
    height = row['height']
    user_prod_desc = row['product_desc']
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

    except Exception as e:
        print(f"Error in azure_llm_suggested_product_dimensions: {str(e)}")
        print(f"LLM output was: {llm_json_output}")
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

def weight_anomaly_detection(client, DEPLOYMENT_NAME, df_input):
    # Initialize progress bar and text
    progress_bar = st.progress(0)
    status_text = st.empty()  

    # Process DataFrame
    df = process_dataframe(df_input)
    status_text.text("Dataframe Created for Processing...")
    progress_bar.progress(10)

    status_text.text("Web searching the density threshold for each product.. this can take a while...")
    progress_bar.progress(20)

    client_name = st.session_state.scs_client_name,
    product_types = st.session_state.scs_product_types

    df_unique_prod = df[['product_desc']].drop_duplicates().reset_index(drop=True)

    rows = [row for index, row in df_unique_prod.iterrows()]

    ###### Calling LLM to get weight and density thresholds
    density_threshold_output = parallel_apply_with_progress(lambda row: azure_llm_density_threshold(client, DEPLOYMENT_NAME, row,
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

def missing_weight_dimensions_estimator_agent(client, DEPLOYMENT_NAME, df_input):
    print(f"DEBUG: Starting missing_weight_dimensions_estimator_agent with df shape: {df_input.shape}")
    print(f"DEBUG: Input columns: {list(df_input.columns)}")
    
    # Initialize progress bar and text
    progress_bar = st.progress(0)
    status_text = st.empty()   

    ##### Cleaning input data
    try:
        df = process_dataframe(df_input)
        print(f"DEBUG: After process_dataframe, df shape: {df.shape}")
        print(f"DEBUG: Processed columns: {list(df.columns)}")
    except Exception as e:
        print(f"ERROR in process_dataframe: {str(e)}")
        return None
        
    status_text.text("Dataframe Created for Processing...")
    progress_bar.progress(20)

    status_text.text("Web searching the density threshold for each product.. this can take a while...")
    progress_bar.progress(10)

    ##### Create df with unique product description
    try:
        df_unique_prod = df[['product_desc']].drop_duplicates().reset_index(drop=True)
        print(f"DEBUG: Unique products shape: {df_unique_prod.shape}")
        print(f"DEBUG: Sample product descriptions: {df_unique_prod['product_desc'].head().tolist()}")
    except Exception as e:
        print(f"ERROR creating unique products df: {str(e)}")
        return None

    rows = [row for index, row in df_unique_prod.iterrows()]

    try:
        client_name = st.session_state.scs_client_name
        product_types = st.session_state.scs_product_types
        print(f"DEBUG: client_name: {client_name}, type: {type(client_name)}")
        print(f"DEBUG: product_types: {product_types}, type: {type(product_types)}")
    except Exception as e:
        print(f"ERROR getting session state variables: {str(e)}")
        return None

    ##### Calling LLM to get density & weight threshold for unique products
    try:
        print(f"DEBUG: About to call azure_llm_density_threshold for {len(rows)} rows")
        density_threshold_output = parallel_apply_with_progress(lambda row: azure_llm_density_threshold(client, DEPLOYMENT_NAME, row,
                                                client_name, product_types), rows)
        print(f"DEBUG: density_threshold_output length: {len(density_threshold_output) if density_threshold_output else 0}")
    except Exception as e:
        print(f"ERROR in azure_llm_density_threshold: {str(e)}")
        return None

    try:
        density_df = pd.DataFrame([
            item.tolist() if item is not None else [None] * 8
            for item in density_threshold_output], columns=[
            'input_prod_desc','detailed_product_description', 'product_category', 'density_unit',
            'density_lower_threshold', 'density_upper_threshold','weight_lower_threshold', 'weight_upper_threshold'])
        print(f"DEBUG: density_df shape: {density_df.shape}")
        print(f"DEBUG: density_df columns: {list(density_df.columns)}")
        print(f"DEBUG: Sample density_df:\n{density_df.head()}")
    except Exception as e:
        print(f"ERROR creating density_df: {str(e)}")
        return None

    try:
        df = df.merge(density_df, left_on='product_desc', right_on='input_prod_desc', how='left').drop(columns =['input_prod_desc'])
        print(f"DEBUG: After merge, df shape: {df.shape}")
        
        # Add Values_Added column if it doesn't exist - this is critical!
        if 'Values_Added' not in df.columns:
            print("DEBUG: Adding missing 'Values_Added' column")
            # Check if we have complete data (non-null dimensions and weight)
            df['Values_Added'] = ((df['length'].notna() & (df['length'] != '') & (df['length'] != '0.0')) & 
                                  (df['width'].notna() & (df['width'] != '') & (df['width'] != '0.0')) & 
                                  (df['height'].notna() & (df['height'] != '') & (df['height'] != '0.0')) & 
                                  (df['product_weight'].notna() & (df['product_weight'] != '') & (df['product_weight'] != '0.0')))
        
        print(f"DEBUG: Values_Added value counts:\n{df['Values_Added'].value_counts()}")
        
    except Exception as e:
        print(f"ERROR merging density_df: {str(e)}")
        return None
    
    try:
        true_df, false_df = mapping_and_missing_df_create(df)
        print(f"DEBUG: true_df shape: {true_df.shape}, false_df shape: {false_df.shape}")
    except Exception as e:
        print(f"ERROR in mapping_and_missing_df_create: {str(e)}")
        return None
    print('input df', df.shape)
    print('Mapping dataset ', true_df.shape)
    print('Data to map', false_df.shape)

    # Creating the column for density by assuming dimension unit as (mm, cm, m, inch, ft)
    try:
        print(f"DEBUG: About to compute density. true_df shape: {true_df.shape}")
        if len(true_df) > 0:
            for unit in conversion_factors:
                true_df[f'density_{unit}'] = compute_density(true_df, unit)
            print(f"DEBUG: Added density columns. true_df shape: {true_df.shape}")
            
            true_df['guessed_unit'] = true_df.apply(best_guess_unit, axis=1) #guessing the dimension unit based on density
            print(f"DEBUG: Added guessed_unit column")
            
            true_df_true = true_df[true_df['guessed_unit']!= 'unknown']
            true_df_false = true_df[true_df['guessed_unit'] == 'unknown'] #Move the rows which are not in the density range to False
        else:
            print("DEBUG: true_df is empty, skipping density computation")
            true_df_true = true_df.copy()
            true_df_false = pd.DataFrame()
    except Exception as e:
        print(f"ERROR in density computation: {str(e)}")
        # Continue with empty dataframes if density computation fails
        true_df_true = pd.DataFrame()
        true_df_false = true_df.copy() if len(true_df) > 0 else pd.DataFrame()


    print('true_df within density range',true_df_true.shape)
    print('true_df outside density range',true_df_false.shape)

    false_df_combined = pd.concat([false_df, true_df_false], ignore_index=True) # combined outside density range true_df to false_df

    print('overall false_df_combined rows',false_df_combined.shape)
    status_text.text("Finding best match in mapping file this can take a while...")
    progress_bar.progress(30)

    rows = [row for index, row in false_df_combined.iterrows()]
    k1 = false_df_combined.copy()

    ##### 1. Finding best match for anomolous values within golden dataset
    best_match_output = parallel_apply_with_progress(lambda row: find_best_match(row,true_df_true), rows)

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
    client_name = st.session_state.scs_client_name
    product_types = st.session_state.scs_product_types
    weight_unit = st.session_state.scs_weight_unit
    dimension_unit = st.session_state.scs_dimension_unit
    llm_output_lbh = pd.DataFrame(parallel_apply_with_progress(lambda row: azure_llm_suggested_product_dimensions(client, DEPLOYMENT_NAME, row,
                        client_name, product_types, weight_unit , dimension_unit ), rows))

    false_df_2_process[['predicted_length', 'predicted_width', 'predicted_height', 'predicted_unit','predicted_weight', 'predicted_pack_size',
                        'Values_Added','Matched_From', 'Matched_Material', 'Matched_Description', 'Match_confidence']] = llm_output_lbh

    try:
        true_df_true['Values_Added'] = 'Dimensions and Weight present' ## Golden dataset
        false_df_2_skip['Values_Added']= 'NA' ## Unmapped dataset by LLM and internal mapping
        
        print(f"DEBUG: Final concat - true_df_true: {true_df_true.shape}, false_df_mapped: {false_df_mapped.shape}")
        print(f"DEBUG: false_df_2_process: {false_df_2_process.shape}, false_df_2_skip: {false_df_2_skip.shape}")
        
        df_combined = pd.concat([true_df_true, false_df_mapped, false_df_2_process, false_df_2_skip], ignore_index=True)
        df=df_combined
        
        print(f"DEBUG: Final df_combined shape: {df_combined.shape}")
        print(f"DEBUG: Final df columns: {list(df.columns)}")
        
        # columns_to_fill = ['Matched_From', 'Matched_Material', 'Matched_Description', 'Match_confidence']
        # df[columns_to_fill] = df[columns_to_fill].replace('', 'NA').fillna('NA')
        
        status_text.text("Process completed successfully!")
        progress_bar.progress(100)
        
        print(f"DEBUG: Function returning df with shape: {df.shape}")
        return df
        
    except Exception as e:
        print(f"ERROR in final processing: {str(e)}")
        import traceback
        print(f"DEBUG: Full traceback: {traceback.format_exc()}")
        return None

# Test function to debug the missing_weight_dimensions_estimator_agent
def test_missing_weight_dimensions_estimator_agent():
    """
    Simple test function to debug the missing_weight_dimensions_estimator_agent.
    Use this to test with a minimal dataset.
    """
    import pandas as pd
    
    # Create a simple test dataframe
    test_df = pd.DataFrame({
        'product_desc': ['Test Product 1', 'Test Product 2', 'not available'],
        'length': ['', '', ''],
        'width': ['', '', ''],
        'height': ['', '', ''],
        'product_weight': ['', '', '']
    })
    
    print("DEBUG: Created test dataframe:")
    print(test_df)
    print(f"Shape: {test_df.shape}")
    print(f"Columns: {list(test_df.columns)}")
    
    # You would call this function like:
    # result = missing_weight_dimensions_estimator_agent(client, DEPLOYMENT_NAME, test_df)
    # print(f"Result: {result}")
    
    return test_df

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

def join_all_df(df1, df2, df3, original_df_columns):
                dataframes_to_join = [df for df in [df1, df2, df3] if df is not None]
                if not dataframes_to_join: return None
                if len(dataframes_to_join) == 1: return dataframes_to_join[0]
                result = dataframes_to_join[0].fillna('0')
                joininig_cols = [col for col in original_df_columns if col in result.columns]
                result = clean_df(result, joininig_cols)
                for df_to_merge in dataframes_to_join[1:]:
                    df_to_merge = clean_df(df_to_merge.fillna('0'), joininig_cols)
                    common_cols = list(set(result.columns) & set(df_to_merge.columns) - set(joininig_cols))
                    result = pd.merge(result, df_to_merge.drop(columns=common_cols, errors='ignore'), on=joininig_cols, how='left')
                return result

# ==============================================================================
# 5. TMS WORKFLOW: FUNCTION DEFINITIONS
# ==============================================================================
# ORDER CLEANER: FUNCTIONS
def oc_get_llm_mapping_suggestions(input_cols, config, llm_client, llm_deployment_name, client_name):
    if not llm_client:
        return {'mapped_columns': {}}
    prompt = f"""
    You are a data mapping assistant. Map standard output columns to input CSV columns based on semantic similarity for client "{client_name}".
    Prioritize exact or very close matches. Respond ONLY with a JSON object with a key `mapped_columns` (dict of standard_col: input_col).
    Input columns must exist in the provided list.
    Standard Output Columns to Map: {json.dumps(config["standard_direct_mappable_columns"])}
    Input CSV Columns: {json.dumps(input_cols)}
    Return JSON object ONLY: ```json {{"mapped_columns": {{...}} }} ```
    """
    try:
        with st.spinner(f"🔮 Calling Azure OpenAI for column mapping..."):
            completion = llm_client.chat.completions.create(
                model=llm_deployment_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                response_format={"type": "json_object"}
            )
        result = json.loads(completion.choices[0].message.content)
        validated_mapped = {
            k: v for k, v in result.get("mapped_columns", {}).items()
            if k in config["standard_direct_mappable_columns"] and v in input_cols
        }
        return {'mapped_columns': validated_mapped}
    except Exception as e:
        st.warning(f"LLM mapping failed: {e}. Proceeding with manual mapping.")
        return {'mapped_columns': {}}

def oc_standardize_dataframe(df_input, mapping, defaults, client_name, config):
    st.info("🛠️ Initializing standardized DataFrame structure...")
    df = pd.DataFrame(index=df_input.index, columns=config["standard_output_columns"])
    for out_col, in_col in mapping.items():
        if out_col in df.columns and in_col in df_input.columns:
            df[out_col] = df_input[in_col]
    for col in config["columns_always_null"]:
        df[col] = pd.NA
    for col, value in defaults.items():
        df[col] = value
    df['Weight Unit'] = 'KG'
    df['Volume Unit'] = 'CFT'
    if 'origin_city' in mapping:
        city_series = df_input[mapping['origin_city']].astype(str).str.strip().str.lower()
        df['Origin Facility ID'] = city_series.str[:3].str.upper() + client_name[:3].upper() + '1'
    if 'Transaction Type' in mapping:
        tt_series = df_input[mapping['Transaction Type']].astype(str).str.lower()
        conditions = [
            tt_series.str.contains('sto|stock transfer order', na=False),
            tt_series.str.contains('sales|sales order', na=False)
        ]
        df['Movement'] = np.select(conditions, ['Primary', 'Secondary'], default=pd.NA)
    st.success("✅ Standardized DataFrame structure initialized.")
    return df

def oc_enrich_product_details_with_llm(_df, client_name, llm_client, llm_deployment_name):
    if not llm_client: return _df
    df = _df.copy()
    missing_mask = (df['Weight'].isna() | df['Volume'].isna()) & df['Product Code'].notna() & df['Product Name'].notna()
    if not missing_mask.any(): return df
    unique_products = df.loc[missing_mask, ['Product Code', 'Product Name']].drop_duplicates()
    
    # Add the guard rail check here
    if len(unique_products) > 6000:
        st.warning(f"⚠️ Too many products for LLM enrichment: {len(unique_products)}. Skipping LLM call for product details.")
        return df
    
    st.info(f"Found {len(unique_products)} unique products needing Weight/Volume. Querying LLM...")
    progress_bar = st.progress(0, text=f"Fetching product details... 0/{len(unique_products)}")
    product_details_map = {}
    for i, row_dict in enumerate(unique_products.to_dict('records')):
        code, name = row_dict['Product Code'], row_dict['Product Name']
        prompt = f"""Estimate weight in kg and volume in cubic feet for the product.
        Product Code: {code}, Product Name: {name}, Client: {client_name}
        Respond ONLY with a JSON object: {{"Product Code": "{code}", "Weight (kg)": value, "Volume (cft)": value}}"""
        try:
            completion = llm_client.chat.completions.create(
                model=llm_deployment_name, messages=[{"role": "user", "content": prompt}],
                temperature=0.1, response_format={"type": "json_object"}
            )
            data = json.loads(completion.choices[0].message.content)
            if data.get("Product Code") == code:
                product_details_map[code] = {
                    'Weight': pd.to_numeric(data.get('Weight (kg)'), errors='coerce'),
                    'Volume': pd.to_numeric(data.get('Volume (cft)'), errors='coerce')
                }
        except Exception: pass
        progress_bar.progress((i + 1) / len(unique_products), text=f"Fetching product details... {i+1}/{len(unique_products)}")
    progress_bar.empty()
    if not product_details_map: return df
    details_df = pd.DataFrame.from_dict(product_details_map, orient='index')
    df = df.set_index('Product Code'); df.update(details_df); df = df.reset_index()
    st.success(f"✅ LLM enrichment complete. Updated {len(details_df)} products.")
    return df

def oc_perform_final_transforms(_df):
    df = _df.copy()
    st.info("🔢 Performing final transformations...")
    if 'Order Number' in df.columns:
        df['Order Item No.'] = df.groupby('Order Number').cumcount() + 1
    if 'Order Date' in df.columns:
        df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce').dt.strftime('%Y-%m-%d')
    st.success("✅ Final transformations applied.")
    return df

def oc_validate_and_enrich_data(_df, config):
    df = _df.copy()
    st.info("✨ Performing data validation and enrichment...")
    origin_ref_dict, pincodes_set = {}, set()
    try:
        origin_ref_df = pd.read_csv(config["paths"]["origin_ref_repo"])
        origin_ref_dict = origin_ref_df.set_index('Ref ID')['pin code'].astype(str).to_dict()
    except FileNotFoundError:
        st.warning(f"Origin reference file not found at '{config['paths']['origin_ref_repo']}'. Origin validation will be skipped.")
    try:
        pincodes_df = pd.read_csv(config["paths"]["pincodes_repo"])
        pincodes_set = set(pincodes_df['zipCode'].astype(str))
    except FileNotFoundError:
        st.warning(f"Pincodes repository file not found at '{config['paths']['pincodes_repo']}'. Destination PIN validation will be skipped.")
    
    if origin_ref_dict:
        def validate_origin(row):
            origin_id = str(row.get('Origin Facility ID', '')).upper()
            pin = str(row.get('Origin address PIN code', ''))
            if not origin_id or not pin or origin_id == 'NAN' or pin == 'NAN': return 'Missing Data'
            if origin_id in origin_ref_dict:
                return 'Valid' if origin_ref_dict[origin_id] == pin else f'Invalid PIN (Expected: {origin_ref_dict[origin_id]})'
            return 'Invalid ID'
        df['Origin_Validation_Status'] = df.apply(validate_origin, axis=1)
    else:
        df['Origin_Validation_Status'] = 'Skipped - No Ref File'
    
    null_addresses = df['Destination address'].isna().sum()
    if null_addresses > 0:
        df['Destination address'].fillna("Simulator Building, Shastri Park", inplace=True)
        st.write(f"✓ Replaced {null_addresses} NULL 'Destination address' entries.")
    
    if pincodes_set:
        df['pincode_is_valid'] = df['Destination address PIN code'].astype(str).isin(pincodes_set).astype(int)
    else:
        df['pincode_is_valid'] = 'Skipped - No Ref File'
    st.success("✅ Validation and enrichment complete.")
    return df

def oc_calculate_total_weight_volume(_df):
    df = _df.copy()
    try:
        df['Weight'] = pd.to_numeric(df['Weight'], errors='coerce') * pd.to_numeric(df['Quantity'], errors='coerce')
        df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce') * pd.to_numeric(df['Quantity'], errors='coerce')
        st.success("✅ Total Weight and Volume calculated.")
    except Exception as e:
        st.error(f"Could not calculate total weight/volume: {e}")
    return df

# PRICING CLEANER: FUNCTIONS
# @st.cache_data
def pc_load_reference_data(file_path):
    try:
        ref_df = pd.read_csv(file_path)
        ref_df['State'] = ref_df['State'].str.lower().str.strip()
        ref_df['City'] = ref_df['City'].str.lower().str.strip()
        return ref_df
    except FileNotFoundError:
        st.error(f"🚨 CRITICAL: Pricing Reference ID file not found at '{file_path}'. Ref ID mapping will fail.")
        return None
    except Exception as e:
        st.error(f"🚨 CRITICAL: Failed to load or process pricing reference ID file: {e}")
        return None

# @st.cache_data
def pc_load_default_ltl_prices(file_path):
    try:
        df_default_ltl = pd.read_csv(file_path)
        df_default_ltl = df_default_ltl.drop_duplicates(subset=['origin_state', 'destination_state'])
        df_default_ltl['origin_state'] = df_default_ltl['origin_state'].astype(str).str.lower().str.strip()
        df_default_ltl['destination_state'] = df_default_ltl['destination_state'].astype(str).str.lower().str.strip()
        return df_default_ltl
    except FileNotFoundError:
        st.error(f"🚨 Default LTL prices file not found at '{file_path}'. Default LTL rates will not be added.")
        return None
    except Exception as e:
        st.error(f"🚨 Error loading or processing default LTL prices: {e}")
        return None

def pc_get_llm_suggestions(llm_client, deployment_name, function, *args):
    if not llm_client: return None
    try: return function(llm_client, deployment_name, *args)
    except Exception as e: st.error(f"An error occurred during an LLM call: {e}"); return None

# @st.cache_data
def pc_get_llm_column_mapping_suggestions(_client, deployment_name, headers, target_keys_dict):
    prompt = f"CSV headers:\n{headers}\n\nMap to logical fields:\n{list(target_keys_dict.keys())}\n\nReturn JSON mapping logical fields to CSV headers (use empty string or null for no match)."
    response = _client.chat.completions.create(
        model=deployment_name, messages=[{"role": "system", "content": "You are a data mapping assistant. Output valid JSON."}, {"role": "user", "content": prompt}],
        temperature=0.1, response_format={"type": "json_object"}
    )
    suggestions = json.loads(response.choices[0].message.content)
    return {suffix: suggestions.get(disp_name) for disp_name, suffix in target_keys_dict.items()}

# @st.cache_data
def pc_get_llm_wide_format_detection(_client, deployment_name, column_names, df_sample_data):
    sample_data_str = df_sample_data.to_string(index=False) if not df_sample_data.empty else "N/A"
    prompt = f"Column headers:\n{column_names}\n\nSample data:\n{sample_data_str}\n\nIdentify columns where the header is a specific vehicle type and values are rates (wide format). Do NOT flag generic ID/data columns. Return JSON mapping column headers to 0 (no) or 1 (yes)."
    response = _client.chat.completions.create(
        model=deployment_name, messages=[{"role": "system", "content": "You analyze CSV structure. Output valid JSON."}, {"role": "user", "content": prompt}],
        temperature=0.0, response_format={"type": "json_object"}
    )
    column_flags = json.loads(response.choices[0].message.content)
    return {col: (1 if column_flags.get(col) == 1 else 0) for col in column_names}

# @st.cache_data
def pc_get_llm_batch_truck_type_mapping(_client, deployment_name, client_truck_types_list, standard_truck_types_list):
    client_types_str = "\n".join(f'- "{ct}"' for ct in client_truck_types_list)
    standard_types_str = "\n".join(f'- "{st}"' for st in standard_truck_types_list)
    prompt = f"Client truck types:\n{client_types_str}\n\nStandard truck types:\n{standard_types_str}\n\nMap each client type to the closest matching standard type. You must select a type from the standard list. Return a valid JSON object mapping client types to standard types."
    response = _client.chat.completions.create(
        model=deployment_name, messages=[{"role": "system", "content": "You are a vehicle type mapping assistant. Output valid JSON."}, {"role": "user", "content": prompt}],
        temperature=0.0, response_format={"type": "json_object"}
    )
    mapping_dict = json.loads(response.choices[0].message.content)
    return {ct: mapping_dict.get(ct) for ct in client_truck_types_list if mapping_dict.get(ct) in standard_truck_types_list}

# @st.cache_data
def pc_get_llm_state_mapping(_client, deployment_name, user_states_list, standard_states_list):
    user_states_str = "\n".join(f'- "{s}"' for s in user_states_list)
    standard_states_str = ", ".join(standard_states_list)
    prompt = f"Map messy state names to a standard list.\n\nUser's states:\n{user_states_str}\n\nMap each to one of these standard lowercase names:\n{standard_states_str}\n\nProvide a JSON object mapping original state names to standard names. If no match, map to null. Return ONLY the valid JSON object."
    response = _client.chat.completions.create(
        model=deployment_name, messages=[{"role": "system", "content": "You are a data mapping assistant for Indian geography. Output valid JSON."}, {"role": "user", "content": prompt}],
        temperature=0.0, response_format={"type": "json_object"}
    )
    mapping_dict = json.loads(response.choices[0].message.content)
    for state in user_states_list:
        if state not in mapping_dict:
            mapping_dict[state] = state.lower()
    return mapping_dict

# @st.cache_data
def prod_get_llm_column_mapping_suggestions(_client, deployment_name, input_columns, standard_columns):
    """Get LLM suggestions for mapping input columns to standard product columns."""
    input_cols_str = "\n".join(f'- "{col}"' for col in input_columns)
    standard_cols_str = "\n".join(f'- "{col}"' for col in standard_columns)
    
    prompt = f"""You are a data mapping assistant for product information. Map the user's column headers to standard product columns.

User's columns:
{input_cols_str}

Standard product columns:
{standard_cols_str}

Instructions:
1. Map each user column to the most appropriate standard column
2. If no good match exists, map to null
3. Each standard column should be mapped to at most one user column
4. Focus on semantic meaning, not exact text match
5. Common mappings:
   - product_code: SKU, item_code, product_id, code, sku_code
   - product_name: name, item_name, product_description, description, title
   - product_weight: weight, wt, mass, product_wt
   - product_length: length, len, l, product_length
   - product_breadth: breadth, width, w, product_width, product_breadth
   - product_height: height, h, product_height
   - product_volume: volume, vol, cubic_volume
   - product_category: category, type, product_type, cat
   - max_stackable_quantities: max_stack, stackable_qty, stack_limit
   - max_stackable_weight: max_stack_weight, stackable_weight
   - product_orientation: orientation, position, placement

Return a valid JSON object mapping standard columns to user columns (or null if no match).
"""
    
    try:
        response = _client.chat.completions.create(
            model=deployment_name,
            messages=[
                {"role": "system", "content": "You are a data mapping assistant. Output valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        mapping_dict = json.loads(response.choices[0].message.content)
        return mapping_dict
    except Exception as e:
        st.warning(f"⚠️ LLM mapping failed: {e}. Proceeding with manual mapping.")
        return {}

def pc_transform_wide_to_long(df_raw, llm_col_flags):
    identified_wide_cols = [col for col, flag in llm_col_flags.items() if flag == 1]
    if not identified_wide_cols: return df_raw, False
    id_vars = [h for h in df_raw.columns if h not in identified_wide_cols]
    if not id_vars:
        st.error("🚨 LLM identified all columns as truck types; no ID columns left for transformation.")
        return pd.DataFrame(), True
    df_melted = df_raw.melt(id_vars=id_vars, value_vars=identified_wide_cols,
                            var_name='Original Truck Header', value_name='Rate Value')
    df_melted.dropna(subset=['Rate Value'], inplace=True)
    df_melted = df_melted[df_melted['Rate Value'].astype(str).str.strip() != '']
    return df_melted, True

def pc_find_ref_id(city, state, ref_df, pincode=None):
    """Find reference ID with pincode-based mapping as primary method, fallback to city/state."""
    if ref_df is None: return None
    
    
    # Primary method: Use pincode if available and Pin_code column exists
    if pincode and 'Pin_code' in ref_df.columns:
        try:
            # Convert input pincode to integer for consistent comparison
            try:
                # Handle both string and float inputs by converting to float first, then int
                pincode_str = str(pincode).strip()
                if pincode_str.lower() in ['nan', 'none', '']:
                    return None
                
                # Convert via float to handle cases like '263153.0'
                input_pincode_float = float(pincode_str)
                input_pincode_int = int(input_pincode_float)
            except (ValueError, TypeError):
                return None
            
            # Convert reference Pin_code column to numeric for comparison
            ref_df_copy = ref_df.copy()
            ref_df_copy['Pin_code_int'] = pd.to_numeric(ref_df_copy['Pin_code'], errors='coerce')
            
            
            # Look for matches using integer comparison
            pincode_matches = ref_df_copy[ref_df_copy['Pin_code_int'] == input_pincode_int]
            
            if not pincode_matches.empty:
                result_id = pincode_matches.iloc[0]['Id']
                return result_id
                
        except Exception:
            # If pincode lookup fails, continue to fallback methods
            pass
    
    # Fallback method: Use city and state (existing logic)
    if not state: return None
    if city and 'State' in ref_df.columns and 'City' in ref_df.columns:
        try:
            matches = ref_df[(ref_df['State'] == state) & (ref_df['City'] == city)]
            if not matches.empty:
                # Check if 'rank' column exists (old format) or just take first match (new format)
                if 'rank' in ref_df.columns:
                    return matches.sort_values(by='rank', ascending=False).iloc[0]['Id']
                else:
                    return matches.iloc[0]['Id']
        except Exception:
            # If city/state lookup fails, continue to final fallback
            pass
    
    # Final fallback: Generate ID from state
    return state.replace(' ', '_') + '_in'

def pc_process_data_rows(df_input, mappings, defaults, ref_id_df, llm_state_map, llm_truck_type_map, app_config, data_was_transformed):
    final_rows = []
    pc_conf = app_config['pricing_cleaner']
    target_keys = pc_conf["llm_mapping_target_keys"]
    o_city_col, o_state_col = mappings.get(target_keys['origin_city']), mappings.get(target_keys['origin_state'])
    d_city_col, d_state_col = mappings.get(target_keys['destination_city']), mappings.get(target_keys['destination_state'])
    o_pincode_col, d_pincode_col = mappings.get(target_keys['origin_pincode']), mappings.get(target_keys['destination_pincode'])
    rate_val_col = mappings.get(target_keys['Rate Value'])

    for _, source_row in df_input.iterrows():
        output_row = {
            'Origin Type': 'cluster',
            'Destination Type': 'cluster',
            'Service Type': defaults.get('Service Type', 'FTL'), # Use default, else FTL
            'Rate Type': defaults.get('Rate Type', 'Flat'),       # Use default, else Flat
            'Transit Days': defaults.get('Transit Days', 5)       # Use default, else 5
        }
        
        service_type_mapped_col = mappings.get(target_keys.get('Service Type')) 
        
        if service_type_mapped_col and service_type_mapped_col in source_row and pd.notna(source_row[service_type_mapped_col]):
            input_service_type = str(source_row[service_type_mapped_col]).strip()
            output_row['Service Type'] = str(source_row[service_type_mapped_col]).strip()
            output_row['Service Type'] = input_service_type.upper() # Store in uppercase for consistency
            # NEW: Conditional Rate Type assignment for "Both FTL and LTL" flow
            # This logic assumes the 'pricing_type_selection' effectively leads here
            # and that 'Service Type' is being read from the input file.
            lower_input_service_type = input_service_type.lower()
            if lower_input_service_type == 'ftl':
                output_row['Rate Type'] = 'Flat'
            elif lower_input_service_type in ['ptl', 'ltl']:
                output_row['Rate Type'] = 'Per Km'
            # If the service type is not 'ftl', 'ptl', or 'ltl', Rate Type will remain its default or previous value.

        o_city_raw, o_state_raw = (str(source_row.get(o_city_col, '')) if o_city_col else ''), (str(source_row.get(o_state_col, '')) if o_state_col else '')
        d_city_raw, d_state_raw = (str(source_row.get(d_city_col, '')) if d_city_col else ''), (str(source_row.get(d_state_col, '')) if d_state_col else '')
        o_pincode_raw, d_pincode_raw = (str(source_row.get(o_pincode_col, '')) if o_pincode_col else ''), (str(source_row.get(d_pincode_col, '')) if d_pincode_col else '')
        
        o_city_val, o_state_val = o_city_raw.strip().lower() or defaults.get('origin_city', '').strip().lower(), o_state_raw.strip() or defaults.get('origin_state', '').strip()
        d_city_val, d_state_val = d_city_raw.strip().lower(), d_state_raw.strip()
        o_pincode_val, d_pincode_val = o_pincode_raw.strip() or defaults.get('origin_pincode', '').strip(), d_pincode_raw.strip() or defaults.get('destination_pincode', '').strip()
        
        o_state_std, d_state_std = llm_state_map.get(o_state_val, o_state_val.lower()), llm_state_map.get(d_state_val, d_state_val.lower() if d_state_val else None)
        

        output_row.update({'origin_city': o_city_val, 'origin_state': o_state_std, 'destination_city': d_city_val, 'destination_state': d_state_std})
        origin_pincode_to_use = o_pincode_val if o_pincode_val else None
        dest_pincode_to_use = d_pincode_val if d_pincode_val else None
        
        output_row['Origin Ref ID'] = pc_find_ref_id(o_city_val, o_state_std, ref_id_df, origin_pincode_to_use)
        output_row['Destination Ref ID'] = pc_find_ref_id(d_city_val, d_state_std, ref_id_df, dest_pincode_to_use)
        
        raw_rate = source_row.get('Rate Value') if data_was_transformed else source_row.get(rate_val_col)
        output_row['Rate Value'] = pd.to_numeric(re.sub(r'[^\d\.]', '', str(raw_rate)), errors='coerce')

        client_truck_original = ""
        if data_was_transformed:
            client_truck_original = str(source_row.get('Original Truck Header', "")).strip()
            if 'Original Truck Header' in source_row.index:
                output_row['Original Truck Header'] = source_row.get('Original Truck Header')
        else:
            truck_type_col = mappings.get(target_keys['Truck Type'])
            if truck_type_col:
                client_truck_original = str(source_row.get(truck_type_col, "")).strip()
        
        output_row['Original Client Truck Type'] = client_truck_original
        output_row['Truck Type'] = llm_truck_type_map.get(client_truck_original) if llm_truck_type_map else None 
        final_rows.append(output_row)
    return pd.DataFrame(final_rows)

def pc_finalize_dataframe(df, app_config):
    if 'Truck Type' in df.columns: df['Truck Type'].fillna('UNKNOWN_STANDARD_TRUCK', inplace=True)
    else: df['Truck Type'] = 'NOT_MAPPED_BY_USER'
    if 'Rate Value' in df.columns: df['Rate Value'] = df['Rate Value'].replace(0, 10).fillna(10)
    
    final_cols = app_config["pricing_cleaner"]["standard_output_columns"][:]
    if 'Original Client Truck Type' in df.columns: final_cols.insert(final_cols.index('Truck Type') + 1, 'Original Client Truck Type')
    if 'Original Truck Header' in df.columns: final_cols.insert(final_cols.index('Original Client Truck Type') + 1, 'Original Truck Header')
    df = df.reindex(columns=[col for col in final_cols if col in df.columns])

    df_ltl_prices = pc_load_default_ltl_prices(app_config["paths"]["ltl_prices"])
    if df_ltl_prices is not None:
        ltl_rate_map = {(row['origin_state'], row['destination_state']): row['Rate Value'] for _, row in df_ltl_prices.iterrows()}
        new_ltl_rows = [
            {'Origin Ref ID': r.get('Origin Ref ID'), 'Origin Type': r.get('Origin Type'), 'origin_city': r.get('origin_city'), 'origin_state': r.get('origin_state'),
             'Destination Ref ID': r.get('Destination Ref ID'), 'Destination Type': r.get('Destination Type'), 'destination_city': r.get('destination_city'), 'destination_state': r.get('destination_state'),
             'Truck Type': None, 'Service Type': 'LTL', 'Rate Type': 'Per Km', 'Rate Value': ltl_rate_map.get((r.get('origin_state'), r.get('destination_state'))), 'Transit Days': r.get('Transit Days')}
            for _, r in df.iterrows() if ltl_rate_map.get((r.get('origin_state'), r.get('destination_state'))) is not None
        ]
        if new_ltl_rows:
            df = pd.concat([df, pd.DataFrame(new_ltl_rows)], ignore_index=True)
    return df


# TMS PLANNER: FUNCTIONS
def tms_validate_uploaded_xlsx(uploaded_file):
    """
    Validates that the uploaded XLSX file contains the required sheets and columns.
    """
    required_sheets_cols = {
        'orders': [
            'Origin Facility ID', 'Origin address', 'Origin address PIN code', 'origin_city', 
            'Destination address', 'Destination address PIN code', 'destination_city', 'Order Number', 
            'Order Date', 'Movement', 'Order Item No.', 'Product Code', 'Product Name', 'Quantity', 
            'Unit of measurement', 'Weight', 'Volume', 'Weight Unit', 'Volume Unit', 
            'Product Category', 'Total_cost'
        ],
        'rate_master': [
            'Origin Ref ID', 'Origin Type', 'Destination Ref ID', 'Destination Type', 'Truck Type', 
            'Service Type', 'Rate Type', 'Rate Value', 'Transit Days'
        ],
        'vehicles_types': [
            'Model Name', 'Type', 'Weight Capacity', 'Volume Capacity', 'Length', 'Breadth', 'Height'
        ],
        'plan_setting': [
            'Min Weight Utilisation', 'Min Volume Utilisation', 'Max Detour', 'Customer Split', 
            'Order Split', 'Plan LTL Loads', 'Consolidate LTL', 'Max Stops'
        ]
    }
    try:
        xls = pd.ExcelFile(uploaded_file)
        sheet_names = xls.sheet_names
        missing_sheets = [sheet for sheet in required_sheets_cols.keys() if sheet not in sheet_names]
        if missing_sheets:
            st.error(f"Upload failed. Missing sheets: {', '.join(missing_sheets)}")
            return False
        all_checks_passed = True
        for sheet_name, required_cols in required_sheets_cols.items():
            df = pd.read_excel(xls, sheet_name=sheet_name)
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                st.error(f"Validation failed for sheet '{sheet_name}'. Missing columns: {', '.join(missing_cols)}")
                all_checks_passed = False
        return all_checks_passed
    except Exception as e:
        st.error(f"An error occurred while reading the uploaded Excel file: {e}")
        return False

def tms_extract_date_from_sub_id(sub_id_str):
    if not isinstance(sub_id_str, str): return None
    match = re.search(r'::(\d{4}-\d{1,2}-\d{1,2})::', sub_id_str)
    return pd.to_datetime(match.group(1)).date() if match else None

def tms_process_orders_data(df):
    if df.empty: return pd.DataFrame()
    processed_df = df.copy()
    if 'pincode_is_valid' in processed_df.columns:
        initial_rows = len(processed_df)
        processed_df = processed_df[processed_df['pincode_is_valid'] != 0]
        rows_removed = initial_rows - len(processed_df)
        if rows_removed > 0: st.info(f"ℹ️ Removed {rows_removed} order(s) where 'pincode_is_valid' was 0.")
    cols_to_drop = ["Origin_Validation_Status", "pincode_is_valid", "Transaction Type"]
    processed_df = processed_df.drop(columns=[col for col in cols_to_drop if col in processed_df.columns], errors='ignore')
    return processed_df

def tms_process_rate_master_data(df):
    if df.empty: return pd.DataFrame()
    processed_df = df.copy()
    cols_to_drop = ['Original Client Truck Type', 'Original Truck Header', 'destination_city', 'destination_state', 'origin_city', 'origin_state']
    processed_df = processed_df.drop(columns=[col for col in cols_to_drop if col in processed_df.columns], errors='ignore')
    return processed_df

def tms_generate_tms_input_excel(orders_df, rates_df, vehicles_df, plan_settings_df, product_df=None):
    excel_buffer = BytesIO()
    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
        orders_df.to_excel(writer, sheet_name='orders', index=False)
        rates_df.to_excel(writer, sheet_name='rate_master', index=False)
        vehicles_df.to_excel(writer, sheet_name='vehicles_types', index=False)
        plan_settings_df.to_excel(writer, sheet_name='plan_setting', index=False)
        # Add product_info sheet if product data is available
        if product_df is not None and not product_df.empty:
            product_df.to_excel(writer, sheet_name='product_info', index=False)
    excel_buffer.seek(0)
    return excel_buffer

def tms_upload_to_s3_and_get_signed_url(file_buffer, bucket_name, region):
    if not all([bucket_name, region]):
        st.error("S3 bucket name or region are not configured. Please set S3_BUCKET and S3_REGION in your .env file.")
        return None
    s3_file_name = f"tms-inputs/input-{int(time.time())}.xlsx"
    s3_config = Config(signature_version='s3v4', region_name=region)
    s3_client = boto3.client('s3', config=s3_config)
    try:
        file_buffer.seek(0)
        s3_client.upload_fileobj(file_buffer, bucket_name, s3_file_name)
        st.info(f"Successfully uploaded file to s3://{bucket_name}/{s3_file_name}")
        return s3_client.generate_presigned_url('get_object', Params={'Bucket': bucket_name, 'Key': s3_file_name}, ExpiresIn=3600)
    except NoCredentialsError:
        st.error("AWS credentials not found. Ensure the application is running on an EC2 instance with a correctly configured IAM role.")
        return None
    except Exception as e:
        st.error(f"An error occurred during S3 upload: {e}")
        return None

def tms_trigger_tms_plan(api_url, signed_url):
    headers = {'Content-Type': 'application/json'}
    payload = {"action": "trigger_plan", "file_url": signed_url}
    response = requests.post(api_url, headers=headers, data=json.dumps(payload))
    response.raise_for_status()
    return response.json()

def tms_get_tms_status(api_url, req_id):
    headers = {'Content-Type': 'application/json'}
    payload = {"action": "status", "req_id": req_id}
    response = requests.post(api_url, headers=headers, data=json.dumps(payload))
    response.raise_for_status()
    return response.json()


def tms_perform_data_validations(orders_df, rates_df, config):
    """
    Performs specific validation checks on the orders and rate_master DataFrames.
    Returns True if all validations pass, False otherwise, along with error messages.
    """
    errors = []

    # --- A) Validate rate_master DataFrame ---
    required_rate_cols = [
        'Origin Ref ID', 'Origin Type', 'Destination Ref ID', 'Destination Type',
        'Truck Type', 'Service Type', 'Rate Type', 'Rate Value', 'Transit Days'
    ]
    
    missing_rate_cols = [col for col in required_rate_cols if col not in rates_df.columns]
    if missing_rate_cols:
        errors.append(f"Rate Master: Missing required columns: {', '.join(missing_rate_cols)}.")

    if 'Service Type' in rates_df.columns:
        invalid_service_types = rates_df[~rates_df['Service Type'].isin(['FTL', 'LTL'])]['Service Type'].unique()
        if len(invalid_service_types) > 0:
            errors.append(f"Rate Master: 'Service Type' column contains invalid values (must be 'FTL' or 'LTL'): {', '.join(map(str, invalid_service_types))}.")
    
    # Check for NULL rows in critical rate_master columns
    critical_rate_null_cols = ['Origin Ref ID', 'Destination Ref ID', 'Service Type', 'Rate Type', 'Rate Value', 'Transit Days']
    for col in critical_rate_null_cols:
        if col in rates_df.columns and rates_df[col].isnull().any():
            errors.append(f"Rate Master: Column '{col}' contains NULL values.")


    # --- B) Validate orders DataFrame ---
    required_order_cols = [
        'Origin Facility ID', 'Origin address', 'Origin address PIN code', 'origin_city',
        'Destination address', 'Destination address PIN code', 'destination_city', 'Order Number',
        'Order Date', 'Movement', 'Order Item No.', 'Product Code', 'Product Name', 'Quantity',
        'Unit of measurement', 'Weight', 'Volume', 'Weight Unit', 'Volume Unit', 'Product Category'
    ]

    missing_order_cols = [col for col in required_order_cols if col not in orders_df.columns]
    if missing_order_cols:
        errors.append(f"Orders: Missing required columns: {', '.join(missing_order_cols)}.")

    # Check 'Order Date' format and convert to string
    if 'Order Date' in orders_df.columns and not orders_df['Order Date'].empty:
        # Check if the column can be converted to datetime with a strict format
        try:
            # Attempt to convert to datetime using a strict YYYY-MM-DD format.
            # `errors='raise'` will stop the execution immediately if a single value fails,
            # which is what we want for a strict check.
            temp_dates = pd.to_datetime(orders_df['Order Date'], format='%Y-%m-%d', errors='raise')

            # If the check passes, convert the column to string type to ensure consistency.
            orders_df['Order Date'] = temp_dates.astype(str)
            st.success("✅ 'Order Date' column is in YYYY-MM-DD format and has been converted to string.")

        except Exception:
            # If the to_datetime call raises an error, it means the format is incorrect.
            errors.append("Orders: 'Order Date' column is not in YYYY-MM-DD format. Please correct and re-upload.")

    if 'Movement' in orders_df.columns:
        invalid_movements = orders_df[~orders_df['Movement'].isin(['Primary', 'Secondary'])]['Movement'].unique()
        if len(invalid_movements) > 0:
            errors.append(f"Orders: 'Movement' column contains invalid values (must be 'Primary' or 'Secondary'): {', '.join(map(str, invalid_movements))}.")

    if 'Unit of measurement' in orders_df.columns:
        valid_uom = config["order_cleaner"]["default_uom_options"][1:] # Exclude "-- Select Default (Optional) --"
        invalid_uoms = orders_df[~orders_df['Unit of measurement'].isin(valid_uom)]['Unit of measurement'].unique()
        if len(invalid_uoms) > 0:
            errors.append(f"Orders: 'Unit of measurement' column contains invalid values (must be 'BOX', 'BUNDLE', or 'PCS'): {', '.join(map(str, invalid_uoms))}.")

    if 'Product Category' in orders_df.columns:
        valid_product_category = config["order_cleaner"]["default_product_category_options"][1:] # Exclude "-- Select Default (Optional) --"
        invalid_categories = orders_df[~orders_df['Product Category'].isin(valid_product_category)]['Product Category'].unique()
        if len(invalid_categories) > 0:
            errors.append(f"Orders: 'Product Category' column contains invalid values (must be 'DRY' or 'WET'): {', '.join(map(str, invalid_categories))}.")

    if errors:
        for error in errors:
            st.warning(f"Validation Warning: {error}")
        return False
    return True

def tms_fetch_all_tms_results(api_url, req_id, sub_ids_df):
    if sub_ids_df.empty: return [], [], [], []
    st.info("Fetching detailed plan results...")
    progress_bar = st.progress(0, text="Fetching results...")
    plan_summaries, route_summaries, route_sequences, failed_sub_ids = [], [], [], []
    headers = {'Content-Type': 'application/json'}
    for i, row in sub_ids_df.iterrows():
        sub_plan_id, date = row['sub_id'], row['date']
        payload = {"action": "result", "req_id": req_id, "sub_plan_id": sub_plan_id}
        try:
            response = requests.post(api_url, headers=headers, data=json.dumps(payload))
            response.raise_for_status()
            result_payload = response.json().get('results', {}).get('payload')
            if result_payload is None:
                st.warning(f"Payload is NULL for sub_plan_id: {sub_plan_id}")
                failed_sub_ids.append(sub_plan_id)
                continue
            
            # Handle new API response structure with nested 'data' key
            # Check if the response has the new structure with 'success' and 'data' keys
            if 'success' in result_payload and 'data' in result_payload:
                # New API structure: extract data from nested 'data' key
                actual_data = result_payload.get('data', {})
            else:
                # Old API structure: data is directly in result_payload
                actual_data = result_payload
            
            if ps := actual_data.get('plan_summary'):
                ps.update({'sub_id': sub_plan_id, 'date': date}); plan_summaries.append(ps)
            for route in actual_data.get('routes_info', []):
                if rs := route.get('route_summary'):
                    rs.update({'date': date, 'sub_plan_id': sub_plan_id}); route_summaries.append(rs)
                for seq in route.get('route_sequence', []):
                    seq.update({'sub_plan_id': sub_plan_id, 'date': date}); route_sequences.append(seq)
        except (requests.RequestException, KeyError, json.JSONDecodeError) as e:
            st.error(f"Failed for sub_plan_id {sub_plan_id}: {e}"); failed_sub_ids.append(sub_plan_id)
        finally:
            progress_bar.progress((i + 1) / len(sub_ids_df), text=f"Fetching result {i+1}/{len(sub_ids_df)}")
    progress_bar.empty(); st.success("Finished fetching results.")
    return plan_summaries, route_summaries, route_sequences, failed_sub_ids

def tms_generate_client_dod_summary(plan_summaries, route_sequences, initial_orders_df):
    """Generates the Client Day-over-Day (DOD) summary DataFrame."""
    if not plan_summaries or not route_sequences or initial_orders_df.empty:
        return pd.DataFrame()
        
    plan_summary_df = pd.DataFrame(plan_summaries)
    route_sequence_df = pd.DataFrame(route_sequences)
    orders_input = initial_orders_df.copy()

    orders_input['orderline_code'] = orders_input['Order Number'].astype(str) + '::' + orders_input['Order Item No.'].astype(str)
    
    cba_cols = [
        'date', 'total_cost', 'total_shipments', 'total_weight', 'total_volume',
        'total_vehicles_used', 'total_ftl_vehicles_weight_utilisation',
        'total_ftl_vehicles_volume_utilisation', 'total_orders_available',
        'total_orders_served', 'total_orders_left', 'total_orderlines_available',
        'total_orderlines_served', 'total_orderlines_left'
    ]
    cba_summary_df = plan_summary_df[[col for col in cba_cols if col in plan_summary_df.columns]]

    merged_df = orders_input.merge(route_sequence_df, how='left', on='orderline_code')
    served_df = merged_df[merged_df['vehicle_index'].notnull()].copy()

    for col in ['Total_cost', 'Weight', 'Volume', 'Quantity']:
        if col in served_df.columns:
            served_df[col] = pd.to_numeric(served_df[col], errors='coerce')
    
    client_summary_df = served_df.groupby('Order Date', as_index=False).agg({
        'Total_cost': 'sum', 'Weight': 'sum', 'Volume': 'sum', 'Quantity': 'sum'
    }).rename(columns={
        'Order Date': 'date', 'Total_cost': 'client_total_cost',
        'Weight': 'client_total_weight', 'Volume': 'client_total_volume',
        'Quantity': 'client_total_qty'
    })

    client_summary_df['date'] = pd.to_datetime(client_summary_df['date']).dt.date
    cba_summary_df['date'] = pd.to_datetime(cba_summary_df['date']).dt.date

    final_summary_df = cba_summary_df.merge(client_summary_df, how='left', on='date')
    return final_summary_df


def tms_generate_plan_output_excel(plan_summaries, route_summaries, route_sequences, initial_orders_df):
    if initial_orders_df is None or initial_orders_df.empty:
        st.error("Cannot generate plan output: Initial order data is missing.")
        return None
    plan_summary_df = pd.DataFrame(plan_summaries)
    route_summary_df = pd.DataFrame(route_summaries)
    route_sequence_df = pd.DataFrame(route_sequences)
    orders_input = initial_orders_df.copy()
    
    orders_input['orderline_code'] = orders_input['Order Number'].astype(str) + '::' + orders_input['Order Item No.'].astype(str)
    ##check and remove post validation
    orders_input.to_csv('orders_input_check.csv', index=False)
    route_sequence_df.to_csv('route_sequence_check.csv', index=False)
    plan_summary_df.to_csv('plan_summary_check.csv', index=False)
    route_summary_df.to_csv('route_summary_check.csv', index=False)
    #### check till here   /home/ubuntu/ref_id_file.csv
    merged_orders = orders_input.merge(route_sequence_df, how='left', on='orderline_code')
    unallocated_data = []
    if 'unallocated_orderlines_info' in plan_summary_df.columns:
        for row_list in plan_summary_df['unallocated_orderlines_info'].dropna():
            if isinstance(row_list, list):
                for entry in row_list:
                     if isinstance(entry, dict):
                        unallocated_data.append({'orderline_code': entry.get('orderline_code'), 'reason': entry.get('reason')})
    if unallocated_data:
        unallocated_df = pd.DataFrame(unallocated_data).drop_duplicates()
        merged_orders = merged_orders.merge(unallocated_df, how='left', on='orderline_code')
    
    merged_orders['Order Status'] = 'pending'
    merged_orders.loc[merged_orders['sub_plan_id'].isnull(), 'Order Status'] = 'unassigned'
    output_orders_cols = ['Order Date', 'vehicle_index', 'Order Number', 'Origin Facility ID', 'Destination Facility ID', 'Order Item No.', 'Product Code', 'Product Name', 'Quantity', 'Unit of measurement', 'Weight', 'Weight Unit', 'Volume', 'Volume Unit', 'sub_plan_id', 'orderline_code', 'Order Status', 'reason']
    plan_output_orders = merged_orders[[col for col in output_orders_cols if col in merged_orders.columns]]
    plan_output_loads = route_summary_df.copy()
    if 'route_location_codes_sequence' in plan_output_loads.columns:
        seq_col = plan_output_loads['route_location_codes_sequence']
        plan_output_loads['Origin Facility ID'] = seq_col.apply(lambda x: x[0] if isinstance(x, list) and x else None)
        plan_output_loads['Destination Facility ID'] = seq_col.apply(lambda x: x[-1] if isinstance(x, list) and x else None)
    summary_cols = ['date', 'sub_id', 'total_shipments', 'total_orders_served', 'total_cost', 'total_weight', 'total_vehicles_used']
    plan_output_summary = plan_summary_df[[col for col in summary_cols if col in plan_summary_df.columns]].copy()
    plan_output_summary['cost_per_kg'] = plan_output_summary['total_cost'] / plan_output_summary['total_weight'].replace(0, pd.NA)
    output_buffer = BytesIO()
    with pd.ExcelWriter(output_buffer, engine='openpyxl') as writer:
        plan_output_summary.to_excel(writer, sheet_name='summary', index=False)
        plan_output_loads.to_excel(writer, sheet_name='loads', index=False)
        plan_output_orders.to_excel(writer, sheet_name='orders', index=False)
    output_buffer.seek(0)
    return output_buffer

# ==============================================================================
# 6. UI RENDERING FUNCTIONS
# ==============================================================================

def render_scs_standardiser_page(client, deployment_name, config):
    """Renders the UI and orchestrates the logic for the SCS Standardiser."""
    st.markdown("<h1><span style='color:white;'>SCS - </span><span style='color:red;'>Standardiser</span></h1>", unsafe_allow_html=True)
    st.markdown("Upload a file to clean, map, and enrich your logistics data. The output can then be passed to the TMS Workflow.")

    uploaded_file = st.file_uploader(
        "Upload Excel/CSV File",
        type=["xlsx", "xls", "csv"],
        key=f"scs_uploader_{st.session_state.uploader_key}"
    )

    if not uploaded_file:
        st.info("Please upload a file to begin.")
        # Reset relevant session state when no file is uploaded
        st.session_state.df_renamed = None
        st.session_state.scs_processed_df = None
        st.session_state.eda_summary_json = None
        st.session_state.scs_llm_response = None
        return

    try:
        file_name = uploaded_file.name.lower()
        if file_name.endswith(".csv"):
            df_input = pd.read_csv(uploaded_file)
        else: # .xlsx, .xls
            xls = pd.ExcelFile(uploaded_file)
            sheet_name = st.selectbox("Select Sheet", xls.sheet_names)
            df_input = pd.read_excel(xls, sheet_name=sheet_name)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return

    st.dataframe(df_input.head())

    skus_columns = ["product_dimension", "length", "width", "height", "product_weight", "product_code", "product_desc", "pack_size", "product_category"]
    order_columns = ['invoice_no','order_id','quantity','date']
    pincode_columns = ['pincode', 'city', 'state', 'pincode_origin', 'city_origin', 'state_origin']
    pin_1 = ['pincode', 'city', 'state']
    pin_2 = ['pincode_origin', 'city_origin', 'state_origin']

    with st.form("scs_config_form"):
        st.subheader("Configuration")
        st.session_state.scs_client_name = st.text_input("Client Name", value=st.session_state.scs_client_name)
        st.session_state.scs_product_types = st.text_input("General Product Types", value=st.session_state.scs_product_types)
        
        cols = st.columns(2)
        st.session_state.scs_dimension_unit = cols[0].selectbox("Dimension Unit", ["inch", "cm", "mm", "m", "ft"], index=0)
        st.session_state.scs_weight_unit = cols[1].selectbox("Weight Unit", ["kg", "gm"], index=0)
        
        submitted = st.form_submit_button("Confirm Configuration and Auto-Map Columns")

    if submitted:
        with st.spinner("🤖 Asking AI to suggest column mappings..."):
            st.session_state.scs_llm_response = get_llm_suggestions(client, deployment_name, list(df_input.columns))
        # Reset downstream states when re-mapping
        st.session_state.df_renamed = None
        st.session_state.scs_processed_df = None
        st.session_state.eda_summary_json = None


    if st.session_state.scs_llm_response:
        st.subheader("Column Mapping Verification")
        st.info("Please verify the AI-suggested mappings. Correct any if needed.")
        
        column_map = {}
        expected_columns = [
            "pincode", "city", "state", "pincode_origin", "city_origin", "state_origin",
            "invoice_no", "order_id", "quantity", "date", "product_category", "product_code",
            "product_desc", "product_dimension", "length", "width", "height", "product_weight", "pack_size"
        ]
        
        with st.form("scs_mapping_form"):
            num_cols_per_row = 3
            for i in range(0, len(expected_columns), num_cols_per_row):
                row_cols = expected_columns[i:i + num_cols_per_row]
                cols = st.columns(len(row_cols))
                for j, col in enumerate(row_cols):
                    suggested_col = st.session_state.scs_llm_response.get(col)
                    options = [None] + list(df_input.columns)
                    index = options.index(suggested_col) if suggested_col in options else 0
                    
                    # Add critical indicator for 'product_desc'
                    label = f"❗️ Map '{col}'" if col == 'product_desc' else f"Map '{col}'"
                    
                    column_map[col] = cols[j].selectbox(label, options, index=index, key=f"map_{col}")
            
            
            mapping_submitted = st.form_submit_button("✅ Confirm Mappings to Run Agents")

        if mapping_submitted:
            st.session_state.scs_column_map = column_map
            clean_mapping = {v: k for k, v in column_map.items() if v is not None}
            df_renamed = df_input.rename(columns=clean_mapping)

            if 'date' in df_renamed.columns:
                try:
                    df_renamed['date'] = pd.to_datetime(df_renamed['date'], errors='coerce').dt.strftime('%Y-%m-%d')
                    st.success("✅ 'date' standardized to YYYY-MM-DD format.")
                except Exception as e:
                    st.warning(f"⚠️ Could not standardize 'date' column: {e}")
            
            st.session_state.df_renamed = df_renamed
            # Reset results from previous runs
            st.session_state.scs_processed_df = None
            st.session_state.eda_summary_json = None

            st.write('Mapping Preview', df_renamed.head())


    # --- Agent Buttons and Logic (displayed after mapping is confirmed) ---
    if st.session_state.df_renamed is not None:
        st.markdown("---")
        st.subheader("🤖 Choose an Agent to Run")

        # Define columns for button layout first
        col1, col2, col3 = st.columns(3)
        run_super_agent = col1.button("💥 Run SCS Super-Agent", use_container_width=True)
        run_eda_agent = col2.button("📊 Run EDA Agent", use_container_width=True)
        run_standardiser_agent = col3.button("🧹 Run File Standardiser Agent", use_container_width=True)


        # --- Agent Execution Logic (Full Page Width) ---
        if run_super_agent:
            df_renamed = st.session_state.df_renamed
            #if user submits only product code terminate the code then and there...
            if len(df_renamed.columns) == 1 and 'product_code' in df_renamed.columns:
                st.error("❌ **Processing Stopped:** A file with only 'product_code' is not sufficient. Please provide 'product_desc' for predictions or full weight and dimension data for validation.")
                return

            with st.spinner("🚀 Running SCS Super-Agent... This may take a while."):
                # Display Plan of Action
                st.subheader("🤖 Plan of Action")
                df_to_check = df_renamed
                has_orders = any(col in df_to_check.columns for col in order_columns)
                has_skus = any(col in df_to_check.columns for col in skus_columns)
                has_dest_pin = any(col in df_to_check.columns for col in pin_1)
                has_origin_pin = any(col in df_to_check.columns for col in pin_2)
                sku_cols_1, sku_cols_2 = ["product_dimension", "product_weight"], ["length", "width", "height", "product_weight"]
                sku_base = ["product_category","product_code", "product_desc", "pack_size"]
                # (Plan of action logic remains the same)

                # A) Only 'product_desc' uploaded
                if 'product_desc' in df_to_check.columns and len(df_to_check.columns) == 1:
                    with st.status("Plan of Action", expanded=True):
                        st.markdown("""
                        - ✅ Found only 'product_desc' column.
                        - 🧠 Agent will call LLM to predict missing dimensions and weight based on product description.
                        """)
                    df_weight_dimensions_result = missing_weight_dimensions_estimator_agent(client, deployment_name, df_renamed.copy())
                    st.session_state['missing_value_button'] = df_weight_dimensions_result

                # B) 'product_code' along with 'weight' and 'dimension'
                elif 'product_code' in df_to_check.columns and (all(col in df_to_check.columns for col in ['length', 'width', 'height', 'product_weight']) or all(col in df_to_check.columns for col in ['product_dimension', 'product_weight'])):
                    has_cols_1 = all(c in df_to_check.columns for c in sku_cols_1)
                    has_cols_2 = all(c in df_to_check.columns for c in sku_cols_2)
                    is_missing_or_zero = False
                    if has_cols_1:
                        is_missing_or_zero = df_to_check[sku_cols_1].isnull().any().any() or (df_to_check[sku_cols_1] == 0).any().any()
                    elif has_cols_2:
                        is_missing_or_zero = df_to_check[sku_cols_2].isnull().any().any() or (df_to_check[sku_cols_2] == 0).any().any()

                    if (has_cols_1 or has_cols_2) and not is_missing_or_zero:
                        with st.status("Plan of Action", expanded=True):
                            st.markdown("""
                            - ✅ Found 'product_code' along with complete Weight and Dimension data.
                            - 🤖 EDA and Anomaly Agent will be Activated.
                            - 📊 EDA Agent will show Data Summary and Frequency distributions of the columns.
                            - 🧹 Dimension/Weight Analyzer Agent will clean the data, fetch Weight & Density threshold from LLM and creates golden/true dataset.
                            - 🔍 Agent finds the anomalous weight/dimension products based on Weight and Density thresholds.
                            """)
                        df_weight_dimensions_result = weight_anomaly_detection(client, deployment_name, df_renamed.copy())
                        st.session_state['anomaly_button'] = df_weight_dimensions_result
                    else: # This covers cases where 'product_code' + (weight/dimension) are present but incomplete/have zeros
                        with st.status("Plan of Action", expanded=True):
                            st.markdown("""
                            - ✅ Found 'product_code' along with incomplete/zero Weight and Dimension data.
                            - 🤖 EDA and Dimension/Weight Analyzer Agent will be Activated.
                            - 📊 EDA Agent will show Data Summary and Frequency distributions of the columns.
                            - 🧹 Dimension/Weight Analyzer Agent will clean the data, fetch Weight & Density threshold from LLM and creates golden/true dataset.
                            - 🔍 Agent finds best match for missing/anomalous values within true dataset on product code and description.
                            - 🧠 Agent calls LLM to predict for unmapped products.
                            """)
                        df_weight_dimensions_result = missing_weight_dimensions_estimator_agent(client, deployment_name, df_renamed.copy())
                        st.session_state['missing_value_button'] = df_weight_dimensions_result

                # ... (rest of your existing if/elif conditions for other scenarios)
                elif has_orders:
                    nulls_detail = missing_col_null_check(df_to_check, order_columns)
                    with st.status("Plan of Action", expanded=True): st.markdown(f"""- ✅ Found columns for Order data\n- **Details:** `{nulls_detail}`\n- 🤖 EDA Agent will be activated\n- 📊  EDA Agent will show order Summary, order pareto analysis and date level trend\n- 🤖 Super cleaning Agent will be activated\n- 🗺️ Agent will map clean the pincode, find anomalies and predict missing weight and dimension based on data""")
                elif has_dest_pin and has_origin_pin and not has_skus:
                    nulls_detail = missing_col_null_check(df_to_check, pincode_columns)
                    with st.status("Plan of Action", expanded=True): st.markdown(f"""- ✅ Found columns for Origin and destination Pincode/City/State\n- **Details:** `{nulls_detail}`\n- 🤖 Pincode Distance Agent will be Activated\n- 🗺️ Agent will map the origin and destination data to delhivery internal data: pincode, city, state and TAT city\n- 📍 Pincode Agent will fetch Latitude and Longitude using LLM\n- 📏 Agent will calculate distance between origin and destination pincode using OSRM API""")
                elif not has_dest_pin and not has_origin_pin and has_skus:
                    has_cols_1 = all(col in df_to_check.columns for col in sku_cols_1)
                    has_cols_2 = all(col in df_to_check.columns for col in sku_cols_2)
                    if has_cols_1 and not (df_to_check[sku_cols_1].isnull().any() | (df_to_check[sku_cols_1] == 0).any()).any():
                        req_cols = sku_base + sku_cols_1
                        nulls_detail = missing_col_null_check(df_to_check, req_cols)
                        with st.status("Plan of Action", expanded=True): st.markdown(f"""- ✅ Found columns for Product 📦 Weights and Dimension\n- **Details:** `{nulls_detail}`\n- 🤖 EDA and Anomaly Agent will be Activated\n- 📊  EDA Agent will show Data Summary and Frequency distributions of the columns\n- 🧹  Dimension/Weight Analyzer Agent will clean the data, fetch Weight & Density threshold from LLM and creates golden/true dataset\n- 🔍 Agent finds the anomalous weight/dimension products based on Weight and Density thresholds""")
                    elif has_cols_2 and not (df_to_check[sku_cols_2].isnull().any() | (df_to_check[sku_cols_2] == 0).any()).any():
                        req_cols = sku_base + sku_cols_2
                        nulls_detail = missing_col_null_check(df_to_check, req_cols)
                        with st.status("Plan of Action", expanded=True): st.markdown(f"""- ✅ Found columns for Product 📦 Weights and Dimension\n- **Details:** `{nulls_detail}`\n- 🤖 EDA and Anomaly Agent will be Activated\n- 📊  EDA Agent will show Data Summary and Frequency distributions of the columns\n- 🧹  Dimension/Weight Analyzer Agent will clean the data, fetch Weight & Density threshold from LLM and creates golden/true dataset\n- 🔍 Agent finds the anomalous weight/dimension products based on Weight and Density thresholds""")
                    else:
                        if has_cols_1: 
                            req_cols = sku_base + sku_cols_1
                        elif has_cols_2: 
                            req_cols = sku_base + sku_cols_2
                        else:
                            req_cols = sku_base  # fallback to base columns if neither cols_1 nor cols_2 are available
                        nulls_detail = missing_col_null_check(df_to_check, req_cols)
                        with st.status("Plan of Action", expanded=True): st.markdown(f"""- ✅ Found columns for Product 📦 Weights and Dimension\n- **Details:** `{nulls_detail}`\n- 🤖 EDA and Dimension/Weight Analyzer Agent will be Activated\n- 📊  EDA Agent will show Data Summary and Frequency distributions of the columns\n- 🧹  Dimension/Weight Analyzer Agent will clean the data, fetch Weight & Density threshold from LLM and creates golden/true dataset\n- 🔍 Agent finds best match for missing/anomalous values within true dataset on product code and description\n- 🧠 Agent calls LLM to predict for unmapped products""")
                elif has_dest_pin and not has_origin_pin and has_skus:
                    has_cols_1 = all(col in df_to_check.columns for col in sku_cols_1)
                    has_cols_2 = all(col in df_to_check.columns for col in sku_cols_2)
                    if has_cols_1 and not (df_to_check[sku_cols_1].isnull().any() | (df_to_check[sku_cols_1] == 0).any()).any():
                        req_cols = list(pin_1) + sku_base + sku_cols_1
                        nulls_detail = missing_col_null_check(df_to_check, req_cols)
                        with st.status("Plan of Action", expanded=True): st.markdown(f"""- ✅ Found columns for Pincode/City/State\n- **Details:** `{nulls_detail}`\n- 🤖 Pincode Agent will be Activated\n- 🗺️ Agent will map the data to Delhivery internal data: pincode, city, state, and TAT city\n- 📍 Agent will fetch Latitude and Longitude using LLM\n- ✅ Found columns for Product 📦 Weights and Dimension\n- 🤖 EDA and Anomaly Agent will be Activated\n- 📊  EDA Agent will show Data Summary and Frequency distributions of the columns\n- 🧹  Dimension/Weight Analyzer Agent will clean the data, fetch Weight & Density threshold from LLM and creates golden/true dataset\n- 🔍 Agent finds the anomalous weight/dimension products based on Weight and Density thresholds""")
                    elif has_cols_2 and not (df_to_check[sku_cols_2].isnull().any() | (df_to_check[sku_cols_2] == 0).any()).any():
                        req_cols = list(pin_1) + sku_base + sku_cols_2
                        nulls_detail = missing_col_null_check(df_to_check, req_cols)
                        with st.status("Plan of Action", expanded=True): st.markdown(f"""- ✅ Found columns for Pincode/City/State\n- **Details:** `{nulls_detail}`\n- 🤖 Pincode Agent will be Activated\n- 🗺️ Agent will map the data to Delhivery internal data: pincode, city, state, and TAT city\n- 📍 Agent will fetch Latitude and Longitude using LLM\n- ✅ Found columns for Product 📦 Weights and Dimension\n- 🤖 EDA and Anomaly Agent will be Activated\n- 📊  EDA Agent will show Data Summary and Frequency distributions of the columns\n- 🧹  Dimension/Weight Analyzer Agent will clean the data, fetch Weight & Density threshold from LLM and creates golden/true dataset\n- 🔍 Agent finds the anomalous weight/dimension products based on Weight and Density thresholds""")
                    else:
                        if has_cols_1: 
                            req_cols = list(pin_1) + sku_base + sku_cols_1
                        elif has_cols_2: 
                            req_cols = list(pin_1) + sku_base + sku_cols_2
                        else:
                            req_cols = list(pin_1) + sku_base  # fallback to pin and base columns if neither cols_1 nor cols_2 are available
                        nulls_detail = missing_col_null_check(df_to_check, req_cols)
                        with st.status("Plan of Action", expanded=True): st.markdown(f"""- ✅ Found columns for Pincode/City/State\n- **Details:** `{nulls_detail}`\n- 🤖 Pincode Agent will be Activated\n- 🗺️ Agent will map the data to Delhivery internal data: pincode, city, state, and TAT city\n- 📍 Agent will fetch Latitude and Longitude using LLM\n- ✅ Found columns for Product 📦 Weights and Dimension\n- 🤖 EDA and Dimension/Weight Analyzer Agent will be Activated\n- 📊  EDA Agent will show Data Summary and Frequency distributions of the columns\n- 🧹  Dimension/Weight Analyzer Agent will clean the data, fetch Weight & Density threshold from LLM and creates golden/true dataset\n- 🔍 Agent finds best match for missing/anomalous values within true dataset on product code and description\n- 🧠 Agent calls LLM to predict for unmapped products""")
                elif has_dest_pin and has_origin_pin and has_skus:
                    has_cols_1 = all(col in df_to_check.columns for col in sku_cols_1)
                    has_cols_2 = all(col in df_to_check.columns for col in sku_cols_2)
                    if has_cols_1 and not (df_to_check[sku_cols_1].isnull().any() | (df_to_check[sku_cols_1] == 0).any()).any():
                        req_cols = pincode_columns + sku_base + sku_cols_1
                        nulls_detail = missing_col_null_check(df_to_check, req_cols)
                        with st.status("Plan of Action", expanded=True): st.markdown(f"""- ✅ Found columns for Origin and destination Pincode/City/State\n- **Details:** `{nulls_detail}`\n- 🤖 Pincode Distance Agent will be Activated\n- 🗺️ Agent will map the origin and destination data to delhivery internal data: pincode, city, state and TAT city\n- 📍 Pincode Agent will fetch Latitude and Longitude using LLM\n- 📏 Agent will calculate distance between origin and destination pincode using OSRM API\n- ✅ Found columns for Product 📦 Weights and Dimension\n- 🤖 EDA and Anomaly Agent will be Activated\n- 📊  EDA Agent will show Data Summary and Frequency distributions of the columns\n- 🧹  Dimension/Weight Analyzer Agent will clean the data, fetch Weight & Density threshold from LLM and creates golden/true dataset\n- 🔍 Agent finds the anomalous weight/dimension products based on Weight and Density thresholds""")
                    elif has_cols_2 and not (df_to_check[sku_cols_2].isnull().any() | (df_to_check[sku_cols_2] == 0).any()).any():
                        req_cols = pincode_columns + sku_base + sku_cols_2
                        nulls_detail = missing_col_null_check(df_to_check, req_cols)
                        with st.status("Plan of Action", expanded=True): st.markdown(f"""- ✅ Found columns for Origin and destination Pincode/City/State\n- **Details:** `{nulls_detail}`\n- 🤖 Pincode Distance Agent will be Activated\n- 🗺️ Agent will map the origin and destination data to delhivery internal data: pincode, city, state and TAT city\n- 📍 Pincode Agent will fetch Latitude and Longitude using LLM\n- 📏 Agent will calculate distance between origin and destination pincode using OSRM API\n- ✅ Found columns for Product 📦 Weights and Dimension\n- 🤖 EDA and Anomaly Agent will be Activated\n- 📊  EDA Agent will show Data Summary and Frequency distributions of the columns\n- 🧹  Dimension/Weight Analyzer Agent will clean the data, fetch Weight & Density threshold from LLM and creates golden/true dataset\n- 🔍 Agent finds the anomalous weight/dimension products based on Weight and Density thresholds""")
                    else:
                        if has_cols_1: 
                            req_cols = pincode_columns + sku_base + sku_cols_1
                        elif has_cols_2: 
                            req_cols = pincode_columns + sku_base + sku_cols_2
                        else:
                            req_cols = pincode_columns + sku_base  # fallback to pincode and base columns if neither cols_1 nor cols_2 are available
                        nulls_detail = missing_col_null_check(df_to_check, req_cols)
                        with st.status("Plan of Action", expanded=True): st.markdown(f"""- ✅ Found columns for Origin and destination Pincode/City/State\n- **Details:** `{nulls_detail}`\n- 🤖 Pincode Distance Agent will be Activated\n- 🗺️ Agent will map the origin and destination data to delhivery internal data: pincode, city, state and TAT city\n- 📍 Pincode Agent will fetch Latitude and Longitude using LLM\n- 📏 Agent will calculate distance between origin and destination pincode using OSRM API\n- ✅ Found columns for Product 📦 Weights and Dimension\n- 🤖 EDA and Dimension/Weight Analyzer Agent will be Activated\n- 📊  EDA Agent will show Data Summary and Frequency distributions of the columns\n- 🧹  Dimension/Weight Analyzer Agent will clean the data, fetch Weight & Density threshold from LLM and creates golden/true dataset\n- 🔍 Agent finds best match for missing/anomalous values within true dataset on product code and description\n- 🧠 Agent calls LLM to predict for unmapped products""")
                else:
                    st.warning("Could not determine a clear plan of action. Please check your column mappings. Agents will run based on the data provided.", icon="⚠️")
                    
                
                # 1. Pre-Processing EDA
                st.subheader("📊 Processing EDA on Raw Mapped Data")
                if any(col in df_renamed.columns for col in skus_columns) or any(col in df_renamed.columns for col in order_columns):
                    eda_summary_pre, order_summary_pre = eda_agent(client, deployment_name, df_renamed.copy())
                    st.session_state.pre_process_eda_summary_agent_output = eda_summary_pre
                    st.session_state.pre_process_order_summary_agent_output = order_summary_pre
                    st.session_state.eda_summary_json = json.dumps({
                        "pre_processing_summary": {
                            "eda_summary": eda_summary_pre,
                            "order_summary": order_summary_pre
                        }
                    }, indent=4)

                # 2. File Standardization
                final_df_pincode = None
                if any(col in df_renamed.columns for col in pincode_columns):
                    has_pin_1, has_pin_2 = any(col in df_renamed.columns for col in pin_1), any(col in df_renamed.columns for col in pin_2)
                    if has_pin_1 and not has_pin_2: final_df_pincode = pincode_mapping_agent(client, deployment_name, df_renamed.copy())
                    elif not has_pin_1 and has_pin_2: final_df_pincode = pincode_mapping_agent(client, deployment_name, df_renamed.copy())
                    elif has_pin_1 and has_pin_2:
                        initial_raw_df, unique_od_pair_df, unique_pin_city_state_df = unique_pin_city_state(df_renamed.copy())
                        df_input_processed = pincode_mapping_agent(client, deployment_name, unique_pin_city_state_df)
                        initial_data_OD_lat_long = origin_destination_mapping(df_input_processed, unique_od_pair_df)
                        #### Calculating distance using Google API
                        st.write("Fetching distance between origin and destination through Google API. Please wait...")
                        # rows = [row for _, row in initial_data_OD_lat_long.iterrows()]
                        # distance_km = parallel_apply_with_progress_osrm(lambda row: get_distance_from_osrm(row), rows)
                        # distance_df = pd.DataFrame([item.tolist() if item is not None else [None] for item in distance_km], columns=['distance_km'])
                        # initial_data_OD_lat_long['distance_km'] = distance_df
                        google_distance_df = get_distance_batched(df=initial_data_OD_lat_long, api_key=google_distance_api_key, batch_size=10)

                        initial_data_OD_lat_long = initial_data_OD_lat_long.merge(google_distance_df, on = ['mapped_pincode_origin_1','mapped_pincode'],how = 'left')

                        final_df_pincode = join_raw_df(initial_raw_df, initial_data_OD_lat_long).fillna('0').replace('<NA>','0').replace('nan','0')
                    st.session_state['pincode_map_button'] = final_df_pincode
                
                df_weight_dimensions_result = None
                # Check for the new scenarios for dimension/weight processing
                if 'product_desc' in df_renamed.columns and len(df_renamed.columns) == 1:
                    df_weight_dimensions_result = missing_weight_dimensions_estimator_agent(client, deployment_name, df_renamed.copy())
                    st.session_state['missing_value_button'] = df_weight_dimensions_result
                elif 'product_code' in df_renamed.columns and (all(col in df_renamed.columns for col in ['length', 'width', 'height', 'product_weight']) or all(col in df_renamed.columns for col in ['product_dimension', 'product_weight'])):
                    has_cols_1 = all(c in df_renamed.columns for c in sku_cols_1)
                    has_cols_2 = all(c in df_renamed.columns for c in sku_cols_2)
                    is_missing_or_zero = False
                    if has_cols_1:
                        is_missing_or_zero = df_renamed[sku_cols_1].isnull().any().any() or (df_renamed[sku_cols_1] == 0).any().any()
                    elif has_cols_2:
                        is_missing_or_zero = df_renamed[sku_cols_2].isnull().any().any() or (df_renamed[sku_cols_2] == 0).any().any()

                    if (has_cols_1 or has_cols_2) and not is_missing_or_zero:
                        df_weight_dimensions_result = weight_anomaly_detection(client, deployment_name, df_renamed.copy())
                        st.session_state['anomaly_button'] = df_weight_dimensions_result
                    else:
                        df_weight_dimensions_result = missing_weight_dimensions_estimator_agent(client, deployment_name, df_renamed.copy())
                        st.session_state['missing_value_button'] = df_weight_dimensions_result
                elif any(col in df_renamed.columns for col in skus_columns): # Original SKU processing logic
                    has_cols_1 = all(c in df_renamed.columns for c in sku_cols_1)
                    has_cols_2 = all(c in df_renamed.columns for c in sku_cols_2)

                    # Determine if the existing columns have missing or zero values
                    is_missing_or_zero = False
                    if has_cols_1:
                        # Check the first set of columns for nulls or zeros
                        is_missing_or_zero = df_renamed[sku_cols_1].isnull().any().any() or (df_renamed[sku_cols_1] == 0).any().any()
                    elif has_cols_2:
                        # If the first set isn't there, check the second set for nulls or zeros
                        is_missing_or_zero = df_renamed[sku_cols_2].isnull().any().any() or (df_renamed[sku_cols_2] == 0).any().any()

                    # If the necessary columns exist AND they are clean (not missing and not zero)...
                    if (has_cols_1 or has_cols_2) and not is_missing_or_zero:
                        # ...then run the anomaly detector.
                        df_weight_dimensions_result = weight_anomaly_detection(client, deployment_name, df_renamed.copy())
                        st.session_state['anomaly_button'] = df_weight_dimensions_result
                    # For all other cases...
                    else:
                        # ...run the missing value estimator.
                        # This block now correctly handles two scenarios:
                        #   1. The columns exist but contain missing or zero values.
                        #   2. The required columns do not exist in the dataframe at all.
                        df_weight_dimensions_result = missing_weight_dimensions_estimator_agent(client, deployment_name, df_renamed.copy())
                        st.session_state['missing_value_button'] = df_weight_dimensions_result
                
                # 3. Join Results
                df1, df2, df3 = st.session_state.get('missing_value_button'), st.session_state.get('anomaly_button'), st.session_state.get('pincode_map_button')
                final_all_joined = join_all_df(df1, df2, df3, list(df_input.columns))
                
                if final_all_joined is not None:
                    st.session_state.scs_processed_df = final_all_joined
                    st.session_state.scs_original_filename = uploaded_file.name
                else:
                    st.session_state.scs_processed_df = None
        
        elif run_eda_agent:
            df_renamed = st.session_state.df_renamed
            with st.spinner("📊 Running EDA Agent..."):
                if any(col in df_renamed.columns for col in skus_columns) or any(col in df_renamed.columns for col in order_columns) or ('product_desc' in df_renamed.columns and len(df_renamed.columns) == 1) or ('product_code' in df_renamed.columns and (all(col in df_renamed.columns for col in ['length', 'width', 'height', 'product_weight']) or all(col in df_renamed.columns for col in ['product_dimension', 'product_weight']))):
                    st.subheader("📊 Pre-Processing EDA on Raw Mapped Data")
                    eda_summary, order_summary = eda_agent(client, deployment_name, df_renamed.copy())
                    
                    combined_summary = {
                        "eda_summary": eda_summary,
                        "order_summary": order_summary
                    }
                    st.session_state.eda_summary_json = json.dumps(combined_summary, indent=4)
                    st.success("✅ EDA complete. You can now download the summary.")
                else:
                    st.warning("No relevant columns found for EDA Agent.")

        elif run_standardiser_agent:
            df_renamed = st.session_state.df_renamed
            with st.spinner("🧹 Running File Standardiser Agent..."):
                # Display Plan of Action
                st.subheader("🤖 Plan of Action")
                df_to_check = df_renamed
                has_orders = any(col in df_to_check.columns for col in order_columns)
                has_skus = any(col in df_to_check.columns for col in skus_columns)
                has_dest_pin = any(col in df_to_check.columns for col in pin_1)
                has_origin_pin = any(col in df_to_check.columns for col in pin_2)
                sku_cols_1, sku_cols_2 = ["product_dimension", "product_weight"], ["length", "width", "height", "product_weight"]
                sku_base = ["product_category","product_code", "product_desc", "pack_size"]
                # (Plan of action logic remains the same)

                # A) Only 'product_desc' uploaded
                if 'product_desc' in df_to_check.columns and len(df_to_check.columns) == 1:
                    with st.status("Plan of Action", expanded=True):
                        st.markdown("""
                        - ✅ Found only 'product_desc' column.
                        - 🧠 Agent will call LLM to predict missing dimensions and weight based on product description.
                        """)
                    df_weight_dimensions_result = missing_weight_dimensions_estimator_agent(client, deployment_name, df_renamed.copy())
                    st.session_state['missing_value_button'] = df_weight_dimensions_result

                # B) 'product_code' along with 'weight' and 'dimension'
                elif 'product_code' in df_to_check.columns and (all(col in df_to_check.columns for col in ['length', 'width', 'height', 'product_weight']) or all(col in df_to_check.columns for col in ['product_dimension', 'product_weight'])):
                    has_cols_1 = all(c in df_to_check.columns for c in sku_cols_1)
                    has_cols_2 = all(c in df_to_check.columns for c in sku_cols_2)
                    is_missing_or_zero = False
                    if has_cols_1:
                        is_missing_or_zero = df_to_check[sku_cols_1].isnull().any().any() or (df_to_check[sku_cols_1] == 0).any().any()
                    elif has_cols_2:
                        is_missing_or_zero = df_to_check[sku_cols_2].isnull().any().any() or (df_to_check[sku_cols_2] == 0).any().any()

                    if (has_cols_1 or has_cols_2) and not is_missing_or_zero:
                        with st.status("Plan of Action", expanded=True):
                            st.markdown("""
                            - ✅ Found 'product_code' along with complete Weight and Dimension data.
                            - 🤖 EDA and Anomaly Agent will be Activated.
                            - 📊 EDA Agent will show Data Summary and Frequency distributions of the columns.
                            - 🧹 Dimension/Weight Analyzer Agent will clean the data, fetch Weight & Density threshold from LLM and creates golden/true dataset.
                            - 🔍 Agent finds the anomalous weight/dimension products based on Weight and Density thresholds.
                            """)
                        df_weight_dimensions_result = weight_anomaly_detection(client, deployment_name, df_renamed.copy())
                        st.session_state['anomaly_button'] = df_weight_dimensions_result
                    else: # This covers cases where 'product_code' + (weight/dimension) are present but incomplete/have zeros
                        with st.status("Plan of Action", expanded=True):
                            st.markdown("""
                            - ✅ Found 'product_code' along with incomplete/zero Weight and Dimension data.
                            - 🤖 EDA and Dimension/Weight Analyzer Agent will be Activated.
                            - 📊 EDA Agent will show Data Summary and Frequency distributions of the columns.
                            - 🧹 Dimension/Weight Analyzer Agent will clean the data, fetch Weight & Density threshold from LLM and creates golden/true dataset.
                            - 🔍 Agent finds best match for missing/anomalous values within true dataset on product code and description.
                            - 🧠 Agent calls LLM to predict for unmapped products.
                            """)
                        df_weight_dimensions_result = missing_weight_dimensions_estimator_agent(client, deployment_name, df_renamed.copy())
                        st.session_state['missing_value_button'] = df_weight_dimensions_result

                # ... (rest of your existing if/elif conditions for other scenarios)
                elif has_orders:
                    nulls_detail = missing_col_null_check(df_to_check, order_columns)
                    with st.status("Plan of Action", expanded=True): st.markdown(f"""- ✅ Found columns for Order data\n- **Details:** `{nulls_detail}`\n- 🤖 EDA Agent will be activated\n- 📊  EDA Agent will show order Summary, order pareto analysis and date level trend\n- 🤖 Super cleaning Agent will be activated\n- 🗺️ Agent will map clean the pincode, find anomalies and predict missing weight and dimension based on data""")
                elif has_dest_pin and has_origin_pin and not has_skus:
                    nulls_detail = missing_col_null_check(df_to_check, pincode_columns)
                    with st.status("Plan of Action", expanded=True): st.markdown(f"""- ✅ Found columns for Origin and destination Pincode/City/State\n- **Details:** `{nulls_detail}`\n- 🤖 Pincode Distance Agent will be Activated\n- 🗺️ Agent will map the origin and destination data to delhivery internal data: pincode, city, state and TAT city\n- 📍 Pincode Agent will fetch Latitude and Longitude using LLM\n- 📏 Agent will calculate distance between origin and destination pincode using OSRM API""")
                elif not has_dest_pin and not has_origin_pin and has_skus:
                    has_cols_1 = all(col in df_to_check.columns for col in sku_cols_1)
                    has_cols_2 = all(col in df_to_check.columns for col in sku_cols_2)
                    if has_cols_1 and not (df_to_check[sku_cols_1].isnull().any() | (df_to_check[sku_cols_1] == 0).any()).any():
                        req_cols = sku_base + sku_cols_1
                        nulls_detail = missing_col_null_check(df_to_check, req_cols)
                        with st.status("Plan of Action", expanded=True): st.markdown(f"""- ✅ Found columns for Product 📦 Weights and Dimension\n- **Details:** `{nulls_detail}`\n- 🤖 EDA and Anomaly Agent will be Activated\n- 📊  EDA Agent will show Data Summary and Frequency distributions of the columns\n- 🧹  Dimension/Weight Analyzer Agent will clean the data, fetch Weight & Density threshold from LLM and creates golden/true dataset\n- 🔍 Agent finds the anomalous weight/dimension products based on Weight and Density thresholds""")
                    elif has_cols_2 and not (df_to_check[sku_cols_2].isnull().any() | (df_to_check[sku_cols_2] == 0).any()).any():
                        req_cols = sku_base + sku_cols_2
                        nulls_detail = missing_col_null_check(df_to_check, req_cols)
                        with st.status("Plan of Action", expanded=True): st.markdown(f"""- ✅ Found columns for Product 📦 Weights and Dimension\n- **Details:** `{nulls_detail}`\n- 🤖 EDA and Anomaly Agent will be Activated\n- 📊  EDA Agent will show Data Summary and Frequency distributions of the columns\n- 🧹  Dimension/Weight Analyzer Agent will clean the data, fetch Weight & Density threshold from LLM and creates golden/true dataset\n- 🔍 Agent finds the anomalous weight/dimension products based on Weight and Density thresholds""")
                    else:
                        if has_cols_1: 
                            req_cols = sku_base + sku_cols_1
                        elif has_cols_2: 
                            req_cols = sku_base + sku_cols_2
                        else:
                            req_cols = sku_base  # fallback to base columns if neither cols_1 nor cols_2 are available
                        nulls_detail = missing_col_null_check(df_to_check, req_cols)
                        with st.status("Plan of Action", expanded=True): st.markdown(f"""- ✅ Found columns for Product 📦 Weights and Dimension\n- **Details:** `{nulls_detail}`\n- 🤖 EDA and Dimension/Weight Analyzer Agent will be Activated\n- 📊  EDA Agent will show Data Summary and Frequency distributions of the columns\n- 🧹  Dimension/Weight Analyzer Agent will clean the data, fetch Weight & Density threshold from LLM and creates golden/true dataset\n- 🔍 Agent finds best match for missing/anomalous values within true dataset on product code and description\n- 🧠 Agent calls LLM to predict for unmapped products""")
                elif has_dest_pin and not has_origin_pin and has_skus:
                    has_cols_1 = all(col in df_to_check.columns for col in sku_cols_1)
                    has_cols_2 = all(col in df_to_check.columns for col in sku_cols_2)
                    if has_cols_1 and not (df_to_check[sku_cols_1].isnull().any() | (df_to_check[sku_cols_1] == 0).any()).any():
                        req_cols = list(pin_1) + sku_base + sku_cols_1
                        nulls_detail = missing_col_null_check(df_to_check, req_cols)
                        with st.status("Plan of Action", expanded=True): st.markdown(f"""- ✅ Found columns for Pincode/City/State\n- **Details:** `{nulls_detail}`\n- 🤖 Pincode Agent will be Activated\n- 🗺️ Agent will map the data to Delhivery internal data: pincode, city, state, and TAT city\n- 📍 Agent will fetch Latitude and Longitude using LLM\n- ✅ Found columns for Product 📦 Weights and Dimension\n- 🤖 EDA and Anomaly Agent will be Activated\n- 📊  EDA Agent will show Data Summary and Frequency distributions of the columns\n- 🧹  Dimension/Weight Analyzer Agent will clean the data, fetch Weight & Density threshold from LLM and creates golden/true dataset\n- 🔍 Agent finds the anomalous weight/dimension products based on Weight and Density thresholds""")
                    elif has_cols_2 and not (df_to_check[sku_cols_2].isnull().any() | (df_to_check[sku_cols_2] == 0).any()).any():
                        req_cols = list(pin_1) + sku_base + sku_cols_2
                        nulls_detail = missing_col_null_check(df_to_check, req_cols)
                        with st.status("Plan of Action", expanded=True): st.markdown(f"""- ✅ Found columns for Pincode/City/State\n- **Details:** `{nulls_detail}`\n- 🤖 Pincode Agent will be Activated\n- 🗺️ Agent will map the data to Delhivery internal data: pincode, city, state, and TAT city\n- 📍 Agent will fetch Latitude and Longitude using LLM\n- ✅ Found columns for Product 📦 Weights and Dimension\n- 🤖 EDA and Anomaly Agent will be Activated\n- 📊  EDA Agent will show Data Summary and Frequency distributions of the columns\n- 🧹  Dimension/Weight Analyzer Agent will clean the data, fetch Weight & Density threshold from LLM and creates golden/true dataset\n- 🔍 Agent finds the anomalous weight/dimension products based on Weight and Density thresholds""")
                    else:
                        if has_cols_1: 
                            req_cols = list(pin_1) + sku_base + sku_cols_1
                        elif has_cols_2: 
                            req_cols = list(pin_1) + sku_base + sku_cols_2
                        else:
                            req_cols = list(pin_1) + sku_base  # fallback to pin and base columns if neither cols_1 nor cols_2 are available
                        nulls_detail = missing_col_null_check(df_to_check, req_cols)
                        with st.status("Plan of Action", expanded=True): st.markdown(f"""- ✅ Found columns for Pincode/City/State\n- **Details:** `{nulls_detail}`\n- 🤖 Pincode Agent will be Activated\n- 🗺️ Agent will map the data to Delhivery internal data: pincode, city, state, and TAT city\n- 📍 Agent will fetch Latitude and Longitude using LLM\n- ✅ Found columns for Product 📦 Weights and Dimension\n- 🤖 EDA and Dimension/Weight Analyzer Agent will be Activated\n- 📊  EDA Agent will show Data Summary and Frequency distributions of the columns\n- 🧹  Dimension/Weight Analyzer Agent will clean the data, fetch Weight & Density threshold from LLM and creates golden/true dataset\n- 🔍 Agent finds best match for missing/anomalous values within true dataset on product code and description\n- 🧠 Agent calls LLM to predict for unmapped products""")
                elif has_dest_pin and has_origin_pin and has_skus:
                    has_cols_1 = all(col in df_to_check.columns for col in sku_cols_1)
                    has_cols_2 = all(col in df_to_check.columns for col in sku_cols_2)
                    if has_cols_1 and not (df_to_check[sku_cols_1].isnull().any() | (df_to_check[sku_cols_1] == 0).any()).any():
                        req_cols = pincode_columns + sku_base + sku_cols_1
                        nulls_detail = missing_col_null_check(df_to_check, req_cols)
                        with st.status("Plan of Action", expanded=True): st.markdown(f"""- ✅ Found columns for Origin and destination Pincode/City/State\n- **Details:** `{nulls_detail}`\n- 🤖 Pincode Distance Agent will be Activated\n- 🗺️ Agent will map the origin and destination data to delhivery internal data: pincode, city, state and TAT city\n- 📍 Pincode Agent will fetch Latitude and Longitude using LLM\n- 📏 Agent will calculate distance between origin and destination pincode using OSRM API\n- ✅ Found columns for Product 📦 Weights and Dimension\n- 🤖 EDA and Anomaly Agent will be Activated\n- 📊  EDA Agent will show Data Summary and Frequency distributions of the columns\n- 🧹  Dimension/Weight Analyzer Agent will clean the data, fetch Weight & Density threshold from LLM and creates golden/true dataset\n- 🔍 Agent finds the anomalous weight/dimension products based on Weight and Density thresholds""")
                    elif has_cols_2 and not (df_to_check[sku_cols_2].isnull().any() | (df_to_check[sku_cols_2] == 0).any()).any():
                        req_cols = pincode_columns + sku_base + sku_cols_2
                        nulls_detail = missing_col_null_check(df_to_check, req_cols)
                        with st.status("Plan of Action", expanded=True): st.markdown(f"""- ✅ Found columns for Origin and destination Pincode/City/State\n- **Details:** `{nulls_detail}`\n- 🤖 Pincode Distance Agent will be Activated\n- 🗺️ Agent will map the origin and destination data to delhivery internal data: pincode, city, state and TAT city\n- 📍 Pincode Agent will fetch Latitude and Longitude using LLM\n- 📏 Agent will calculate distance between origin and destination pincode using OSRM API\n- ✅ Found columns for Product 📦 Weights and Dimension\n- 🤖 EDA and Anomaly Agent will be Activated\n- 📊  EDA Agent will show Data Summary and Frequency distributions of the columns\n- 🧹  Dimension/Weight Analyzer Agent will clean the data, fetch Weight & Density threshold from LLM and creates golden/true dataset\n- 🔍 Agent finds the anomalous weight/dimension products based on Weight and Density thresholds""")
                    else:
                        if has_cols_1: 
                            req_cols = pincode_columns + sku_base + sku_cols_1
                        elif has_cols_2: 
                            req_cols = pincode_columns + sku_base + sku_cols_2
                        else:
                            req_cols = pincode_columns + sku_base  # fallback to pincode and base columns if neither cols_1 nor cols_2 are available
                        nulls_detail = missing_col_null_check(df_to_check, req_cols)
                        with st.status("Plan of Action", expanded=True): st.markdown(f"""- ✅ Found columns for Origin and destination Pincode/City/State\n- **Details:** `{nulls_detail}`\n- 🤖 Pincode Distance Agent will be Activated\n- 🗺️ Agent will map the origin and destination data to delhivery internal data: pincode, city, state and TAT city\n- 📍 Pincode Agent will fetch Latitude and Longitude using LLM\n- 📏 Agent will calculate distance between origin and destination pincode using OSRM API\n- ✅ Found columns for Product 📦 Weights and Dimension\n- 🤖 EDA and Dimension/Weight Analyzer Agent will be Activated\n- 📊  EDA Agent will show Data Summary and Frequency distributions of the columns\n- 🧹  Dimension/Weight Analyzer Agent will clean the data, fetch Weight & Density threshold from LLM and creates golden/true dataset\n- 🔍 Agent finds best match for missing/anomalous values within true dataset on product code and description\n- 🧠 Agent calls LLM to predict for unmapped products""")
                else:
                    st.warning("Could not determine a clear plan of action. Please check your column mappings. Agents will run based on the data provided.", icon="⚠️")

                # Run Standardization Agents
                final_df_pincode = None
                if any(col in df_renamed.columns for col in pincode_columns):
                    has_pin_1, has_pin_2 = any(col in df_renamed.columns for col in pin_1), any(col in df_renamed.columns for col in pin_2)
                    if has_pin_1 and not has_pin_2: final_df_pincode = pincode_mapping_agent(client, deployment_name, df_renamed.copy())
                    elif not has_pin_1 and has_pin_2: final_df_pincode = pincode_mapping_agent(client, deployment_name, df_renamed.copy())
                    elif has_pin_1 and has_pin_2:
                        initial_raw_df, unique_od_pair_df, unique_pin_city_state_df = unique_pin_city_state(df_renamed.copy())
                        df_input_processed = pincode_mapping_agent(client, deployment_name, unique_pin_city_state_df)
                        initial_data_OD_lat_long = origin_destination_mapping(df_input_processed, unique_od_pair_df)

                        # st.write("Fetching distance between origin and destination through API. Please wait...")
                        # rows = [row for _, row in initial_data_OD_lat_long.iterrows()]
                        # distance_km = parallel_apply_with_progress_osrm(lambda row: get_distance_from_osrm(row), rows)
                        # distance_df = pd.DataFrame([item.tolist() if item is not None else [None] for item in distance_km], columns=['distance_km'])
                        # initial_data_OD_lat_long['distance_km'] = distance_df

                        #### Calculating distance using Google API
                        st.write("Fetching distance between origin and destination through Google API. Please wait...")
                        # rows = [row for _, row in initial_data_OD_lat_long.iterrows()]
                        # distance_km = parallel_apply_with_progress_osrm(lambda row: get_distance_from_osrm(row), rows)
                        # distance_df = pd.DataFrame([item.tolist() if item is not None else [None] for item in distance_km], columns=['distance_km'])
                        # initial_data_OD_lat_long['distance_km'] = distance_df
                        google_distance_df = get_distance_batched(df=initial_data_OD_lat_long, api_key=google_distance_api_key, batch_size=10)
                        initial_data_OD_lat_long = initial_data_OD_lat_long.merge(google_distance_df, on = ['mapped_pincode_origin_1','mapped_pincode'],how = 'left')

                        final_df_pincode = join_raw_df(initial_raw_df, initial_data_OD_lat_long).fillna('0').replace('<NA>','0').replace('nan','0')
                    st.session_state['pincode_map_button'] = final_df_pincode

                df_weight_dimensions_result = None
                # Check for the new scenarios for dimension/weight processing
                if 'product_desc' in df_renamed.columns and len(df_renamed.columns) == 1:
                    df_weight_dimensions_result = missing_weight_dimensions_estimator_agent(client, deployment_name, df_renamed.copy())
                    st.session_state['missing_value_button'] = df_weight_dimensions_result
                elif 'product_code' in df_renamed.columns and (all(col in df_renamed.columns for col in ['length', 'width', 'height', 'product_weight']) or all(col in df_renamed.columns for col in ['product_dimension', 'product_weight'])):
                    has_cols_1 = all(c in df_renamed.columns for c in sku_cols_1)
                    has_cols_2 = all(c in df_renamed.columns for c in sku_cols_2)
                    is_missing_or_zero = False
                    if has_cols_1:
                        is_missing_or_zero = df_renamed[sku_cols_1].isnull().any().any() or (df_renamed[sku_cols_1] == 0).any().any()
                    elif has_cols_2:
                        is_missing_or_zero = df_renamed[sku_cols_2].isnull().any().any() or (df_renamed[sku_cols_2] == 0).any().any()

                    if (has_cols_1 or has_cols_2) and not is_missing_or_zero:
                        df_weight_dimensions_result = weight_anomaly_detection(client, deployment_name, df_renamed.copy())
                        st.session_state['anomaly_button'] = df_weight_dimensions_result
                    else:
                        df_weight_dimensions_result = missing_weight_dimensions_estimator_agent(client, deployment_name, df_renamed.copy())
                        st.session_state['missing_value_button'] = df_weight_dimensions_result
                elif any(col in df_renamed.columns for col in skus_columns): # Original SKU processing logic
                    has_cols_1 = all(c in df_renamed.columns for c in sku_cols_1)
                    has_cols_2 = all(c in df_renamed.columns for c in sku_cols_2)


                    # Determine if the existing columns have missing or zero values
                    is_missing_or_zero = False
                    if has_cols_1:
                        # Check the first set of columns for nulls or zeros
                        is_missing_or_zero = df_renamed[sku_cols_1].isnull().any().any() or (df_renamed[sku_cols_1] == 0).any().any()
                    elif has_cols_2:
                        # If the first set isn't there, check the second set
                        is_missing_or_zero = df_renamed[sku_cols_2].isnull().any().any() or (df_renamed[sku_cols_2] == 0).any().any()

                    # If the necessary columns exist AND they are clean...
                    if (has_cols_1 or has_cols_2) and not is_missing_or_zero:
                        # ...run the anomaly detector.
                        df_weight_dimensions_result = weight_anomaly_detection(client, deployment_name, df_renamed.copy())
                        st.session_state['anomaly_button'] = df_weight_dimensions_result
                    # For all other cases...
                    else:
                        # ...run the missing value estimator.
                        # This block now covers both cases:
                        # 1. Columns exist but have missing/zero values.
                        # 2. The required columns do not exist at all.
                        df_weight_dimensions_result = missing_weight_dimensions_estimator_agent(client, deployment_name, df_renamed.copy())
                        st.session_state['missing_value_button'] = df_weight_dimensions_result

                
                # Join Results
                df1, df2, df3 = st.session_state.get('missing_value_button'), st.session_state.get('anomaly_button'), st.session_state.get('pincode_map_button')
                final_all_joined = join_all_df(df1, df2, df3, list(df_input.columns))
                
                if final_all_joined is not None:
                    st.session_state.scs_processed_df = final_all_joined
                    st.session_state.scs_original_filename = uploaded_file.name
                else:
                    st.session_state.scs_processed_df = None

    # --- EDA Summary Download (if available) ---
    if st.session_state.eda_summary_json:
        st.download_button(
            label="📥 Download EDA Summary (JSON)",
            data=st.session_state.eda_summary_json.encode('utf-8'),
            file_name=f"eda_summary_{os.path.splitext(uploaded_file.name)[0]}.json" if uploaded_file else "eda_summary.json",
            mime="application/json"
        )

    # --- RESULTS DISPLAY AND NAVIGATION (for Super-Agent and Standardiser Agent) ---
    if st.session_state.scs_processed_df is not None:
        st.markdown("---")
        
        if 'pre_process_eda_summary_agent_output' in st.session_state and st.session_state.pre_process_eda_summary_agent_output:
            st.subheader("📊 Post-Processing EDA on Final Data")
            if any(col in st.session_state.scs_processed_df.columns for col in skus_columns) or any(col in st.session_state.scs_processed_df.columns for col in order_columns):
                with st.spinner("Running EDA on the final processed data..."):
                    eda_summary_post, order_summary_post = eda_agent(client, deployment_name, st.session_state.scs_processed_df.copy())
                    st.session_state.post_process_eda_summary_agent_output = eda_summary_post
                    st.session_state.post_process_order_summary_agent_output = order_summary_post
                    
                    if st.session_state.eda_summary_json:
                        try:
                            summary_data = json.loads(st.session_state.eda_summary_json)
                        except (json.JSONDecodeError, TypeError):
                            summary_data = {}
                        
                        summary_data["post_processing_summary"] = {
                            "eda_summary": eda_summary_post,
                            "order_summary": order_summary_post
                        }
                        st.session_state.eda_summary_json = json.dumps(summary_data, indent=4)
            else:
                st.info("Skipping Post-Processing EDA as relevant columns are not present.")
            st.session_state.pre_process_eda_summary_agent_output = None

        st.subheader("✅ SCS Processing Complete!")
        st.markdown("---")
        st.subheader("📦 Volume Calculation and Download")
        final_df = st.session_state.scs_processed_df.copy()
        final_df = add_velocity_column(final_df)
        
        if all(c in final_df.columns for c in ['length', 'width', 'height']):
            for col in ['length', 'width', 'height']: final_df[col] = pd.to_numeric(final_df[col], errors='coerce').fillna(0)
            final_df['Volume'] = final_df['length'] * final_df['width'] * final_df['height']
            st.info("✅ 'Volume' column created/updated.")
        
        if all(c in final_df.columns for c in ['predicted_length', 'predicted_width', 'predicted_height']):
            for col in ['predicted_length', 'predicted_width', 'predicted_height']: final_df[col] = pd.to_numeric(final_df[col], errors='coerce').fillna(0)
            final_df['predicted_volume'] = final_df['predicted_length'] * final_df['predicted_width'] * final_df['predicted_height']
            st.info("✅ 'predicted_volume' column created/updated.")
        
        st.success("✅ File Standardization Complete!")
        st.dataframe(final_df.head())

        output_excel = BytesIO()
        with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
            final_df.to_excel(writer, index=False, sheet_name='SCS_Processed_Data')
        output_excel.seek(0)

        # --- Display Buttons in a three-column layout ---
        col_dl_proc, col_dl_summaries, col_tms_flow = st.columns(3)

        with col_dl_proc:
            st.download_button(
                label="📥 Download Processed Data",
                data=output_excel,
                file_name=f"processed_{st.session_state.scs_original_filename}.xlsx",
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                use_container_width=True
            )
        
        with col_dl_summaries:
            # --- Combined EDA Summary Download ---
            pre_eda = st.session_state.get('pre_process_eda_summary_agent_output')
            post_eda = st.session_state.get('post_process_eda_summary_agent_output')
            
            if pre_eda or post_eda:
                combined_eda_data = {}
                # Split the string into a list of lines for readability
                if pre_eda: combined_eda_data["pre_processed_eda_summary"] = pre_eda.split('\n')
                if post_eda: combined_eda_data["post_processed_eda_summary"] = post_eda.split('\n')
                
                eda_json_output = json.dumps(combined_eda_data, indent=4)
                
                st.download_button(
                    label="📊 Download Combined EDA Summary",
                    data=eda_json_output.encode('utf-8'),
                    file_name=f"combined_eda_summary_{os.path.splitext(st.session_state.scs_original_filename)[0]}.json",
                    mime="application/json",
                    key="download_overall_eda_summary",
                    use_container_width=True
                )
            
            # --- Combined Order Summary Download ---
            pre_order = st.session_state.get('pre_process_order_summary_agent_output')
            post_order = st.session_state.get('post_process_order_summary_agent_output')

            if pre_order or post_order:
                combined_order_data = {}
                # Split the string into a list of lines for readability
                if pre_order: combined_order_data["pre_processed_order_summary"] = pre_order.split('\n')
                if post_order: combined_order_data["post_processed_order_summary"] = post_order.split('\n')
                
                order_json_output = json.dumps(combined_order_data, indent=4)

                st.download_button(
                    label="📋 Download Combined Order Summary",
                    data=order_json_output.encode('utf-8'),
                    file_name=f"combined_order_summary_{os.path.splitext(st.session_state.scs_original_filename)[0]}.json",
                    mime="application/json",
                    key="download_overall_order_summary",
                    use_container_width=True
                )

        with col_tms_flow:
            if st.button("▶️ Process with TMS Workflow", use_container_width=True):
                st.session_state.tms_input_from_scs = st.session_state.scs_processed_df.copy()
                st.session_state.tms_source_is_scs = True
                st.session_state.page = "TMS: Process Order File"
                # Reset SCS state for a new run
                st.session_state.scs_processed_df = None
                st.session_state.scs_llm_response = None
                st.session_state.df_renamed = None
                st.session_state.uploader_key += 1
                st.rerun()


        # dl_col1, dl_col2 = st.columns([1, 1.5])
        # dl_col1.download_button(
        #     label="📥 Download Processed Data",
        #     data=output_excel,
        #     file_name=f"processed_{st.session_state.scs_original_filename}.xlsx",
        #     mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        # )
        
       

        # if dl_col2.button("▶️ Process this with TMS Workflow"):
        #     st.session_state.tms_input_from_scs = st.session_state.scs_processed_df.copy()
        #     st.session_state.tms_source_is_scs = True
        #     st.session_state.page = "TMS: Process Order File"
        #     st.session_state.scs_processed_df = None
        #     st.session_state.scs_llm_response = None
        #     st.session_state.df_renamed = None
        #     st.session_state.uploader_key += 1
        #     st.rerun()

# ... (render_tms_home_page, render_order_cleaner_page, etc. would be here)
# Note: These functions would also be refactored for clarity and robustness.

def render_tms_home_page():
    st.title("🚚 Consolidated TMS Planning Suite")
    st.markdown("Welcome! This application combines three powerful tools into a single, streamlined workflow:")
    st.markdown("""
    1.  **Order File Cleaner**: Upload and standardize your raw order files.
    2.  **Pricing File Cleaner**: Process and standardize complex pricing sheets.
    3.  **Product Information Cleaner**: Upload and standardize product data with dimensions and weights.
    4.  **TMS Planner**: Use the cleaned data to configure and trigger a TMS planning run, then analyze the results.

    **Please use the sidebar to navigate between the steps.**
    """)
    st.info("Start with Step 1 to process your order file.")
    
    st.subheader("Current Status")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.session_state.processed_orders_df is not None:
            st.success(f"✅ **Order file processed:** {len(st.session_state.processed_orders_df)} rows ready.")
        else:
            st.warning("🟡 **Order file not processed yet.**")
    with col2:
        if st.session_state.processed_pricing_df is not None:
            st.success(f"✅ **Pricing file processed:** {len(st.session_state.processed_pricing_df)} rows ready.")
        else:
            st.warning("🟡 **Pricing file not processed yet.**")
    with col3:
        if st.session_state.processed_product_df is not None:
            st.success(f"✅ **Product file processed:** {len(st.session_state.processed_product_df)} rows ready.")
        else:
            st.warning("🟡 **Product file not processed yet.**")


def render_order_cleaner_page(llm_client, llm_available, llm_deployment_name):
    st.title("TMS Order File Cleaner")
    st.write("➡️ Upload your order file, select a client, map columns, and let the GenIE enrich your data!")
    st.markdown("---")

    oc_config = st.session_state.config['order_cleaner']

    # Check if input is coming from SCS workflow
    if 'tms_input_from_scs' in st.session_state and st.session_state.tms_input_from_scs is not None:
        df_raw = st.session_state.tms_input_from_scs
        client_name = st.session_state.get('scs_client_name', 'SCS_Client')
        session_key = f"oc_scs_input_{client_name}"
        if session_key not in st.session_state:
            st.session_state[session_key] = {}
            st.session_state[session_key]['raw_df'] = df_raw
    else:
        # Normal file upload flow
        uploaded_file = st.file_uploader("📂 Choose an Order file (CSV, XLSX, XLS)", type=['csv', 'xlsx', 'xls'], key="order_uploader")
        selected_client_option = st.selectbox("👤 Select the Client", oc_config["predefined_clients"], key="order_client_select")
        client_name = st.text_input("📝 Or Enter Client Name:", key="order_client_text") if selected_client_option == 'Other client' else selected_client_option

        if not uploaded_file or not client_name:
            st.info("Please upload a file and select a client to begin.")
            return

        session_key = f"oc_{uploaded_file.name}_{client_name}"
        if session_key not in st.session_state:
            st.session_state[session_key] = {}

    # --- File Loading Logic ---
    if 'raw_df' not in st.session_state.get(session_key, {}):
        df_raw = None
        # This block runs once per new file/client combo
        if 'sheet_names' not in st.session_state.get(session_key, {}):
            file_name = uploaded_file.name
            if file_name.endswith('.csv'):
                df_raw = load_csv_from_upload(uploaded_file)
                if df_raw is not None:
                    st.session_state[session_key]['raw_df'] = df_raw
                    st.rerun()
            elif file_name.endswith(('.xlsx', '.xls')):
                try:
                    uploaded_file.seek(0)
                    xls = pd.ExcelFile(uploaded_file)
                    st.session_state[session_key]['sheet_names'] = xls.sheet_names
                    st.rerun()
                except Exception as e:
                    st.error(f"Error reading Excel file: {e}")
                    return
        
        # --- Sheet selection UI (if applicable) ---
        if 'sheet_names' in st.session_state.get(session_key, {}):
            st.subheader("Select Sheet")
            st.info("Your Excel file contains multiple sheets. Please select the one to process.")
            selected_sheet = st.selectbox(
                "Which sheet contains the order data?",
                st.session_state[session_key]['sheet_names'],
                key=f"{session_key}_sheet_select"
            )
            if st.button("Load Selected Sheet", key=f"{session_key}_load_sheet"):
                try:
                    uploaded_file.seek(0)
                    df_raw = pd.read_excel(uploaded_file, sheet_name=selected_sheet)
                    st.session_state[session_key]['raw_df'] = df_raw
                    del st.session_state[session_key]['sheet_names'] # Clean up
                    st.rerun()
                except Exception as e:
                    st.error(f"Error loading sheet '{selected_sheet}': {e}")
            return

    # --- Mapping and Processing Logic (only if raw_df is loaded) ---
    if 'raw_df' in st.session_state.get(session_key, {}):
        st.subheader("📜 Original Data Preview")
        st.dataframe(st.session_state[session_key]['raw_df'].head())

        if 'final_df' not in st.session_state.get(session_key, {}):
            st.subheader("🎯 Column Mapping Review")
            if 'mapping_suggestions' not in st.session_state[session_key]:
                st.session_state[session_key]['mapping_suggestions'] = oc_get_llm_mapping_suggestions(
                    st.session_state[session_key]['raw_df'].columns.tolist(), oc_config, llm_client, llm_deployment_name, client_name
                )
            
            mapping_state = st.session_state[session_key]['mapping_suggestions']
            input_options = ["-- Select or Leave Blank --"] + st.session_state[session_key]['raw_df'].columns.tolist()
            ui_columns = st.columns(3)
            for i, output_col in enumerate(oc_config["standard_direct_mappable_columns"]):
                with ui_columns[i % 3]:
                    is_critical = output_col in oc_config["initial_critical_columns"]
                    label = f"❗️'{output_col}' (Critical)" if is_critical else f"'{output_col}'"
                    llm_suggestion = mapping_state.get('mapped_columns', {}).get(output_col)
                    default_index = input_options.index(llm_suggestion) if llm_suggestion in input_options else 0
                    if llm_suggestion: st.caption(f"LLM Suggests: **{llm_suggestion}**")
                    st.selectbox(label, input_options, index=default_index, key=f"{session_key}_map_{output_col}")
                    if st.session_state[f"{session_key}_map_{output_col}"] == "-- Select or Leave Blank --":
                        if output_col == 'Unit of measurement': st.selectbox(f"↳ Default:", oc_config["default_uom_options"], key=f"{session_key}_default_uom")
                        elif output_col == 'Product Category': st.selectbox(f"↳ Default:", oc_config["default_product_category_options"], key=f"{session_key}_default_pc")
            
            if st.button("✅ Confirm Mapping & Process Data", key=f"{session_key}_process"):
                confirmed_mapping, defaults, missing_critical = {}, {}, []
                for col in oc_config["standard_direct_mappable_columns"]:
                    selection = st.session_state[f"{session_key}_map_{col}"]
                    if selection != "-- Select or Leave Blank --": confirmed_mapping[col] = selection
                    elif col in oc_config["initial_critical_columns"]: missing_critical.append(col)
                if missing_critical:
                    st.error(f"☠️ Please map all critical columns: {', '.join(missing_critical)}")
                else:
                    if 'Unit of measurement' not in confirmed_mapping and st.session_state[f"{session_key}_default_uom"] != "-- Select Default (Optional) --": defaults['Unit of measurement'] = st.session_state[f"{session_key}_default_uom"]
                    if 'Product Category' not in confirmed_mapping and st.session_state[f"{session_key}_default_pc"] != "-- Select Default (Optional) --": defaults['Product Category'] = st.session_state[f"{session_key}_default_pc"]
                    with st.spinner("Processing data..."):
                        df = oc_standardize_dataframe(st.session_state[session_key]['raw_df'], confirmed_mapping, defaults, client_name, oc_config)
                        df = oc_enrich_product_details_with_llm(df, client_name, llm_client, llm_deployment_name)
                        df = oc_perform_final_transforms(df)
                        df = oc_validate_and_enrich_data(df, st.session_state.config)
                        st.session_state[session_key]['final_df'] = df
                        st.session_state['initial_orders_df_for_reports'] = st.session_state[session_key]['raw_df'].copy() # Save for TMS reports
                        

        if 'final_df' in st.session_state.get(session_key, {}):
            final_df = st.session_state[session_key]['final_df']
            st.markdown("---"); st.subheader("🏁 Final Processed Data")
            if st.checkbox("Update Weight & Volume totals (multiply by Quantity)?", key="oc_recalc"):
                final_df = oc_calculate_total_weight_volume(final_df)
            st.dataframe(final_df.head(10))
            st.session_state.processed_orders_df = final_df # STORE IN GLOBAL STATE
            st.success(f"✅ Order data processed and stored in session. {len(final_df)} rows are ready.")
            csv_data = final_df.to_csv(index=False).encode('utf-8')
            if 'tms_input_from_scs' in st.session_state and st.session_state.tms_input_from_scs is not None:
                base_filename = os.path.splitext(st.session_state.get('scs_original_filename', 'tms_output'))[0]
            else:
                base_filename = os.path.splitext(uploaded_file.name)[0]
            st.download_button(label="📥 Download Standardized CSV", data=csv_data, file_name=f"{base_filename}_standardized.csv", mime='text/csv', type="primary")

def render_pricing_cleaner_page(llm_client, llm_available, deployment_name):
    st.title("Pricing File Cleaner")
    st.write("➡️ Upload your pricing file, select the sheet, map columns, and let the GenIE standardize your rates!")
    st.markdown("---")

    pc_config = st.session_state.config['pricing_cleaner']
    app_paths = st.session_state.config['paths']
    print(app_paths['pricing_ref_id'])
    
    # Force reload reference file to ensure we're using the new ref_id_file2.csv
    if 'pc_ref_id_df' not in st.session_state or st.session_state.pc_ref_id_df is None:
        st.session_state.pc_ref_id_df = pc_load_reference_data(app_paths["pricing_ref_id"])
    
    # Add debug info about the reference file
    if st.session_state.pc_ref_id_df is not None:
        ref_columns = list(st.session_state.pc_ref_id_df.columns)
        st.info(f"📋 Reference file loaded: {len(st.session_state.pc_ref_id_df)} records, Columns: {ref_columns}")
        
        # Check if it's the new format with Pin_code
        if 'Pin_code' in ref_columns:
            st.success("✅ New pincode-enabled reference file detected!")
        else:
            st.warning("⚠️ Using old reference file format (no Pin_code column)")
    else:
        st.error("Cannot proceed without the pricing reference file.")
        return

    uploaded_file = st.file_uploader("📂 Choose a Pricing file (CSV, XLSX, XLS)", type=['csv', 'xlsx', 'xls'], key="pricing_uploader")
    if not uploaded_file:
        st.info("Please upload a file to begin.")
        return
        
    session_key = f"pc_{uploaded_file.name}"
    if session_key not in st.session_state:
        
        st.session_state[session_key] = {}

    # --- File Loading Logic ---
    if 'df_for_mapping' not in st.session_state.get(session_key, {}):
        df_raw = None
        if 'sheet_names' not in st.session_state.get(session_key, {}):
            file_name = uploaded_file.name
            if file_name.endswith('.csv'):
                df_raw = load_csv_from_upload(uploaded_file)
            elif file_name.endswith(('.xlsx', '.xls')):
                try:
                    uploaded_file.seek(0)
                    xls = pd.ExcelFile(uploaded_file)
                    st.session_state[session_key]['sheet_names'] = xls.sheet_names
                    st.rerun()
                except Exception as e:
                    st.error(f"Error reading Excel file: {e}")
                    return
        
        if 'sheet_names' in st.session_state.get(session_key, {}):
            st.subheader("Select Sheet")
            st.info("Your Excel file contains multiple sheets. Please select the one to process.")
            selected_sheet = st.selectbox(
                "Which sheet contains the pricing data?",
                st.session_state[session_key]['sheet_names'],
                key=f"{session_key}_sheet_select"
            )
            if st.button("Load Selected Sheet", key=f"{session_key}_load_sheet"):
                try:
                    uploaded_file.seek(0)
                    df_raw = pd.read_excel(uploaded_file, sheet_name=selected_sheet)
                    del st.session_state[session_key]['sheet_names'] # Clean up
                    # df_raw is loaded, to proceed with the processing logic
                    raw_headers = [str(col).strip() for col in df_raw.columns]
                    pc_config = st.session_state.config['pricing_cleaner']
                    found_generic_truck_col = any(keyword in [h.lower() for h in raw_headers] for keyword in pc_config["generic_truck_keywords"])
                    df_for_mapping, data_was_transformed = df_raw.copy(), False
                    if not found_generic_truck_col and llm_client:
                        with st.spinner("🤖 AI analyzing file structure for wide format..."):
                            llm_col_flags = pc_get_llm_suggestions(llm_client, deployment_name, pc_get_llm_wide_format_detection, raw_headers, df_raw.head(3))
                        if llm_col_flags:
                            df_for_mapping, data_was_transformed = pc_transform_wide_to_long(df_raw, llm_col_flags)

                    st.session_state[session_key]['df_for_mapping'] = df_for_mapping
                    st.session_state[session_key]['data_was_transformed'] = data_was_transformed
                    if llm_client:
                        with st.spinner("🤖 AI suggesting column mappings..."):
                            st.session_state[session_key]['llm_suggestions'] = pc_get_llm_suggestions(
                                llm_client, deployment_name, pc_get_llm_column_mapping_suggestions, list(df_for_mapping.columns), pc_config["llm_mapping_target_keys"]
                            )
                    st.rerun()
                except Exception as e:
                    st.error(f"Error loading sheet '{selected_sheet}': {e}")
            return
        
        
        
        if df_raw is not None:
            st.session_state[session_key]['df_raw'] = df_raw
            raw_headers = [str(col).strip() for col in df_raw.columns]
            found_generic_truck_col = any(keyword in [h.lower() for h in raw_headers] for keyword in pc_config["generic_truck_keywords"])
            df_for_mapping, data_was_transformed = df_raw.copy(), False
            if not found_generic_truck_col and llm_client:
                with st.spinner("🤖 AI analyzing file structure for wide format..."):
                    llm_col_flags = pc_get_llm_suggestions(llm_client, deployment_name, pc_get_llm_wide_format_detection, raw_headers, df_raw.head(3))
                if llm_col_flags:
                    df_for_mapping, data_was_transformed = pc_transform_wide_to_long(df_raw, llm_col_flags)
            st.session_state[session_key]['df_for_mapping'] = df_for_mapping
            st.session_state[session_key]['data_was_transformed'] = data_was_transformed
            if llm_client:
                with st.spinner("🤖 AI suggesting column mappings..."):
                    st.session_state[session_key]['llm_suggestions'] = pc_get_llm_suggestions(
                        llm_client, deployment_name, pc_get_llm_column_mapping_suggestions, list(df_for_mapping.columns), pc_config["llm_mapping_target_keys"]
                    )

    # --- Mapping and Processing Logic (only if df_for_mapping is loaded) ---
    if 'df_for_mapping' in st.session_state.get(session_key, {}) and 'final_df' not in st.session_state.get(session_key, {}):
        st.subheader("📜 Preview of Data to be Mapped")
        st.dataframe(st.session_state[session_key]['df_for_mapping'].head())

        st.subheader("Pricing File Type")
        pricing_type_selection = st.radio(
            "The file contains pricing for:",
            ("FTL Only", "LTL Only", "Both FTL and LTL", "FTL Pricing Predictor(To be Added)"),
            key=f"{session_key}_pricing_type_radio"
            )
        st.markdown("---") # Separator for cleaner UI
        
        st.subheader("🎯 Column Mapping & Defaults")
        with st.container(border=True):
                available_cols = ["--- Not Applicable ---"] + list(st.session_state[session_key]['df_for_mapping'].columns)
                mappings, cols = {}, st.columns(2)
                
                # Dynamic mapping target keys based on selection
                current_mapping_target_keys = pc_config["llm_mapping_target_keys"].copy()
                if pricing_type_selection == "LTL Only":
                    # Remove 'Truck Type' mapping for LTL only
                    if 'Truck Type' in current_mapping_target_keys:
                        del current_mapping_target_keys['Truck Type']
                
                for i, (disp_name, key_suffix) in enumerate(current_mapping_target_keys.items()):
                    with cols[i % 2]:
                        llm_suggestion = st.session_state[session_key].get('llm_suggestions', {}).get(key_suffix)
                        default_index = available_cols.index(llm_suggestion) if llm_suggestion in available_cols else 0
                        selected = st.selectbox(f"Map '{disp_name.replace('_',' ').title()}' to:", available_cols, index=default_index, key=f"{session_key}_map_{key_suffix}")
                        mappings[key_suffix] = selected if selected != "--- Not Applicable ---" else None
                
                # Explicitly set Truck Type mapping to None for "LTL Only"
                if pricing_type_selection == "LTL Only":
                    mappings[pc_config["llm_mapping_target_keys"]['Truck Type']] = None # Ensure it's null for LTL

                st.session_state[session_key]['column_mappings'] = mappings
        with st.container(border=True):
            defaults, cols = {}, st.columns(4)
            with cols[0]: defaults['origin_city'] = st.text_input("Default Origin City:", key=f"{session_key}_default_city")
            with cols[1]: defaults['origin_state'] = st.text_input("Default Origin State:", key=f"{session_key}_default_state")
            with cols[2]: defaults['origin_pincode'] = st.text_input("Default Origin Pincode:", key=f"{session_key}_default_origin_pincode")
            with cols[3]: defaults['destination_pincode'] = st.text_input("Default Destination Pincode:", key=f"{session_key}_default_dest_pincode")
            st.session_state[session_key]['defaults'] = defaults
        
        if st.button("✅ Confirm Mapping & Process Data", key=f"{session_key}_process"):
            with st.spinner("⚙️ Standardizing pricing data..."):
                df_input = st.session_state[session_key]['df_for_mapping'].copy()
                state_cols = [mappings.get(pc_config["llm_mapping_target_keys"]['origin_state']), mappings.get(pc_config["llm_mapping_target_keys"]['destination_state'])]
                all_states_series = [df_input[col].dropna() for col in state_cols if col and col in df_input.columns]
                unique_states_list = pd.unique(pd.concat(all_states_series).astype(str)).tolist() if all_states_series else []
                llm_state_map = pc_get_llm_suggestions(llm_client, deployment_name, pc_get_llm_state_mapping, unique_states_list, pc_config["standard_states"]) if unique_states_list and llm_client else {}
                unique_trucks = []
                if st.session_state[session_key]['data_was_transformed']:
                    if 'Original Truck Header' in df_input.columns: unique_trucks = df_input['Original Truck Header'].dropna().astype(str).unique().tolist()
                else:
                    truck_col = mappings.get(pc_config["llm_mapping_target_keys"]['Truck Type'])
                    if truck_col and truck_col in df_input.columns: unique_trucks = df_input[truck_col].dropna().astype(str).unique().tolist()
                llm_truck_type_map = pc_get_llm_suggestions(llm_client, deployment_name, pc_get_llm_batch_truck_type_mapping, unique_trucks, pc_config["truck_types"]) if unique_trucks and llm_client else {}
                
                # Adjust defaults and mappings based on pricing type selection -- new pricing
                if pricing_type_selection == "LTL Only":
                    # For LTL Only, hardcode Service Type, Rate Type, Transit Days
                    defaults['Service Type'] = 'LTL'
                    defaults['Rate Type'] = 'Per Km'
                    defaults['Transit Days'] = 5
                    llm_truck_type_map = {} # No truck type mapping needed for LTL
                elif pricing_type_selection == "Both FTL and LTL":
                    # Ensure 'Service Type' is mapped and not null in the input file
                    service_type_col_key = "Service Type_Source" # Assuming this is the key used in target_keys_dict if mapped
                    service_type_mapped_col = mappings.get(service_type_col_key) # Get the actual input column name

                    if service_type_mapped_col is None:
                        st.error("☠️ For 'Both FTL and LTL', 'Service Type' column must be mapped from your input file.")
                        return # Stop processing
                    
                    # Ensure Service Type column is not entirely null in the raw data
                    if st.session_state[session_key]['df_for_mapping'][service_type_mapped_col].isnull().all():
                        st.error("☠️ For 'Both FTL and LTL', the mapped 'Service Type' column cannot be entirely empty.")
                        return # Stop processing


                processed_df = pc_process_data_rows(df_input, mappings, defaults, st.session_state.pc_ref_id_df, llm_state_map, llm_truck_type_map, st.session_state.config, st.session_state[session_key]['data_was_transformed'])
                if pricing_type_selection == "FTL Only":
                    final_df = pc_finalize_dataframe(processed_df, st.session_state.config)
                else: # "LTL Only" or "Both FTL and LTL"
                    final_df = processed_df
                st.session_state[session_key]['final_df'] = final_df

                

    if 'final_df' in st.session_state.get(session_key, {}):
        final_df = st.session_state[session_key]['final_df']
        st.markdown("---"); st.subheader("🏁 Final Processed Data")
        st.dataframe(final_df.head(10))
        st.session_state.processed_pricing_df = final_df
        st.success(f"✅ Pricing data processed and stored in session. {len(final_df)} rows are ready.")
        csv_data = final_df.to_csv(index=False).encode('utf-8')
        st.download_button(label="📥 Download Standardized CSV", data=csv_data, file_name=f"{os.path.splitext(uploaded_file.name)[0]}_standardized.csv", mime='text/csv', type="primary")

def render_product_cleaner_page(llm_client, llm_available, llm_deployment_name):
    """Renders the Product Information Cleaner page."""
    st.title("TMS Product Information Cleaner")
    st.write("➡️ Upload your product file, map columns, and let the GenIE standardize your product data!")
    st.markdown("---")

    prod_config = st.session_state.config['product_cleaner']

    # File upload
    uploaded_file = st.file_uploader(
        "📂 Choose a Product file (CSV, XLSX, XLS)", 
        type=['csv', 'xlsx', 'xls'], 
        key="product_uploader"
    )

    if not uploaded_file:
        st.info("Please upload a file to begin.")
        return

    # Create session key for this file
    session_key = f"prod_{uploaded_file.name}"
    if session_key not in st.session_state:
        st.session_state[session_key] = {}

    # File loading logic
    if 'raw_df' not in st.session_state.get(session_key, {}):
        df_raw = None
        
        # Handle different file types
        if 'sheet_names' not in st.session_state.get(session_key, {}):
            file_name = uploaded_file.name
            if file_name.endswith('.csv'):
                df_raw = load_csv_from_upload(uploaded_file)
                if df_raw is not None:
                    st.session_state[session_key]['raw_df'] = df_raw
                    st.rerun()
            elif file_name.endswith(('.xlsx', '.xls')):
                try:
                    uploaded_file.seek(0)
                    xls = pd.ExcelFile(uploaded_file)
                    st.session_state[session_key]['sheet_names'] = xls.sheet_names
                    st.rerun()
                except Exception as e:
                    st.error(f"Error reading Excel file: {e}")
                    return
        
        # Sheet selection for Excel files
        if 'sheet_names' in st.session_state.get(session_key, {}):
            st.subheader("Select Sheet")
            st.info("Your Excel file contains multiple sheets. Please select the one to process.")
            selected_sheet = st.selectbox(
                "Which sheet contains the product data?",
                st.session_state[session_key]['sheet_names'],
                key=f"{session_key}_sheet_select"
            )
            if st.button("Load Selected Sheet", key=f"{session_key}_load_sheet"):
                try:
                    uploaded_file.seek(0)
                    df_raw = pd.read_excel(uploaded_file, sheet_name=selected_sheet)
                    st.session_state[session_key]['raw_df'] = df_raw
                    del st.session_state[session_key]['sheet_names']
                    st.rerun()
                except Exception as e:
                    st.error(f"Error loading sheet '{selected_sheet}': {e}")
            return

    # Main processing logic
    if 'raw_df' in st.session_state.get(session_key, {}):
        st.subheader("📜 Original Data Preview")
        raw_df = st.session_state[session_key]['raw_df']
        st.dataframe(raw_df.head())

        if 'final_df' not in st.session_state.get(session_key, {}):
            st.subheader("🎯 Column Mapping Review")
            
            # Get LLM suggestions if available
            if 'mapping_suggestions' not in st.session_state[session_key]:
                if llm_client and llm_available:
                    with st.spinner("🤖 Getting AI column mapping suggestions..."):
                        st.session_state[session_key]['mapping_suggestions'] = prod_get_llm_column_mapping_suggestions(
                            llm_client, llm_deployment_name, 
                            raw_df.columns.tolist(), 
                            prod_config["standard_output_columns"]
                        )
                else:
                    st.session_state[session_key]['mapping_suggestions'] = {}

            mapping_suggestions = st.session_state[session_key]['mapping_suggestions']
            input_options = ["-- Select or Leave Blank --"] + raw_df.columns.tolist()
            
            # Create mapping interface
            st.info("Map your columns to the standard product schema. Required fields are marked with ❗️")
            ui_columns = st.columns(3)
            
            for i, standard_col in enumerate(prod_config["standard_output_columns"]):
                with ui_columns[i % 3]:
                    is_required = standard_col in prod_config["required_columns"]
                    label = f"❗️'{standard_col}' (Required)" if is_required else f"'{standard_col}'"
                    
                    # Get LLM suggestion
                    llm_suggestion = mapping_suggestions.get(standard_col)
                    default_index = input_options.index(llm_suggestion) if llm_suggestion in input_options else 0
                    
                    if llm_suggestion:
                        st.caption(f"AI Suggests: **{llm_suggestion}**")
                    
                    st.selectbox(
                        label, 
                        input_options, 
                        index=default_index, 
                        key=f"{session_key}_map_{standard_col}"
                    )

            # Special handling for product_orientation
            st.subheader("📐 Product Orientation Default")
            st.selectbox(
                "Default orientation for products (if not mapped above):",
                prod_config["orientation_options"],
                index=0,
                key=f"{session_key}_default_orientation",
                help="'all' means product can be placed in any orientation, 'upright' means it must remain upright"
            )

            # Volume calculation option
            st.subheader("📦 Volume Calculation")
            calculate_volume = st.checkbox(
                "Calculate volume from dimensions (length × breadth × height) if volume is not provided",
                value=True,
                key=f"{session_key}_calc_volume"
            )

            # Process button
            if st.button("✅ Confirm Mapping & Process Data", key=f"{session_key}_process"):
                # Collect mappings
                confirmed_mapping = {}
                missing_required = []
                
                for col in prod_config["standard_output_columns"]:
                    selection = st.session_state[f"{session_key}_map_{col}"]
                    if selection != "-- Select or Leave Blank --":
                        confirmed_mapping[col] = selection
                    elif col in prod_config["required_columns"]:
                        missing_required.append(col)

                if missing_required:
                    st.error(f"❌ Please map all required columns: {', '.join(missing_required)}")
                else:
                    with st.spinner("Processing product data..."):
                        # Create standardized dataframe
                        processed_df = prod_standardize_dataframe(
                            raw_df, 
                            confirmed_mapping, 
                            st.session_state[f"{session_key}_default_orientation"],
                            calculate_volume,
                            prod_config
                        )
                        
                        # Check if validation failed (empty dataframe returned)
                        if processed_df.empty:
                            st.error("❌ Processing failed due to validation errors. Please fix the issues above and try again.")
                        else:
                            st.session_state[session_key]['final_df'] = processed_df
                            st.rerun()

    # Display final results
    if 'final_df' in st.session_state.get(session_key, {}):
        final_df = st.session_state[session_key]['final_df']
        st.markdown("---")
        st.subheader("🏁 Final Processed Product Data")
        st.dataframe(final_df.head(10))
        
        # Store in session for TMS workflow
        st.session_state.processed_product_df = final_df
        st.success(f"✅ Product data processed and stored in session. {len(final_df)} rows are ready.")
        
        # Check for null values and display warning
        st.subheader("📊 Data Quality Check")
        null_counts = final_df.isnull().sum()
        columns_with_nulls = null_counts[null_counts > 0]
        
        # Debug: Show what we're checking
        st.write("**Debug Info:**")
        st.write(f"Total rows: {len(final_df)}")
        st.write("---")
        
        if len(columns_with_nulls) > 0:
            st.warning("⚠️ **Missing Data Detected** - Please review before final upload:")
            
            # Create a formatted display of null counts
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write("**Columns with missing values:**")
                for column, null_count in columns_with_nulls.items():
                    percentage = (null_count / len(final_df)) * 100
                    st.write(f"• **{column}**: {null_count} missing ({percentage:.1f}%)")
            
            with col2:
                # Show total statistics
                total_nulls = columns_with_nulls.sum()
                total_cells = len(final_df) * len(final_df.columns)
                overall_percentage = (total_nulls / total_cells) * 100
                
                st.metric(
                    label="Total Missing Values", 
                    value=f"{total_nulls:,}",
                    delta=f"{overall_percentage:.1f}% of all data"
                )
            
            # Show which columns are critical
            prod_config = st.session_state.config['product_cleaner']
            critical_nulls = [col for col in columns_with_nulls.index if col in prod_config["required_columns"]]
            
            if critical_nulls:
                st.error(f"🚨 **Critical fields with missing data**: {', '.join(critical_nulls)}")
                st.write("These are required fields and should be completed before using in TMS planning.")
            
            # Expandable detailed view
            with st.expander("🔍 View detailed null analysis"):
                st.write("**Complete null count summary:**")
                null_summary_df = pd.DataFrame({
                    'Column': final_df.columns,
                    'Null Count': [final_df[col].isnull().sum() for col in final_df.columns],
                    'Non-Null Count': [final_df[col].notnull().sum() for col in final_df.columns],
                    'Null Percentage': [f"{(final_df[col].isnull().sum() / len(final_df)) * 100:.1f}%" for col in final_df.columns]
                })
                st.dataframe(null_summary_df, use_container_width=True)
        else:
            st.success("✅ **No missing data detected** - All columns are complete!")
        
        # Download button
        csv_data = final_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Standardized Product CSV", 
            data=csv_data, 
            file_name=f"{os.path.splitext(uploaded_file.name)[0]}_standardized_products.csv", 
            mime='text/csv', 
            type="primary"
        )

def prod_standardize_dataframe(raw_df, column_mapping, default_orientation, calculate_volume, config):
    """Standardizes the product dataframe based on column mappings."""
    # Create output dataframe with only mapped columns (to preserve original null status)
    output_df = pd.DataFrame(index=raw_df.index)
    
    # Map columns
    for standard_col, user_col in column_mapping.items():
        if user_col in raw_df.columns:
            output_df[standard_col] = raw_df[user_col]
    
    # Add any unmapped standard columns as empty (this will show as nulls in quality check)
    for col in config["standard_output_columns"]:
        if col not in output_df.columns:
            output_df[col] = None
    
    # Handle product_orientation default
    if 'product_orientation' in output_df.columns:
        output_df['product_orientation'] = output_df['product_orientation'].fillna(default_orientation)
    
    # Calculate volume if requested and dimensions are available
    if calculate_volume and 'product_volume' in output_df.columns:
        has_dimensions = all(col in output_df.columns for col in ['product_length', 'product_breadth', 'product_height'])
        if has_dimensions:
            # Calculate volume where it's missing
            mask = output_df['product_volume'].isna()
            if mask.any():
                calculated_volume = (
                    pd.to_numeric(output_df['product_length'], errors='coerce') *
                    pd.to_numeric(output_df['product_breadth'], errors='coerce') *
                    pd.to_numeric(output_df['product_height'], errors='coerce')
                )
                output_df.loc[mask, 'product_volume'] = calculated_volume[mask]
                st.info(f"📦 Calculated volume for {mask.sum()} products using dimensions")
    
    # Clean up numeric columns
    numeric_cols = ['product_weight', 'product_volume', 'max_stackable_quantities', 
                   'max_stackable_weight', 'product_length', 'product_breadth', 'product_height']
    
    for col in numeric_cols:
        if col in output_df.columns:
            output_df[col] = pd.to_numeric(output_df[col], errors='coerce')
    
    # Validate for duplicate product_code + product_name pairs with different dimensions/weights
    if 'product_code' in output_df.columns and 'product_name' in output_df.columns:
        # Remove rows where both product_code and product_name are null
        valid_products = output_df.dropna(subset=['product_code', 'product_name'], how='all')
        
        if not valid_products.empty:
            # Group by product_code and product_name
            grouped = valid_products.groupby(['product_code', 'product_name'])
            
            # Check for inconsistent dimensions/weights within each group
            dimension_cols = ['product_length', 'product_breadth', 'product_height', 'product_weight', 'product_volume']
            available_dim_cols = [col for col in dimension_cols if col in valid_products.columns]
            
            duplicate_issues = []
            
            for (prod_code, prod_name), group in grouped:
                if len(group) > 1:  # Multiple rows for same product_code + product_name
                    # Check if dimensions/weights are different
                    for col in available_dim_cols:
                        unique_values = group[col].dropna().unique()
                        if len(unique_values) > 1:  # Different values found
                            duplicate_issues.append({
                                'product_code': prod_code,
                                'product_name': prod_name,
                                'conflicting_column': col,
                                'values': unique_values.tolist()
                            })
                            break  # One conflict is enough to flag this product
            
            if duplicate_issues:
                # Create error message
                error_msg = "❌ **Multiple dimensions/weights found for the same product:**\n\n"
                for issue in duplicate_issues[:5]:  # Show first 5 issues
                    error_msg += f"• **{issue['product_code']}** - **{issue['product_name']}**\n"
                    error_msg += f"  Conflicting {issue['conflicting_column']}: {issue['values']}\n\n"
                
                if len(duplicate_issues) > 5:
                    error_msg += f"... and {len(duplicate_issues) - 5} more conflicts.\n\n"
                
                error_msg += "**Please ensure each product has consistent dimensions and weights across all rows.**"
                st.error(error_msg)
                return pd.DataFrame()  # Return empty dataframe to prevent processing
    
    # Remove duplicate rows (keep first occurrence)
    initial_count = len(output_df)
    output_df = output_df.drop_duplicates()
    final_count = len(output_df)
    
    if initial_count > final_count:
        st.info(f"🔄 Removed {initial_count - final_count} duplicate rows. Final dataset has {final_count} unique products.")
    
    return output_df

def render_tms_planner_page():
    st.title("TMS Planning & API Trigger")
    
    api_config = st.session_state.config['api']
    if not api_config['tms_url']:
        st.error("🚨 **CRITICAL:** `TMS_API_URL` is not configured. The application cannot proceed.")
        return

    st.markdown("---")
    st.subheader("Step 3.1: Choose Input Data Source")

    use_processed_data_ready = st.session_state.processed_orders_df is not None and st.session_state.processed_pricing_df is not None
    
    source_options = ["Use Processed Data from Steps 1, 2 & 3",
                      "Upload separate CSV files",
                      "Upload a single pre-formatted XLSX file",
                      "Use a Signed URL"]
    
    if not use_processed_data_ready:
        st.warning("Data from Steps 1 & 2 not available. Please complete them first or choose another input method below.")

    tms_input_source = st.radio(
        "Choose the source for the TMS input:",
        source_options,
        key='tms_source_radio',
        index=0 if use_processed_data_ready else 1
    )

    # --- Prepare data based on selected source ---
    orders_df, rates_df, vehicles_df = None, None, None
    tms_input_excel_buffer = None
    signed_url_input = None
    # Add variables to hold raw dataframes if coming from uploaders
    raw_orders_df_for_validation = None
    raw_rates_df_for_validation = None

    if tms_input_source == "Use Processed Data from Steps 1, 2 & 3":
        if use_processed_data_ready:
            if st.session_state.processed_product_df is not None:
                st.success("✅ Using cleaned Order, Pricing, and Product data from the previous steps.")
            else:
                st.success("✅ Using cleaned Order and Pricing data from the previous steps.")
            orders_df = tms_process_orders_data(st.session_state.processed_orders_df)
            rates_df = tms_process_rate_master_data(st.session_state.processed_pricing_df)
            vehicles_df = load_csv_from_path(st.session_state.config['paths']['vehicles_types'])
            st.session_state.initial_orders_df_for_reports = st.session_state.processed_orders_df.copy()
            # Set raw dataframes for validation if applicable
            raw_orders_df_for_validation = orders_df # These are already "processed" but good for validation
            raw_rates_df_for_validation = rates_df
        else:
            st.error("Data from previous steps is not ready. Please complete Steps 1 & 2.")
            return

    elif tms_input_source == "Upload separate CSV files":
        st.info("Upload the required CSV files.")
        c1, c2, c3, c4 = st.columns(4)
        uploaded_orders = c1.file_uploader("Upload Orders CSV", type="csv", key="tms_orders_csv")
        uploaded_rates = c2.file_uploader("Upload Pricing CSV", type="csv", key="tms_rates_csv")
        uploaded_product_info = c3.file_uploader("Upload Product Info CSV (Optional)", type="csv", key="tms_product_csv")
        
        # Add checkbox for standard vehicles file
        use_standard_vehicles = c4.checkbox("Use standard vehicles file", key="tms_standard_vehicles_check")
        
        uploaded_vehicles = None
        vehicles_df_raw = None
        product_df_raw = None

        if use_standard_vehicles:
            vehicles_df_raw = load_csv_from_path(st.session_state.config['paths']['vehicles_types'])
            if not vehicles_df_raw.empty:
                c4.success("Standard vehicles file loaded.")
        else:
            uploaded_vehicles = c4.file_uploader("Upload Vehicles CSV", type="csv", key="tms_vehicles_csv", label_visibility="collapsed")
            if uploaded_vehicles:
                vehicles_df_raw = load_csv_from_upload(uploaded_vehicles)
        
        # Load product info if uploaded
        if uploaded_product_info:
            product_df_raw = load_csv_from_upload(uploaded_product_info)

        # Check if all data is ready
        if uploaded_orders and uploaded_rates and (vehicles_df_raw is not None and not vehicles_df_raw.empty):
            orders_df_raw = load_csv_from_upload(uploaded_orders)
            rates_df_raw = load_csv_from_upload(uploaded_rates)
            
            orders_df = tms_process_orders_data(orders_df_raw)
            rates_df = tms_process_rate_master_data(rates_df_raw)
            vehicles_df = vehicles_df_raw
            st.session_state.initial_orders_df_for_reports = orders_df_raw.copy()
            # Set raw dataframes for validation
            raw_orders_df_for_validation = orders_df
            raw_rates_df_for_validation = rates_df


    elif tms_input_source == "Upload a single pre-formatted XLSX file":
        st.info("Upload a single XLSX file with 'orders', 'rate_master', 'vehicles_types', and 'plan_setting' sheets.")
        uploaded_xlsx = st.file_uploader("Upload your custom XLSX file", type="xlsx", key="tms_xlsx")
        if uploaded_xlsx:
            if tms_validate_uploaded_xlsx(uploaded_xlsx):
                st.success("✅ XLSX file validated successfully.")
                tms_input_excel_buffer = BytesIO(uploaded_xlsx.getvalue())
                uploaded_xlsx.seek(0) # Reset pointer after reading for validation
                
                # Load orders and rates for validation from the XLSX
                raw_orders_df_for_validation = pd.read_excel(uploaded_xlsx, sheet_name='orders')
                uploaded_xlsx.seek(0) # Reset pointer again
                raw_rates_df_for_validation = pd.read_excel(uploaded_xlsx, sheet_name='rate_master')
                
                st.session_state.initial_orders_df_for_reports = raw_orders_df_for_validation.copy() # Save the full orders sheet
            else:
                st.error("The uploaded XLSX file is invalid. Please check the required sheets and columns.")
                return

    elif tms_input_source == "Use a Signed URL":
        signed_url_input = st.text_input("Paste the Signed URL for the XLSX file", placeholder="https://your-cloud-storage-url/...")
        st.session_state.direct_trigger_source_url = signed_url_input
        # Reset initial orders df if using URL, it will be fetched later if needed
        if not st.session_state.tms_triggered:
            st.session_state.initial_orders_df_for_reports = None
    
    # --- This block will only execute if we need to generate the excel file from DataFrames ---
    if orders_df is not None and rates_df is not None and vehicles_df is not None:
        st.markdown("---")
        st.subheader("Step 3.2: Configure Plan Settings")
        st.warning("Product details must be shared to enable stacking.", icon="⚠️")
        settings_cols = st.columns(3)
        user_settings = {
            'Min Weight Utilisation': settings_cols[0].number_input("Min Weight Utilisation (%)", 0, 100, 30, 1),
            'Min Volume Utilisation': settings_cols[1].number_input("Min Volume Utilisation (%)", 0, 100, 30, 1),
            'Max Detour': settings_cols[2].number_input("Max Detour (km)", 0, 500, 100, 10),
            'Customer Split': settings_cols[0].checkbox("Allow Customer Split", False),
            'Order Split': settings_cols[1].checkbox("Allow Order Split", False),
            'Plan LTL Loads': settings_cols[0].checkbox("Plan LTL Loads", True),
            'Consolidate LTL': settings_cols[1].checkbox("Consolidate LTL", False),
            'Stacking': settings_cols[2].checkbox("Stacking", False),
            'Max Stops': settings_cols[2].number_input("Max Stops per Vehicle", 0, 20, 5, 1)
        }
        plan_setting_df = pd.DataFrame([user_settings])
        st.write("**Plan Settings:**"); st.dataframe(plan_setting_df)
        
        # tms_input_excel_buffer is generated here if not from XLSX upload
        # Use appropriate product data source based on input method
        product_data_to_use = None
        if tms_input_source == "Use Processed Data from Steps 1, 2 & 3":
            product_data_to_use = st.session_state.processed_product_df
        elif tms_input_source == "Upload separate CSV files" and 'product_df_raw' in locals():
            product_data_to_use = product_df_raw
        
        tms_input_excel_buffer = tms_generate_tms_input_excel(orders_df, rates_df, vehicles_df, plan_setting_df, product_data_to_use)
        st.download_button("📥 Download Generated XLSX for Review", tms_input_excel_buffer, "generated_tms_input.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    st.markdown("---")
    st.subheader("Step 3.3: Trigger TMS Run")
    if st.button("🚀 Trigger TMS Planning Run", type="primary"):
        final_signed_url = None
        
        if tms_input_source == "Use a Signed URL":
            if signed_url_input:
                final_signed_url = signed_url_input
            else:
                st.error("Please enter a signed URL.")
                # IMPORTANT: Return early if validation fails to prevent further processing
                return 
        else: # For all other sources (Processed Data, Separate CSVs, Single XLSX)
            # Perform validations on the raw_orders_df_for_validation and raw_rates_df_for_validation
            if raw_orders_df_for_validation is None or raw_rates_df_for_validation is None:
                st.error("Internal error: DataFrames for validation are not available. Please re-select your input files.")
                return # Stop processing
            
            # --- NEW VALIDATION STEP ---
            with st.spinner("Performing data validations..."):
                if not tms_perform_data_validations(raw_orders_df_for_validation, raw_rates_df_for_validation, st.session_state.config):
                    st.warning("Data validation failed. Please correct the issues and try again.")
                    return # Stop processing if validations fail
            st.success("✅ Data validations passed!")
            # --- END NEW VALIDATION STEP ---

            if tms_input_excel_buffer is not None:
                with st.spinner("Uploading file to S3 and getting secure URL..."):
                    final_signed_url = tms_upload_to_s3_and_get_signed_url(tms_input_excel_buffer, api_config['s3_bucket'], api_config['s3_region'])
                if final_signed_url:
                    st.success("✅ Secure URL generated successfully.")
                else:
                    st.error("❌ Failed to generate secure URL. Cannot trigger TMS.")
                    return # Stop processing
            else:
                st.warning("Please provide a valid data source before triggering.")
                return # Stop processing

        if final_signed_url:
            with st.spinner("Triggering TMS planning... This may take over a minute."):
                try:
                    trigger_response = tms_trigger_tms_plan(api_config['tms_url'], final_signed_url)
                    st.session_state.req_id = trigger_response.get('data', {}).get('req_id')
                    st.success(f"✅ TMS Triggered Successfully! Request ID: `{st.session_state.req_id}`")
                    st.session_state.tms_triggered = True
                    time.sleep(45) # Initial wait
                    status_response = tms_get_tms_status(api_config['tms_url'], st.session_state.req_id)
                    sub_ids = status_response.get('details', {}).get('sub_ids', [])
                    if sub_ids:
                        sub_ids_df = pd.DataFrame(sub_ids, columns=["sub_id"])
                        sub_ids_df['date'] = sub_ids_df['sub_id'].apply(tms_extract_date_from_sub_id)
                        st.session_state.sub_ids_df = sub_ids_df
                    (st.session_state.plan_summaries, st.session_state.route_summaries, st.session_state.route_sequences, st.session_state.failed_sub_ids) = tms_fetch_all_tms_results(api_config['tms_url'], st.session_state.req_id, st.session_state.sub_ids_df)
                    st.rerun() 
                except Exception as e: 
                    st.error(f"An error occurred during TMS trigger or result fetching: {e}")
        # else: # This else is now covered by the `return` statements above if final_signed_url is not set
        #     st.warning("Cannot proceed with TMS trigger without a valid signed URL.")


    if st.session_state.tms_triggered:
        with st.container(border=True):
            st.subheader("Step 3.4: Review Planning Results")
            st.info(f"Displaying results for Request ID: `{st.session_state.req_id}`")
            if st.button("🔄 Refresh Results"):
                with st.spinner("Fetching latest status and results..."):
                    try:
                        status_response = tms_get_tms_status(api_config['tms_url'], st.session_state.req_id)
                        sub_ids = status_response.get('details', {}).get('sub_ids', [])
                        if sub_ids:
                            sub_ids_df = pd.DataFrame(sub_ids, columns=["sub_id"])
                            sub_ids_df['date'] = sub_ids_df['sub_id'].apply(tms_extract_date_from_sub_id)
                            st.session_state.sub_ids_df = sub_ids_df
                            st.info(f"Found {len(sub_ids_df)} sub-plans.")
                        else:
                            st.warning("No new sub-plan IDs found in the latest status check.")
                        (st.session_state.plan_summaries, st.session_state.route_summaries, 
                         st.session_state.route_sequences, st.session_state.failed_sub_ids) = tms_fetch_all_tms_results(
                            api_config['tms_url'], st.session_state.req_id, st.session_state.sub_ids_df
                        )
                    except requests.RequestException as e: st.error(f"API Request Failed during refresh: {e}.")
                    except (KeyError, json.JSONDecodeError) as e: st.error(f"Failed to parse API response during refresh: {e}")
                    except Exception as e: st.error(f"An unexpected error occurred during refresh: {e}")
                
            if (st.session_state.initial_orders_df_for_reports is None or st.session_state.initial_orders_df_for_reports.empty) and st.session_state.direct_trigger_source_url:
                with st.spinner("Downloading original input file to generate reports..."):
                    try:
                        response = requests.get(st.session_state.direct_trigger_source_url)
                        response.raise_for_status()
                        file_content = BytesIO(response.content)
                        st.session_state.initial_orders_df_for_reports = pd.read_excel(file_content, sheet_name='orders')
                        st.rerun() 
                    except Exception as e:
                        st.warning(f"Could not download or read the original input file from the provided URL. Detailed reports cannot be generated. Error: {e}")

            st.metric("Plan Summaries Fetched", len(st.session_state.plan_summaries))
            st.metric("Route Summaries Fetched", len(st.session_state.route_summaries))
            st.metric("Route Sequences Fetched", len(st.session_state.route_sequences))
            if st.session_state.failed_sub_ids:
                st.metric("Failed Sub-Plan IDs", len(st.session_state.failed_sub_ids), delta_color="inverse")

            if st.session_state.plan_summaries:
                if st.session_state.initial_orders_df_for_reports is not None and not st.session_state.initial_orders_df_for_reports.empty:
                    summary_df = tms_generate_client_dod_summary(st.session_state.plan_summaries, st.session_state.route_sequences, st.session_state.initial_orders_df_for_reports)
                    
                    tab1, tab2, tab3, tab4 = st.tabs(["📊 Client DOD Summary", "📄 Plan Output Files", "⚠️ Failed IDs", "Raw Data"])
                    
                    with tab1:
                        st.markdown("#### Client Day-over-Day Summary")
                        st.dataframe(summary_df)
                        st.download_button("📥 Download DOD Summary (CSV)", summary_df.to_csv(index=False).encode('utf-8'), "client_dod_summary.csv", "text/csv")
                        st.markdown("---")
                        st.markdown("#### CBA Analysis - FTL")

                        if not summary_df.empty:
                            cba_cols_map = {
                                'date': 'date', 'total_shipments': 'total planned shipments', 'total_weight': 'total planned weight',
                                'total_volume': 'total planned volume', 'total_vehicles_used': 'total-vehicles', 'total_cost': 'total_cost'
                            }
                            cols_to_show = {k: v for k, v in cba_cols_map.items() if k in summary_df.columns}
                            if len(cols_to_show) == len(cba_cols_map):
                                cba_analysis_df = summary_df[list(cols_to_show.keys())].copy().rename(columns=cols_to_show)
                                cba_analysis_df = cba_analysis_df.sort_values(by='date', ascending=True)
                                st.dataframe(cba_analysis_df)
                            else:
                                st.warning("Could not generate CBA Analysis view as one or more required columns are missing.")
                        else:
                            st.info("Summary data is not available to generate CBA Analysis.")

                    with tab2:
                        st.markdown("#### Detailed Plan Output Files")
                        plan_output_excel = tms_generate_plan_output_excel(st.session_state.plan_summaries, st.session_state.route_summaries, st.session_state.route_sequences, st.session_state.initial_orders_df_for_reports)
                        if plan_output_excel:
                            st.download_button("📥 Download Plan Output (XLSX)", plan_output_excel, "generated_plan_output.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                    
                    with tab3:
                        if st.session_state.failed_sub_ids:
                            st.error("The following sub-plan IDs failed or returned null payloads:")
                            st.dataframe(pd.DataFrame(st.session_state.failed_sub_ids, columns=["Failed Sub-Plan ID"]))
                        else:
                            st.success("No failed sub-plan IDs were found.")

                    with tab4:
                        with st.expander("Raw Plan Summaries"): st.dataframe(pd.DataFrame(st.session_state.plan_summaries))
                        with st.expander("Raw Route Summaries"): st.dataframe(pd.DataFrame(st.session_state.route_summaries))
                        with st.expander("Raw Route Sequences"): st.dataframe(pd.DataFrame(st.session_state.route_sequences))
                
                else:
                    st.warning("Initial order data is not available. Detailed reports cannot be generated.")
                    tab1, tab2 = st.tabs(["⚠️ Failed IDs", "Raw Data"])
                    with tab1:
                        if st.session_state.failed_sub_ids:
                            st.error("The following sub-plan IDs failed or returned null payloads:")
                            st.dataframe(pd.DataFrame(st.session_state.failed_sub_ids, columns=["Failed Sub-Plan ID"]))
                        else:
                            st.success("No failed sub-plan IDs were found.")
                    with tab2:
                        with st.expander("Raw Plan Summaries"): st.dataframe(pd.DataFrame(st.session_state.plan_summaries))
                        with st.expander("Raw Route Summaries"): st.dataframe(pd.DataFrame(st.session_state.route_summaries))
                        with st.expander("Raw Route Sequences"): st.dataframe(pd.DataFrame(st.session_state.route_sequences))
            else:
                st.info("No results have been fetched yet. Click 'Refresh Results' or trigger a new run.")

# ==============================================================================
# 7. MAIN APPLICATION
# ==============================================================================
def main():
    """Main function to configure and run the Streamlit application."""
    st.set_page_config(layout="wide", page_title="Delhivery Data Genie")

    # --- Initial setup calls ---
    # These should be called once at the start of the main function.
    load_env_vars()
    initialize_session_state()
    llm_client, llm_available, llm_deployment_name = initialize_llm_client()
    config = st.session_state.config

    # --- Sidebar UI and Navigation Logic ---
    # All UI elements, including the sidebar, must be inside the main function.
    with st.sidebar:
        st.markdown(
            """
            <div style="padding: 10px; text-align: center;">
                <h1 style="font-weight:bold; margin-bottom: 0;">
                    <span style="color:white;">DELHIVERY DATA</span><span style="color:red;">GENIE</span> 🧞‍♂️
                </h1>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.sidebar.markdown("---")

        # SCS Workflow Section
        st.header("SCS WORKFLOW")
        if st.button("SCS - Standardiser", use_container_width=True):
            st.session_state.page = "SCS - Standardiser"
            st.rerun()

        # TMS Workflow Section
        st.header("TMS WORKFLOW")
        if st.button("TMS Home", use_container_width=True):
            st.session_state.page = "TMS: Home"
            st.rerun()
        if st.button("Process Order File", use_container_width=True):
            st.session_state.page = "TMS: Process Order File"
            st.rerun()
        if st.button("Process Pricing File", use_container_width=True):
            st.session_state.page = "TMS: Process Pricing File"
            st.rerun()
        if st.button("Process Product Information", use_container_width=True):
            st.session_state.page = "TMS: Process Product Information"
            st.rerun()
        if st.button("Run TMS Planner", use_container_width=True):
            st.session_state.page = "TMS: Run TMS Planner"
            st.rerun()

    # --- Page Routing ---
    # This logic determines which page to display based on the session state.
    page = st.session_state.get("page", "SCS - Standardiser")

    if page == "SCS - Standardiser":
        render_scs_standardiser_page(llm_client, llm_deployment_name, config)
    elif page == "TMS: Home":
        render_tms_home_page()
    elif page == "TMS: Process Order File":
        render_order_cleaner_page(llm_client, llm_available, llm_deployment_name)
    elif page == "TMS: Process Pricing File":
        render_pricing_cleaner_page(llm_client, llm_available, llm_deployment_name)
    elif page == "TMS: Process Product Information":
        render_product_cleaner_page(llm_client, llm_available, llm_deployment_name)
    elif page == "TMS: Run TMS Planner":
        render_tms_planner_page()


if __name__ == "__main__":
    main()


