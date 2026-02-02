import google.generativeai as genai
import pandas as pd
import datetime
import chainlit as cl
import prestodb
import asyncio
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

# from vanna.chromadb import ChromaDB_VectorStore
# from vanna.google import GoogleGeminiChat

from vanna.openai import OpenAI_Chat
import openai
from vanna.chromadb import ChromaDB_VectorStore

from dotenv import load_dotenv, find_dotenv
import os

env_path = find_dotenv()
print("Loading .env from :", env_path)
load_dotenv(env_path)

# GOOGLE_API_KEY= os.getenv("GOOGLE_API_KEY")

AZURE_API_KEY = os.getenv("AZURE_API_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME")
API_VERSION = os.getenv("API_VERSION")

print("AZURE_API_KEY:", AZURE_API_KEY)
print("AZURE_ENDPOINT:", AZURE_ENDPOINT)
print("API_VERSION:", API_VERSION)
print("DEPLOYMENT_NAME:", DEPLOYMENT_NAME)

client = openai.AzureOpenAI(
    api_key=AZURE_API_KEY,
    azure_endpoint=AZURE_ENDPOINT,
    api_version=API_VERSION)

# openai_model = 'GPT-4o'
openai_model = DEPLOYMENT_NAME


# vector_db_path = "./Chainlit_rag_chatbot_18_Sept"
# gemini_model = 'gemini-2.5-flash'

# class MyVanna(ChromaDB_VectorStore, GoogleGeminiChat):
#     def __init__(self, config=None):
#         ChromaDB_VectorStore.__init__(self, config={'path': vector_db_path})
#         GoogleGeminiChat.__init__(self, config={'api_key': GOOGLE_API_KEY, 'model_name': gemini_model})

# gemini_model = 'gemini-2.5-flash'
# genai.configure(api_key=GOOGLE_API_KEY)
# model = genai.GenerativeModel(gemini_model)

# vn = MyVanna({'api_key': GOOGLE_API_KEY, 'model_name': gemini_model})


vector_db_path = "./Chainlit_rag_chatbot_22_Oct"

# Custom Vanna class using Azure OpenAI
class MyVanna(ChromaDB_VectorStore, OpenAI_Chat):
    def __init__(self, client, model_name):
        ChromaDB_VectorStore.__init__(self, config={'path': vector_db_path})
        self._default_model_name = model_name
        OpenAI_Chat.__init__(self, client=client, config={})

    def submit_prompt(self, prompt, **kwargs):
        if "model" not in kwargs:
            kwargs["model"] = self._default_model_name
        return super().submit_prompt(prompt, **kwargs)

vn = MyVanna(client, openai_model)

# vn = MyVanna({
#     'api_key': AZURE_API_KEY,
#     'model_name':openai_model,
#     'api_base': AZURE_ENDPOINT,
#     'api_version': API_VERSION,
#     'api_type': 'azure'
# })


# ## connecting to PrestoDB
Auth_token_key = os.getenv("Presto_query_authenticate_token")
user = os.getenv("user")
port = os.getenv("port")
host = os.getenv("host")

conn = prestodb.dbapi.connect(
        host=host,
        port=port,
        user = user,
        source='rstudio',
        catalog="awsdatacatalog",
        schema="express_dwh",
        http_scheme="http",
        http_headers = { "Authorization": f'''Bearer {Auth_token_key}'''})


def run_sql(sql: str) -> pd.DataFrame:
    df = pd.read_sql_query(sql, conn)
    return df


import pandas as pd
vn.run_sql = run_sql
vn.run_sql_is_set = True


# --- Adding Context of revenue and yield table
vn.train(documentation= f'''
Yield is defined as Total Gross Amount divided by Wbn Count - Use this formula to get yield at any level''')

vn.train(documentation= f'''
Table Name: data_analytics.bi_revenue_estimate_and_bird_data

Description:
- This table have granular-level shipment volume and gross revenue data for for B2C clients for latest 6 months. 
It captures volume and revenue data at product_type, client, zone, package type, shipment status, shipment weight, transport mode, closure date, month level.

Column names and details:
- product_type: product category (have values as 'B2C', 'Heavy')
- client: client name
- mode_of_transport: transport mode (have values as 'S', 'E'. S indicates by Surface, E indicates by air)
- bill_zone: zone of the shipment 
- pkg_type: package type (have values as 'Prepaid', 'COD', 'Pickup', 'REPL')
- scan_status: closure status of shipment (have values as 'Delivered', 'RTO', 'DTO')   
- int_wt_bucket: weight of the shipment in kg (have values as 0.0, 0.5, 1.0, 1.5, 2.0, 2.5 .... so on)  
- closure_date: closure date of the shipment (have values in 'YYYY-mm-dd' format like 2025-06-30, 2025-07-06 etc.)      
- closure_month: closure month of shipment (have value in Text format like 'July', 'June', 'May' etc)
- wbn_count: shipment volume at the level of data (value float type values) 
- gross_amount_sum: total gross revenue at the level of data (value in indian rupees and have float type values)
- ad: Table refresh date (last updated)
 ''')

vn.train(documentation= f''' Use this 'data_analytics.bi_revenue_estimate_and_bird_data' table for shipment volume, gross revenue, gross yield data at any level for recent 6 months
          ''')

vn.train(documentation="return the sql query in presto format, dont use semi colons; after query. Gross yield is revenue per shipment and it is in indian rupees ₹, and revenue is also in Indian rupees ₹")

# ----- Monthly gross yield for all b2c client

vn.train(
question="What is monthly trend for overall shipment volume, gross revenue, and gross yield for B2C clients"
,sql=""" 
 select product_type, month(closure_date) as closure_month, year(closure_date) as closure_year,
 cast(sum(wbn_count) as int) as wbn_sum, sum(gross_amount_sum) as gross_revenue_total, 
 round(sum(gross_amount_sum)/sum(wbn_count), 2) as gross_yield
 from
  ( select *
 FROM data_analytics.bi_revenue_estimate_and_bird_data 
    WHERE ad >= (select max(ad) from data_analytics.bi_revenue_estimate_and_bird_data)
        and product_type = 'B2C'
        )
group by 1,2,3
order by 3,2
""")

vn.train(
question="What is overall shipment volume, gross revenue, and gross yield for B2C clients for July 2025?"
,sql=""" 
 select product_type, month(closure_date) as closure_month, year(closure_date) as closure_year,
 cast(sum(wbn_count) as int) as wbn_sum, sum(gross_amount_sum) as gross_revenue_total, 
 round(sum(gross_amount_sum)/sum(wbn_count), 2) as gross_yield
 from
  ( select *
 FROM data_analytics.bi_revenue_estimate_and_bird_data 
    WHERE ad >= (select max(ad) from data_analytics.bi_revenue_estimate_and_bird_data)
        and product_type = 'B2C'
        and  month(closure_date) = 7 AND year(closure_date) = 2025
        )
group by 1,2,3
order by 3,2
""")

# ----- current month gross yield for b2c client wise

vn.train(
question="What is the current month's client-level gross yield, revenue, and volume?"
,sql=""" 
 select product_type, client, month(closure_date) as closure_month, year(closure_date) as closure_year,
 cast(sum(wbn_count) as int) as wbn_sum,
 sum(gross_amount_sum) as gross_revenue_total, 
 round(sum(gross_amount_sum)/sum(wbn_count), 2) as gross_yield
 from
  (select *
 FROM data_analytics.bi_revenue_estimate_and_bird_data 
    WHERE ad >= (select max(ad) from data_analytics.bi_revenue_estimate_and_bird_data)
        and product_type = 'B2C'
         AND month(closure_date) = month(current_date) AND year(closure_date) = year(current_date)
)
group by 1,2,3,4
order by 5 desc
""")


# 1. monthly gross yield for any client
vn.train(
question="What is the monthly volume, gross revenue, and gross yield for B2C client FTPL SURFACE?"
,sql="""
 select product_type, client, month(closure_date) as closure_month, year(closure_date) as closure_year,
 cast(sum(wbn_count) as int) as wbn_sum,
 sum(gross_amount_sum) as gross_revenue_total, 
 round(sum(gross_amount_sum)/sum(wbn_count), 2) as gross_yield
 from
  (select *
 FROM data_analytics.bi_revenue_estimate_and_bird_data 
    WHERE ad >= (select max(ad) from data_analytics.bi_revenue_estimate_and_bird_data)
        and product_type = 'B2C'
        and lower(client) = lower('FTPL SURFACE')
)
group by 1,2,3,4
order by 4,3
""")

vn.train(
question="What is the monthly gross yield, yield trend for B2C client FTPL SURFACE?"
,sql="""
 select product_type, client, month(closure_date) as closure_month, year(closure_date) as closure_year,
 cast(sum(wbn_count) as int) as wbn_sum,
 sum(gross_amount_sum) as gross_revenue_total, 
 round(sum(gross_amount_sum)/sum(wbn_count), 2) as gross_yield
 from
  (select *
 FROM data_analytics.bi_revenue_estimate_and_bird_data 
    WHERE ad >= (select max(ad) from data_analytics.bi_revenue_estimate_and_bird_data)
        and product_type = 'B2C'
        and lower(client) = lower('FTPL SURFACE')
)
group by 1,2,3,4
order by 4,3
""")



# 3. Monthly gross yield for any client for any particular completed month
vn.train(
question="What is the shipment volume, revenue, and gross yield for B2C client AJIO SURFACE for January 2025?"
,sql="""
select product_type, client, month(closure_date) as closure_month, year(closure_date) as closure_year,
 cast(sum(wbn_count) as int) as wbn_sum,
 sum(gross_amount_sum) as gross_revenue_total, 
 round(sum(gross_amount_sum)/sum(wbn_count), 2) as gross_yield
 from
  (select *
 FROM data_analytics.bi_revenue_estimate_and_bird_data 
    WHERE ad >= (select max(ad) from data_analytics.bi_revenue_estimate_and_bird_data)
        and product_type = 'B2C'
        and lower(client) = lower('AJIO SURFACE')
         AND month(closure_date) = 1 AND year(closure_date) = 2025
)
group by 1,2,3,4
order by 4,3 
""")

# 4.  MOM gross yield change any client for last 3 months
vn.train(
question="What is the volume, revenue, gross yield, and month-over-month change in gross yield for recent months for B2C client SNAPDEAL SURFACE?"
,sql = """
with mom_cl_yield as (
select *,  lag(gross_yield) over(partition by product_type, client order by closure_year, closure_month  ) as last_month_gross_yield
from (
 select product_type, client, month(closure_date) as closure_month, year(closure_date) as closure_year,
 cast(sum(wbn_count) as int) as wbn_sum,
 sum(gross_amount_sum) as gross_revenue_total, 
 round(sum(gross_amount_sum)/sum(wbn_count), 2) as gross_yield
 from
  (select *
 FROM data_analytics.bi_revenue_estimate_and_bird_data 
    WHERE ad >= (select max(ad) from data_analytics.bi_revenue_estimate_and_bird_data)
        and product_type = 'B2C'
        and lower(client) = lower('SNAPDEAL SURFACE')
)
group by 1,2,3,4
order by 4,3 )
)

select product_type, client, closure_month, closure_year, wbn_sum, gross_revenue_total, gross_yield,
case when last_month_gross_yield  > 0 then
round((gross_yield - last_month_gross_yield)*100/last_month_gross_yield,2) end as MOM_gross_yield_perc
from mom_cl_yield
""")


#   current month top 10 b2c clients by volume
vn.train(
question="Which are the top 10 B2C clients by shipment volume for the current month?"
,sql="""
select product_type, client, 
 cast(sum(wbn_count) as int) as wbn_sum,
 sum(gross_amount_sum) as gross_revenue_total, 
 round(sum(gross_amount_sum)/sum(wbn_count), 2) as gross_yield
 from
  (select *
    FROM data_analytics.bi_revenue_estimate_and_bird_data 
    WHERE ad >= (select max(ad) from data_analytics.bi_revenue_estimate_and_bird_data)
    and product_type = 'B2C'
    AND month(closure_date) = month(current_date) AND year(closure_date) = year(current_date) )
group by 1,2
order by 3 desc 
limit 10 
""" )

#   any month top 10 b2c clients by yield and volume > 5k
vn.train(
question="Which were the top 10 B2C clients by gross yield and had shipment volume > 5K for the current month?"
,sql="""
select product_type, client, 
 cast(sum(wbn_count) as int) as wbn_sum,
 sum(gross_amount_sum) as gross_revenue_total, 
 round(sum(gross_amount_sum)/sum(wbn_count), 2) as gross_yield
 from
  (select *
    FROM data_analytics.bi_revenue_estimate_and_bird_data 
    WHERE ad >= (select max(ad) from data_analytics.bi_revenue_estimate_and_bird_data)
    and product_type = 'B2C'
    AND month(closure_date) = month(current_date) AND year(closure_date) = year(current_date) )
group by 1,2
having sum(wbn_count) > 5000
order by 5 desc 
limit 10
""" )

#   overall top 10 b2c clients by volume
vn.train(
question="Which are the top 10 B2C clients by shipment volume across all months?"
,sql="""
select product_type, client, 
 cast(sum(wbn_count) as int) as wbn_sum,
 sum(gross_amount_sum) as gross_revenue_total, 
 round(sum(gross_amount_sum)/sum(wbn_count), 2) as gross_yield
 from
  (select *
 FROM data_analytics.bi_revenue_estimate_and_bird_data 
    WHERE ad >= (select max(ad) from data_analytics.bi_revenue_estimate_and_bird_data)
        and product_type = 'B2C'
)
group by 1,2
order by 3 desc 
limit 10
""" )

# 5.  MOM gross yield change for 3 months for top 10 clients
vn.train(
question="What is the month-over-month change in gross yield for the top 10 B2C clients by volume?"
,sql="""
with top_10_b2c_cl as (
select product_type, client, 
 cast(sum(wbn_count) as int) as wbn_sum,
 sum(gross_amount_sum) as gross_revenue_total, 
 round(sum(gross_amount_sum)/sum(wbn_count), 2) as gross_yield
 from data_analytics.bi_revenue_estimate_and_bird_data 
    WHERE ad >= (select max(ad) from data_analytics.bi_revenue_estimate_and_bird_data)
        and product_type = 'B2C'
        AND month(closure_date) = month(current_date) AND year(closure_date) = year(current_date)
group by 1,2
order by 3 desc 
limit 10 
)

, mom_cl_yield as (
select *,  lag(gross_yield) over(partition by product_type, client order by closure_year, closure_month  ) as last_month_gross_yield
from (
 select product_type, client, month(closure_date) as closure_month, year(closure_date) as closure_year,
 cast(sum(wbn_count) as int) as wbn_sum,
 sum(gross_amount_sum) as gross_revenue_total, 
 round(sum(gross_amount_sum)/sum(wbn_count), 2) as gross_yield
 from
  (select *
 FROM data_analytics.bi_revenue_estimate_and_bird_data 
    WHERE ad >= (select max(ad) from data_analytics.bi_revenue_estimate_and_bird_data)
        and product_type = 'B2C'
        and client in (select distinct client from top_10_b2c_cl)
)
group by 1,2,3,4
order by 4,3 )
)

select product_type, client, closure_month, closure_year, wbn_sum, gross_revenue_total, gross_yield,
case when last_month_gross_yield  > 0 then
round((gross_yield - last_month_gross_yield)*100/last_month_gross_yield,2) end as MOM_gross_yield_perc
from mom_cl_yield
order by closure_year, closure_month desc, wbn_sum desc
""")



#  Reasons for yield change for a client -- all granular factors
vn.train(
question="Why gross yield changed (dropped/ increased) for B2C client 'FLIPKART E2E' in the current month compared to previous month?"
,sql="""
select * FROM

(select product_type, client, month(closure_date) as closure_month, year(closure_date) as closure_year,
'mode_of_transport' as affecting_factor,  mode_of_transport as factor_value, 
sum(wbn_count) as wbn_count,  sum(gross_amount_sum) as Gross_Amount, 
sum(gross_amount_sum)/sum(wbn_count) as gross_yield 
from 
(select *
, CASE WHEN int_wt_bucket <= 2 THEN '0-2 KG'
    WHEN int_wt_bucket > 2 AND int_wt_bucket <= 5 THEN '2-5 KG'
    ELSE '5+ KG' END AS int_wt_bucket_group
from data_analytics.bi_revenue_estimate_and_bird_data 
where ad >= (Select max(ad) from data_analytics.bi_revenue_estimate_and_bird_data)
and product_type in ('B2C')
and lower(client) = lower('FLIPKART E2E')
and
((month(closure_date) = month(current_date) AND year(closure_date) = year(current_date)) OR
(month(closure_date) = month(date_add('month', -1, current_date)) AND year(closure_date) = year(date_add('month', -1, current_date))))
)
group by 1,2,3,4, 5, 6
order by 6,3)

union 

(select product_type, client, month(closure_date) as closure_month, year(closure_date) as closure_year,
'bill_zone' as affecting_factor,  bill_zone as factor_value, 
sum(wbn_count) as wbn_count,  sum(gross_amount_sum) as Gross_Amount, 
sum(gross_amount_sum)/sum(wbn_count) as gross_yield 
from 
(select *
, CASE WHEN int_wt_bucket <= 2 THEN '0-2 KG'
    WHEN int_wt_bucket > 2 AND int_wt_bucket <= 5 THEN '2-5 KG'
    ELSE '5+ KG' END AS int_wt_bucket_group
from data_analytics.bi_revenue_estimate_and_bird_data 
where ad >= (Select max(ad) from data_analytics.bi_revenue_estimate_and_bird_data)
and product_type in ('B2C')
and lower(client) = lower('FLIPKART E2E')
and
((month(closure_date) = month(current_date) AND year(closure_date) = year(current_date)) OR
(month(closure_date) = month(date_add('month', -1, current_date)) AND year(closure_date) = year(date_add('month', -1, current_date))))
)
group by 1,2,3,4, 5, 6
order by 6,3 )

union 

(select product_type, client, month(closure_date) as closure_month, year(closure_date) as closure_year,
'pkg_type' as affecting_factor,  pkg_type as factor_value, 
sum(wbn_count) as wbn_count,  sum(gross_amount_sum) as Gross_Amount, 
sum(gross_amount_sum)/sum(wbn_count) as gross_yield 
from 
(select *
, CASE WHEN int_wt_bucket <= 2 THEN '0-2 KG'
    WHEN int_wt_bucket > 2 AND int_wt_bucket <= 5 THEN '2-5 KG'
    ELSE '5+ KG' END AS int_wt_bucket_group
from data_analytics.bi_revenue_estimate_and_bird_data 
where ad >= (Select max(ad) from data_analytics.bi_revenue_estimate_and_bird_data)
and product_type in ('B2C')
and lower(client) = lower('FLIPKART E2E')
and
((month(closure_date) = month(current_date) AND year(closure_date) = year(current_date)) OR
(month(closure_date) = month(date_add('month', -1, current_date)) AND year(closure_date) = year(date_add('month', -1, current_date))))
)
group by 1,2,3,4, 5, 6
order by 6,3 )

union 

(select product_type, client, month(closure_date) as closure_month, year(closure_date) as closure_year,
'scan_status' as affecting_factor,  scan_status as factor_value, 
sum(wbn_count) as wbn_count,  sum(gross_amount_sum) as Gross_Amount, 
sum(gross_amount_sum)/sum(wbn_count) as gross_yield 
from 
(select *
, CASE WHEN int_wt_bucket <= 2 THEN '0-2 KG'
    WHEN int_wt_bucket > 2 AND int_wt_bucket <= 5 THEN '2-5 KG'
    ELSE '5+ KG' END AS int_wt_bucket_group
from data_analytics.bi_revenue_estimate_and_bird_data 
where ad >= (Select max(ad) from data_analytics.bi_revenue_estimate_and_bird_data)
and product_type in ('B2C')
and lower(client) = lower('FLIPKART E2E')
and
((month(closure_date) = month(current_date) AND year(closure_date) = year(current_date)) OR
(month(closure_date) = month(date_add('month', -1, current_date)) AND year(closure_date) = year(date_add('month', -1, current_date))))
)
group by 1,2,3,4, 5, 6
order by 6,3 )

union 

(select product_type, client, month(closure_date) as closure_month, year(closure_date) as closure_year,
'int_wt_bucket_group' as affecting_factor,  int_wt_bucket_group as factor_value, 
sum(wbn_count) as wbn_count,  sum(gross_amount_sum) as Gross_Amount, 
sum(gross_amount_sum)/sum(wbn_count) as gross_yield 
from 
(select *
, CASE WHEN int_wt_bucket <= 2 THEN '0-2 KG'
    WHEN int_wt_bucket > 2 AND int_wt_bucket <= 5 THEN '2-5 KG'
    ELSE '5+ KG' END AS int_wt_bucket_group
from data_analytics.bi_revenue_estimate_and_bird_data 
where ad >= (Select max(ad) from data_analytics.bi_revenue_estimate_and_bird_data)
and product_type in ('B2C')
and lower(client) = lower('FLIPKART E2E')
and
((month(closure_date) = month(current_date) AND year(closure_date) = year(current_date)) OR
(month(closure_date) = month(date_add('month', -1, current_date)) AND year(closure_date) = year(date_add('month', -1, current_date))))
)
group by 1,2,3,4, 5, 6
order by 6,3 )

order by 5,6,3
""")

vn.train(
question="How did different factors affected gross yield for B2C client 'FLIPKART E2E' in current month compared to the previous month?"
,sql="""
select * FROM

(select product_type, client, month(closure_date) as closure_month, year(closure_date) as closure_year,
'mode_of_transport' as affecting_factor,  mode_of_transport as factor_value, 
sum(wbn_count) as wbn_count,  sum(gross_amount_sum) as Gross_Amount, 
sum(gross_amount_sum)/sum(wbn_count) as gross_yield 
from 
(select *
, CASE WHEN int_wt_bucket <= 2 THEN '0-2 KG'
    WHEN int_wt_bucket > 2 AND int_wt_bucket <= 5 THEN '2-5 KG'
    ELSE '5+ KG' END AS int_wt_bucket_group
from data_analytics.bi_revenue_estimate_and_bird_data 
where ad >= (Select max(ad) from data_analytics.bi_revenue_estimate_and_bird_data)
and product_type in ('B2C')
and lower(client) = lower('FLIPKART E2E')
and
((month(closure_date) = month(current_date) AND year(closure_date) = year(current_date)) OR
(month(closure_date) = month(date_add('month', -1, current_date)) AND year(closure_date) = year(date_add('month', -1, current_date))))
)
group by 1,2,3,4, 5, 6
order by 6,3)

union 

(select product_type, client, month(closure_date) as closure_month, year(closure_date) as closure_year,
'bill_zone' as affecting_factor,  bill_zone as factor_value, 
sum(wbn_count) as wbn_count,  sum(gross_amount_sum) as Gross_Amount, 
sum(gross_amount_sum)/sum(wbn_count) as gross_yield 
from 
(select *
, CASE WHEN int_wt_bucket <= 2 THEN '0-2 KG'
    WHEN int_wt_bucket > 2 AND int_wt_bucket <= 5 THEN '2-5 KG'
    ELSE '5+ KG' END AS int_wt_bucket_group
from data_analytics.bi_revenue_estimate_and_bird_data 
where ad >= (Select max(ad) from data_analytics.bi_revenue_estimate_and_bird_data)
and product_type in ('B2C')
and lower(client) = lower('FLIPKART E2E')
and
((month(closure_date) = month(current_date) AND year(closure_date) = year(current_date)) OR
(month(closure_date) = month(date_add('month', -1, current_date)) AND year(closure_date) = year(date_add('month', -1, current_date))))
)
group by 1,2,3,4, 5, 6
order by 6,3 )

union 

(select product_type, client, month(closure_date) as closure_month, year(closure_date) as closure_year,
'pkg_type' as affecting_factor,  pkg_type as factor_value, 
sum(wbn_count) as wbn_count,  sum(gross_amount_sum) as Gross_Amount, 
sum(gross_amount_sum)/sum(wbn_count) as gross_yield 
from 
(select *
, CASE WHEN int_wt_bucket <= 2 THEN '0-2 KG'
    WHEN int_wt_bucket > 2 AND int_wt_bucket <= 5 THEN '2-5 KG'
    ELSE '5+ KG' END AS int_wt_bucket_group
from data_analytics.bi_revenue_estimate_and_bird_data 
where ad >= (Select max(ad) from data_analytics.bi_revenue_estimate_and_bird_data)
and product_type in ('B2C')
and lower(client) = lower('FLIPKART E2E')
and
((month(closure_date) = month(current_date) AND year(closure_date) = year(current_date)) OR
(month(closure_date) = month(date_add('month', -1, current_date)) AND year(closure_date) = year(date_add('month', -1, current_date))))
)
group by 1,2,3,4, 5, 6
order by 6,3 )

union 

(select product_type, client, month(closure_date) as closure_month, year(closure_date) as closure_year,
'scan_status' as affecting_factor,  scan_status as factor_value, 
sum(wbn_count) as wbn_count,  sum(gross_amount_sum) as Gross_Amount, 
sum(gross_amount_sum)/sum(wbn_count) as gross_yield 
from 
(select *
, CASE WHEN int_wt_bucket <= 2 THEN '0-2 KG'
    WHEN int_wt_bucket > 2 AND int_wt_bucket <= 5 THEN '2-5 KG'
    ELSE '5+ KG' END AS int_wt_bucket_group
from data_analytics.bi_revenue_estimate_and_bird_data 
where ad >= (Select max(ad) from data_analytics.bi_revenue_estimate_and_bird_data)
and product_type in ('B2C')
and lower(client) = lower('FLIPKART E2E')
and
((month(closure_date) = month(current_date) AND year(closure_date) = year(current_date)) OR
(month(closure_date) = month(date_add('month', -1, current_date)) AND year(closure_date) = year(date_add('month', -1, current_date))))
)
group by 1,2,3,4, 5, 6
order by 6,3 )

union 

(select product_type, client, month(closure_date) as closure_month, year(closure_date) as closure_year,
'int_wt_bucket_group' as affecting_factor,  int_wt_bucket_group as factor_value, 
sum(wbn_count) as wbn_count,  sum(gross_amount_sum) as Gross_Amount, 
sum(gross_amount_sum)/sum(wbn_count) as gross_yield 
from 
(select *
, CASE WHEN int_wt_bucket <= 2 THEN '0-2 KG'
    WHEN int_wt_bucket > 2 AND int_wt_bucket <= 5 THEN '2-5 KG'
    ELSE '5+ KG' END AS int_wt_bucket_group
from data_analytics.bi_revenue_estimate_and_bird_data 
where ad >= (Select max(ad) from data_analytics.bi_revenue_estimate_and_bird_data)
and product_type in ('B2C')
and lower(client) = lower('FLIPKART E2E')
and
((month(closure_date) = month(current_date) AND year(closure_date) = year(current_date)) OR
(month(closure_date) = month(date_add('month', -1, current_date)) AND year(closure_date) = year(date_add('month', -1, current_date))))
)
group by 1,2,3,4, 5, 6
order by 6,3 )

order by 5,6,3
""")


##  Overall business gross yield changed
vn.train(
question="Why Overall gross yield for B2C business changed in current month compared to the previous month?"
,sql="""
select * FROM

(select product_type, month(closure_date) as closure_month, year(closure_date) as closure_year,
'mode_of_transport' as affecting_factor,  mode_of_transport as factor_value, 
sum(wbn_count) as wbn_count,  sum(gross_amount_sum) as Gross_Amount, 
sum(gross_amount_sum)/sum(wbn_count) as gross_yield 
from 
(select *
, CASE WHEN int_wt_bucket <= 2 THEN '0-2 KG'
    WHEN int_wt_bucket > 2 AND int_wt_bucket <= 5 THEN '2-5 KG'
    ELSE '5+ KG' END AS int_wt_bucket_group
from data_analytics.bi_revenue_estimate_and_bird_data 
where ad >= (Select max(ad) from data_analytics.bi_revenue_estimate_and_bird_data)
and product_type in ('B2C')
and
((month(closure_date) = month(current_date) AND year(closure_date) = year(current_date)) OR
(month(closure_date) = month(date_add('month', -1, current_date)) AND year(closure_date) = year(date_add('month', -1, current_date))))
)
group by 1,2,3,4, 5
order by 5,2)

union 

(select product_type, month(closure_date) as closure_month, year(closure_date) as closure_year,
'bill_zone' as affecting_factor,  bill_zone as factor_value, 
sum(wbn_count) as wbn_count,  sum(gross_amount_sum) as Gross_Amount, 
sum(gross_amount_sum)/sum(wbn_count) as gross_yield 
from 
(select *
, CASE WHEN int_wt_bucket <= 2 THEN '0-2 KG'
    WHEN int_wt_bucket > 2 AND int_wt_bucket <= 5 THEN '2-5 KG'
    ELSE '5+ KG' END AS int_wt_bucket_group
from data_analytics.bi_revenue_estimate_and_bird_data 
where ad >= (Select max(ad) from data_analytics.bi_revenue_estimate_and_bird_data)
and product_type in ('B2C')
and
((month(closure_date) = month(current_date) AND year(closure_date) = year(current_date)) OR
(month(closure_date) = month(date_add('month', -1, current_date)) AND year(closure_date) = year(date_add('month', -1, current_date))))
)
group by 1,2,3,4, 5
order by 5,2)

union 

(select product_type, month(closure_date) as closure_month, year(closure_date) as closure_year,
'pkg_type' as affecting_factor,  pkg_type as factor_value, 
sum(wbn_count) as wbn_count,  sum(gross_amount_sum) as Gross_Amount, 
sum(gross_amount_sum)/sum(wbn_count) as gross_yield 
from 
(select *
, CASE WHEN int_wt_bucket <= 2 THEN '0-2 KG'
    WHEN int_wt_bucket > 2 AND int_wt_bucket <= 5 THEN '2-5 KG'
    ELSE '5+ KG' END AS int_wt_bucket_group
from data_analytics.bi_revenue_estimate_and_bird_data 
where ad >= (Select max(ad) from data_analytics.bi_revenue_estimate_and_bird_data)
and product_type in ('B2C')
and
((month(closure_date) = month(current_date) AND year(closure_date) = year(current_date)) OR
(month(closure_date) = month(date_add('month', -1, current_date)) AND year(closure_date) = year(date_add('month', -1, current_date))))
)
group by 1,2,3,4, 5
order by 5,2)

union 

(select product_type, month(closure_date) as closure_month, year(closure_date) as closure_year,
'scan_status' as affecting_factor,  scan_status as factor_value, 
sum(wbn_count) as wbn_count,  sum(gross_amount_sum) as Gross_Amount, 
sum(gross_amount_sum)/sum(wbn_count) as gross_yield 
from 
(select *
, CASE WHEN int_wt_bucket <= 2 THEN '0-2 KG'
    WHEN int_wt_bucket > 2 AND int_wt_bucket <= 5 THEN '2-5 KG'
    ELSE '5+ KG' END AS int_wt_bucket_group
from data_analytics.bi_revenue_estimate_and_bird_data 
where ad >= (Select max(ad) from data_analytics.bi_revenue_estimate_and_bird_data)
and product_type in ('B2C')
and
((month(closure_date) = month(current_date) AND year(closure_date) = year(current_date)) OR
(month(closure_date) = month(date_add('month', -1, current_date)) AND year(closure_date) = year(date_add('month', -1, current_date))))
)
group by 1,2,3,4, 5
order by 5,2)

union 

(select product_type, month(closure_date) as closure_month, year(closure_date) as closure_year,
'int_wt_bucket_group' as affecting_factor,  int_wt_bucket_group as factor_value, 
sum(wbn_count) as wbn_count,  sum(gross_amount_sum) as Gross_Amount, 
sum(gross_amount_sum)/sum(wbn_count) as gross_yield 
from 
(select *
, CASE WHEN int_wt_bucket <= 2 THEN '0-2 KG'
    WHEN int_wt_bucket > 2 AND int_wt_bucket <= 5 THEN '2-5 KG'
    ELSE '5+ KG' END AS int_wt_bucket_group
from data_analytics.bi_revenue_estimate_and_bird_data 
where ad >= (Select max(ad) from data_analytics.bi_revenue_estimate_and_bird_data)
and product_type in ('B2C')
and
((month(closure_date) = month(current_date) AND year(closure_date) = year(current_date)) OR
(month(closure_date) = month(date_add('month', -1, current_date)) AND year(closure_date) = year(date_add('month', -1, current_date))))
)
group by 1,2,3,4, 5
order by 5,2)

order by 4,5,3
""")

# 1.  Reasons for yield change for given client -- package type
vn.train(
question="How did the current month's gross yield change across package types or payment types for B2C client 'AMAZONINDIA'?"
,sql="""
select product_type, client, pkg_type,
 month(closure_date) as closure_month, year(closure_date) as closure_year,
 closure_month as  Closure_month_str,
sum(wbn_count) as wbn_count,  sum(gross_amount_sum) as Gross_Amount, 
sum(gross_amount_sum)/sum(wbn_count) as gross_yield 
from 
(select *
, CASE WHEN int_wt_bucket <= 2 THEN '0-2 KG'
    WHEN int_wt_bucket > 2 AND int_wt_bucket <= 5 THEN '2-5 KG'
    ELSE '5+ KG' END AS int_wt_bucket_group
from data_analytics.bi_revenue_estimate_and_bird_data 
where ad >= (Select max(ad) from data_analytics.bi_revenue_estimate_and_bird_data)
and product_type in ('B2C')
and lower(client) = lower('AMAZONINDIA')
and
((month(closure_date) = month(current_date) AND year(closure_date) = year(current_date)) OR
(month(closure_date) = month(date_add('month', -1, current_date)) AND year(closure_date) = year(date_add('month', -1, current_date)))) 
)
group by 1,2,3,4, 5, 6
order by 7 desc
""")

vn.train(
question="How has the current month's gross yield changed compared to the previous month across package types or payment types for B2C client 'AMAZONINDIA'?"
,sql="""
select product_type, client, pkg_type,
 month(closure_date) as closure_month, year(closure_date) as closure_year,
 closure_month as  Closure_month_str,
sum(wbn_count) as wbn_count,  sum(gross_amount_sum) as Gross_Amount, 
sum(gross_amount_sum)/sum(wbn_count) as gross_yield 
from 
(select *
, CASE WHEN int_wt_bucket <= 2 THEN '0-2 KG'
    WHEN int_wt_bucket > 2 AND int_wt_bucket <= 5 THEN '2-5 KG'
    ELSE '5+ KG' END AS int_wt_bucket_group
from data_analytics.bi_revenue_estimate_and_bird_data 
where ad >= (Select max(ad) from data_analytics.bi_revenue_estimate_and_bird_data)
and product_type in ('B2C')
and lower(client) = lower('AMAZONINDIA')
and
((month(closure_date) = month(current_date) AND year(closure_date) = year(current_date)) OR
(month(closure_date) = month(date_add('month', -1, current_date)) AND year(closure_date) = year(date_add('month', -1, current_date))))
)
group by 1,2,3,4, 5, 6
order by 7 desc
""")

# 2.  Reasons for yield change for given client -- bill zone
vn.train(
question="How did the current month's gross yield change across bill zones for B2C client 'AJIO SURFACE'?"
,sql="""
select product_type, client, bill_zone,
 month(closure_date) as closure_month, year(closure_date) as closure_year,
 closure_month as  Closure_month_str,
sum(wbn_count) as wbn_count,  sum(gross_amount_sum) as Gross_Amount, 
sum(gross_amount_sum)/sum(wbn_count) as gross_yield 
from 
(select *
, CASE WHEN int_wt_bucket <= 2 THEN '0-2 KG'
    WHEN int_wt_bucket > 2 AND int_wt_bucket <= 5 THEN '2-5 KG'
    ELSE '5+ KG' END AS int_wt_bucket_group
from data_analytics.bi_revenue_estimate_and_bird_data 
where ad >= (Select max(ad) from data_analytics.bi_revenue_estimate_and_bird_data)
and product_type in ('B2C')
and lower(client) = lower('AJIO SURFACE')
and
((month(closure_date) = month(current_date) AND year(closure_date) = year(current_date)) OR
(month(closure_date) = month(date_add('month', -1, current_date)) AND year(closure_date) = year(date_add('month', -1, current_date)))) 
)
group by 1, 2, 3, 4, 5, 6
order by 7 desc
""")

vn.train(
question="How has the current month's gross yield changed compared to the previous month across bill zones for B2C client 'AJIO SURFACE'?"
,sql="""
select product_type, client, bill_zone,
 month(closure_date) as closure_month, year(closure_date) as closure_year,
 closure_month as  Closure_month_str,
sum(wbn_count) as wbn_count,  sum(gross_amount_sum) as Gross_Amount, 
sum(gross_amount_sum)/sum(wbn_count) as gross_yield 
from 
(select *
, CASE WHEN int_wt_bucket <= 2 THEN '0-2 KG'
    WHEN int_wt_bucket > 2 AND int_wt_bucket <= 5 THEN '2-5 KG'
    ELSE '5+ KG' END AS int_wt_bucket_group
from data_analytics.bi_revenue_estimate_and_bird_data 
where ad >= (Select max(ad) from data_analytics.bi_revenue_estimate_and_bird_data)
and product_type in ('B2C')
and lower(client) = lower('AJIO SURFACE')
and
((month(closure_date) = month(current_date) AND year(closure_date) = year(current_date)) OR
(month(closure_date) = month(date_add('month', -1, current_date)) AND year(closure_date) = year(date_add('month', -1, current_date)))
)
group by 1, 2, 3, 4, 5, 6
order by 7 desc
""")

# 3.  Reasons for yield change for a client -- Status
vn.train(
question="How did the current month's gross yield change across shipment statuses for B2C client 'FTPL'?"
,sql = """
select product_type, client, scan_status,
 month(closure_date) as closure_month, year(closure_date) as closure_year,
 closure_month as  Closure_month_str,
sum(wbn_count) as wbn_count,  sum(gross_amount_sum) as Gross_Amount, 
sum(gross_amount_sum)/sum(wbn_count) as gross_yield 
from 
(select *
, CASE WHEN int_wt_bucket <= 2 THEN '0-2 KG'
    WHEN int_wt_bucket > 2 AND int_wt_bucket <= 5 THEN '2-5 KG'
    ELSE '5+ KG' END AS int_wt_bucket_group
from data_analytics.bi_revenue_estimate_and_bird_data 
where ad >= (Select max(ad) from data_analytics.bi_revenue_estimate_and_bird_data)
and product_type in ('B2C')
and lower(client) = lower('FTPL')
and
((month(closure_date) = month(current_date) AND year(closure_date) = year(current_date)) OR
(month(closure_date) = month(date_add('month', -1, current_date)) AND year(closure_date) = year(date_add('month', -1, current_date)))) 
)
group by 1, 2, 3, 4, 5, 6
order by 7 desc
""")

vn.train(
question="How has the current month's gross yield changed compared to the previous month across shipment statuses for B2C client 'FTPL'?"
,sql = """
select product_type, client, scan_status,
 month(closure_date) as closure_month, year(closure_date) as closure_year,
 closure_month as  Closure_month_str,
sum(wbn_count) as wbn_count,  sum(gross_amount_sum) as Gross_Amount, 
sum(gross_amount_sum)/sum(wbn_count) as gross_yield 
from 
(select *
, CASE WHEN int_wt_bucket <= 2 THEN '0-2 KG'
    WHEN int_wt_bucket > 2 AND int_wt_bucket <= 5 THEN '2-5 KG'
    ELSE '5+ KG' END AS int_wt_bucket_group
from data_analytics.bi_revenue_estimate_and_bird_data 
where ad >= (Select max(ad) from data_analytics.bi_revenue_estimate_and_bird_data)
and product_type in ('B2C')
and lower(client) = lower('FTPL')
and
((month(closure_date) = month(current_date) AND year(closure_date) = year(current_date)) OR
(month(closure_date) = month(date_add('month', -1, current_date)) AND year(closure_date) = year(date_add('month', -1, current_date)))) 
)
group by 1, 2, 3, 4, 5, 6
order by 7 desc
""")

# 4.  Reasons for yield change for a client -- mode of transport
vn.train(
question="How did the current month's gross yield change across transport modes for B2C client 'Nykaa E Retail Surface'?"
,sql="""
select product_type, client, mode_of_transport,
 month(closure_date) as closure_month, year(closure_date) as closure_year,
 closure_month as  Closure_month_str,
sum(wbn_count) as wbn_count,  sum(gross_amount_sum) as Gross_Amount, 
sum(gross_amount_sum)/sum(wbn_count) as gross_yield 
from 
(select *
, CASE WHEN int_wt_bucket <= 2 THEN '0-2 KG'
    WHEN int_wt_bucket > 2 AND int_wt_bucket <= 5 THEN '2-5 KG'
    ELSE '5+ KG' END AS int_wt_bucket_group
from data_analytics.bi_revenue_estimate_and_bird_data 
where ad >= (Select max(ad) from data_analytics.bi_revenue_estimate_and_bird_data)
and product_type in ('B2C')
and lower(client) = lower('Nykaa E Retail Surface')
and
((month(closure_date) = month(current_date) AND year(closure_date) = year(current_date)) OR
(month(closure_date) = month(date_add('month', -1, current_date)) AND year(closure_date) = year(date_add('month', -1, current_date)))) 
)
group by 1, 2, 3, 4, 5, 6
order by 7 desc
""")

vn.train(
question="How has the current month's gross yield changed compared to the previous month across transport modes for B2C client 'Nykaa E Retail Surface'?"
,sql="""
select product_type, client, mode_of_transport,
 month(closure_date) as closure_month, year(closure_date) as closure_year,
 closure_month as  Closure_month_str,
sum(wbn_count) as wbn_count,  sum(gross_amount_sum) as Gross_Amount, 
sum(gross_amount_sum)/sum(wbn_count) as gross_yield 
from 
(select *
, CASE WHEN int_wt_bucket <= 2 THEN '0-2 KG'
    WHEN int_wt_bucket > 2 AND int_wt_bucket <= 5 THEN '2-5 KG'
    ELSE '5+ KG' END AS int_wt_bucket_group
from data_analytics.bi_revenue_estimate_and_bird_data 
where ad >= (Select max(ad) from data_analytics.bi_revenue_estimate_and_bird_data)
and product_type in ('B2C')
and lower(client) = lower('Nykaa E Retail Surface')
and
((month(closure_date) = month(current_date) AND year(closure_date) = year(current_date)) OR
(month(closure_date) = month(date_add('month', -1, current_date)) AND year(closure_date) = year(date_add('month', -1, current_date)))) 
)
group by 1, 2, 3, 4, 5, 6
order by 7 desc
""")

# 5.  Reasons for yield change for a client -- Weight bucket
vn.train(
question="How did shipment weights or weight buckets affect the current month's gross yield for B2C client FLIPKART E2E?"
,sql="""
select product_type, client, int_wt_bucket_group,
 month(closure_date) as closure_month, year(closure_date) as closure_year,
 closure_month as  Closure_month_str,
sum(wbn_count) as wbn_count,  sum(gross_amount_sum) as Gross_Amount, 
sum(gross_amount_sum)/sum(wbn_count) as gross_yield 
from 
(select *
, CASE WHEN int_wt_bucket <= 2 THEN '0-2 KG'
    WHEN int_wt_bucket > 2 AND int_wt_bucket <= 5 THEN '2-5 KG'
    ELSE '5+ KG' END AS int_wt_bucket_group
from data_analytics.bi_revenue_estimate_and_bird_data 
where ad >= (Select max(ad) from data_analytics.bi_revenue_estimate_and_bird_data)
and product_type in ('B2C')
and lower(client) = lower('FLIPKART E2E')
and
(month(closure_date) = month(current_date) AND year(closure_date) = year(current_date)) 
)
group by 1, 2, 3, 4, 5, 6
order by 7 desc
""")

vn.train(
question="How did shipment weights or weight buckets affect the current month's gross yield compared to the previous month for B2C client FLIPKART E2E?"
,sql="""
select product_type, client, int_wt_bucket_group,
 month(closure_date) as closure_month, year(closure_date) as closure_year,
 closure_month as  Closure_month_str,
sum(wbn_count) as wbn_count,  sum(gross_amount_sum) as Gross_Amount, 
sum(gross_amount_sum)/sum(wbn_count) as gross_yield 
from 
(select *
, CASE WHEN int_wt_bucket <= 2 THEN '0-2 KG'
    WHEN int_wt_bucket > 2 AND int_wt_bucket <= 5 THEN '2-5 KG'
    ELSE '5+ KG' END AS int_wt_bucket_group
from data_analytics.bi_revenue_estimate_and_bird_data 
where ad >= (Select max(ad) from data_analytics.bi_revenue_estimate_and_bird_data)
and product_type in ('B2C')
and lower(client) = lower('FLIPKART E2E')
and
((month(closure_date) = month(current_date) AND year(closure_date) = year(current_date)) OR
(month(closure_date) = month(date_add('month', -1, current_date)) AND year(closure_date) = year(date_add('month', -1, current_date)))) 
)
group by 1, 2, 3, 4, 5, 6
order by 7 desc
""")


# Adding context about pnl data of b2c heavy clients pnl data table

vn.train(documentation= f'''
Table Name: data_analytics.bi_heavy_clients_revenue_and_pnl_data

Description:
- This table contains revenue, and cost data at month, year, client, shipment status, zone level. This table captures each cost components,
         total cost, margins for B2C heavy shipments only.

Usage Guidance:
- Use this table only when cost, margins, cost components are required for b2c heavy clients.''')

vn.train(documentation= f''' use this table data_analytics.bi_heavy_clients_revenue_and_pnl_data, when user asks about cost components,
         total cost, margins information for B2C heavy clients.''')

# 1. B2C Heavy overall volume, revenue, cost breakout, total cost, margin for particular month
vn.train(
question="What is the overall volume, revenue, cost factors, total costs, and margin percentage for B2C Heavy shipments for February 2025?"
,sql="""
select  month, year, 
sum(mwn_count) as master_waybill_count, sum(wbn_count) as waybill_count, 
sum(charged_weight)/sum(wbn_count) as avg_charged_weight, sum(gross_revenue) as gross_revenue, 
sum(gross_revenue)/sum(wbn_count) as gross_yield, 
sum(fm_fleet_cost) as fm_fleet_cost, sum(fm_mp_cost) as fm_mp_cost, sum(fm_bp_cost) as fm_bp_cost, 
sum(fm_constellation_cost) as fm_constellation_cost,
sum(lm_fleet_cost) as lm_fleet_cost, sum(lm_mp_cost) as lm_mp_cost, sum(lm_bp_cost) as lm_bp_cost, 
sum(lm_constellation_cost) as lm_constellation_cost,
sum(rt_fleet_cost) as rt_fleet_cost, sum(rt_mp_cost) as rt_mp_cost, sum(rt_bp_cost) as rt_bp_cost, 
sum(rt_constellation_cost) as rt_constellation_cost,
sum(ipc_incenter_cost) as ipc_incenter_cost, sum(sc_incenter_cost) as sc_incenter_cost, 
sum(hub_incenter_cost) as hub_incenter_cost,
sum(line_haul_intra_cost) as line_haul_intra_cost, sum(line_haul_inter_cost) as line_haul_inter_cost,
sum(lma_cost) as lma_cost, sum(biker_fuel_cost) as biker_fuel_cost,
sum(consumables_cost) as consumables_cost, sum(other_opex_cost) as other_opex_cost, sum(aws_cost) as aws_cost,
sum(cms_cost) as cms_cost, sum(loss_damage_cost) as loss_damage_cost, sum(total_cost) as total_cost,
sum(total_cost)/sum(wbn_count) as cost_per_shipment,
(sum(gross_revenue)-sum(total_cost))/sum(wbn_count) as margin_per_shipment,
(sum(gross_revenue)-sum(total_cost))*100/sum(gross_revenue) as margin_percentage
from
(select * from data_analytics.bi_heavy_clients_revenue_and_pnl_data
where ad >= (select max(ad) from data_analytics.bi_heavy_clients_revenue_and_pnl_data)
and month = 2 and year = 2025)
group by 1,2
order by 1,2
""")

#  B2C Heavy monthly overall volume, revenue, cost breakout, total cost, margin 
vn.train(
question="What is the monthly overall volume, revenue, total costs, and margin percentage for B2C Heavy shipments?"
,sql="""
select  month, year, 
sum(mwn_count) as master_waybill_count, sum(wbn_count) as waybill_count, 
sum(charged_weight)/sum(wbn_count) as avg_charged_weight, sum(gross_revenue) as gross_revenue, 
sum(gross_revenue)/sum(wbn_count) as gross_yield, 
 sum(total_cost) as total_cost,
sum(total_cost)/sum(wbn_count) as cost_per_shipment,
(sum(gross_revenue)-sum(total_cost))/sum(wbn_count) as margin_per_shipment,
(sum(gross_revenue)-sum(total_cost))*100/sum(gross_revenue) as margin_percentage
from
(select * from data_analytics.bi_heavy_clients_revenue_and_pnl_data
where ad >= (select max(ad) from data_analytics.bi_heavy_clients_revenue_and_pnl_data)
)
group by 1,2
order by 1,2
""")

#  B2C Heavy monthly overall volume, revenue, cost breakout, total cost, margin for a client
vn.train(
question="What is the monthly volume, revenue, total costs, and margin percentage for client 'AMAZONINDIA'?"
,sql="""
select  month, year, client,
sum(mwn_count) as master_waybill_count, sum(wbn_count) as waybill_count, 
sum(charged_weight)/sum(wbn_count) as avg_charged_weight, sum(gross_revenue) as gross_revenue, 
sum(gross_revenue)/sum(wbn_count) as gross_yield, 
 sum(total_cost) as total_cost,
sum(total_cost)/sum(wbn_count) as cost_per_shipment,
(sum(gross_revenue)-sum(total_cost))/sum(wbn_count) as margin_per_shipment,
(sum(gross_revenue)-sum(total_cost))*100/sum(gross_revenue) as margin_percentage
from
(select * from data_analytics.bi_heavy_clients_revenue_and_pnl_data
where ad >= (select max(ad) from data_analytics.bi_heavy_clients_revenue_and_pnl_data)
and lower(client) = lower('AMAZONINDIA')
)
group by 1,2,3
order by 1,2
""")

# 2. volume, revenue, cost breakout, total cost, margin for a client for particular month
vn.train(
question="What is the overall shipment volume, revenue, cost factors, total costs, and margin percentage for client 'FLIPKART - E2E SURFACE' for February 2025?"
,sql="""
select client, month, year, 
sum(mwn_count) as master_waybill_count, sum(wbn_count) as waybill_count, 
sum(charged_weight)/sum(wbn_count) as avg_charged_weight, sum(gross_revenue) as gross_revenue, 
sum(gross_revenue)/sum(wbn_count) as gross_yield, 
sum(fm_fleet_cost) as fm_fleet_cost, sum(fm_mp_cost) as fm_mp_cost, sum(fm_bp_cost) as fm_bp_cost, 
sum(fm_constellation_cost) as fm_constellation_cost,
sum(lm_fleet_cost) as lm_fleet_cost, sum(lm_mp_cost) as lm_mp_cost, sum(lm_bp_cost) as lm_bp_cost, 
sum(lm_constellation_cost) as lm_constellation_cost,
sum(rt_fleet_cost) as rt_fleet_cost, sum(rt_mp_cost) as rt_mp_cost, sum(rt_bp_cost) as rt_bp_cost, 
sum(rt_constellation_cost) as rt_constellation_cost,
sum(ipc_incenter_cost) as ipc_incenter_cost, sum(sc_incenter_cost) as sc_incenter_cost, 
sum(hub_incenter_cost) as hub_incenter_cost,
sum(line_haul_intra_cost) as line_haul_intra_cost, sum(line_haul_inter_cost) as line_haul_inter_cost,
sum(lma_cost) as lma_cost, sum(biker_fuel_cost) as biker_fuel_cost,
sum(consumables_cost) as consumables_cost, sum(other_opex_cost) as other_opex_cost, sum(aws_cost) as aws_cost,
sum(cms_cost) as cms_cost, sum(loss_damage_cost) as loss_damage_cost, sum(total_cost) as total_cost,
sum(total_cost)/sum(wbn_count) as cost_per_shipment,
(sum(gross_revenue)-sum(total_cost))/sum(wbn_count) as margin_per_shipment,
(sum(gross_revenue)-sum(total_cost))*100/sum(gross_revenue) as margin_percentage
from
(select * from data_analytics.bi_heavy_clients_revenue_and_pnl_data
where ad >= (select max(ad) from data_analytics.bi_heavy_clients_revenue_and_pnl_data)
and lower(client) = lower('FLIPKART - E2E SURFACE')
and month = 2 and year = 2025)
group by 1,2,3
order by 5 desc
""")

# 3. volume, revenue, cost breakout, total cost, margin for a client at zone level for particular month
vn.train(
question="How did zones affect or contribute to the revenue, cost factors, total costs, and margin percentage for client 'AMAZONCRETURNS' for February 2025?"
,sql="""
select  month, year, client, zone,
sum(mwn_count) as master_waybill_count, sum(wbn_count) as waybill_count, 
sum(charged_weight)/sum(wbn_count) as avg_charged_weight, sum(gross_revenue) as gross_revenue, 
sum(gross_revenue)/sum(wbn_count) as gross_yield, 
sum(fm_fleet_cost) as fm_fleet_cost, sum(fm_mp_cost) as fm_mp_cost, sum(fm_bp_cost) as fm_bp_cost, 
sum(fm_constellation_cost) as fm_constellation_cost,
sum(lm_fleet_cost) as lm_fleet_cost, sum(lm_mp_cost) as lm_mp_cost, sum(lm_bp_cost) as lm_bp_cost, 
sum(lm_constellation_cost) as lm_constellation_cost,
sum(rt_fleet_cost) as rt_fleet_cost, sum(rt_mp_cost) as rt_mp_cost, sum(rt_bp_cost) as rt_bp_cost, 
sum(rt_constellation_cost) as rt_constellation_cost,
sum(ipc_incenter_cost) as ipc_incenter_cost, sum(sc_incenter_cost) as sc_incenter_cost, 
sum(hub_incenter_cost) as hub_incenter_cost,
sum(line_haul_intra_cost) as line_haul_intra_cost, sum(line_haul_inter_cost) as line_haul_inter_cost,
sum(lma_cost) as lma_cost, sum(biker_fuel_cost) as biker_fuel_cost,
sum(consumables_cost) as consumables_cost, sum(other_opex_cost) as other_opex_cost, sum(aws_cost) as aws_cost,
sum(cms_cost) as cms_cost, sum(loss_damage_cost) as loss_damage_cost, sum(total_cost) as total_cost,
sum(total_cost)/sum(wbn_count) as cost_per_shipment,
(sum(gross_revenue)-sum(total_cost))/sum(wbn_count) as margin_per_shipment,
(sum(gross_revenue)-sum(total_cost))*100/sum(gross_revenue) as margin_percentage
from
(select * from data_analytics.bi_heavy_clients_revenue_and_pnl_data
where ad >= (select max(ad) from data_analytics.bi_heavy_clients_revenue_and_pnl_data)
and lower(client) = lower('AMAZONCRETURNS')
and month = 2 and year = 2025)
group by 1,2,3,4
order by 6 desc
""")

# 4. volume, revenue, cost breakout, total cost, margin for a client at shipment status level for particular month
vn.train(
question="How did shipment status affect or contribute to the revenue, cost factors, total costs, and margin percentage for client 'FLIPKART - E2E SURFACE' for March 2025?"
,sql="""
select  month, year, client, scan_status,
sum(mwn_count) as master_waybill_count, sum(wbn_count) as waybill_count, 
sum(charged_weight)/sum(wbn_count) as avg_charged_weight, sum(gross_revenue) as gross_revenue, 
sum(gross_revenue)/sum(wbn_count) as gross_yield, 
sum(fm_fleet_cost) as fm_fleet_cost, sum(fm_mp_cost) as fm_mp_cost, sum(fm_bp_cost) as fm_bp_cost, 
sum(fm_constellation_cost) as fm_constellation_cost,
sum(lm_fleet_cost) as lm_fleet_cost, sum(lm_mp_cost) as lm_mp_cost, sum(lm_bp_cost) as lm_bp_cost, 
sum(lm_constellation_cost) as lm_constellation_cost,
sum(rt_fleet_cost) as rt_fleet_cost, sum(rt_mp_cost) as rt_mp_cost, sum(rt_bp_cost) as rt_bp_cost, 
sum(rt_constellation_cost) as rt_constellation_cost,
sum(ipc_incenter_cost) as ipc_incenter_cost, sum(sc_incenter_cost) as sc_incenter_cost, 
sum(hub_incenter_cost) as hub_incenter_cost,
sum(line_haul_intra_cost) as line_haul_intra_cost, sum(line_haul_inter_cost) as line_haul_inter_cost,
sum(lma_cost) as lma_cost, sum(biker_fuel_cost) as biker_fuel_cost,
sum(consumables_cost) as consumables_cost, sum(other_opex_cost) as other_opex_cost, sum(aws_cost) as aws_cost,
sum(cms_cost) as cms_cost, sum(loss_damage_cost) as loss_damage_cost, sum(total_cost) as total_cost,
sum(total_cost)/sum(wbn_count) as cost_per_shipment,
(sum(gross_revenue)-sum(total_cost))/sum(wbn_count) as margin_per_shipment,
(sum(gross_revenue)-sum(total_cost))*100/sum(gross_revenue) as margin_percentage
from
(select * from data_analytics.bi_heavy_clients_revenue_and_pnl_data
where ad >= (select max(ad) from data_analytics.bi_heavy_clients_revenue_and_pnl_data)
and lower(client) = lower('FLIPKART - E2E SURFACE')
and month = 3 and year = 2025)
group by 1,2,3,4
order by 6 desc
""")

# 5. volume, revenue, cost breakout, total cost, margin for top 10 clients by margin for particular month
vn.train(
question="Which are the top 10 clients by margin percentage for B2C Heavy shipments for April 2025?"
,sql="""
select client, month, year, 
sum(mwn_count) as master_waybill_count, sum(wbn_count) as waybill_count, 
sum(charged_weight)/sum(wbn_count) as avg_charged_weight, sum(gross_revenue) as gross_revenue, 
sum(gross_revenue)/sum(wbn_count) as gross_yield, 
sum(fm_fleet_cost) as fm_fleet_cost, sum(fm_mp_cost) as fm_mp_cost, sum(fm_bp_cost) as fm_bp_cost, 
sum(fm_constellation_cost) as fm_constellation_cost,
sum(lm_fleet_cost) as lm_fleet_cost, sum(lm_mp_cost) as lm_mp_cost, sum(lm_bp_cost) as lm_bp_cost, 
sum(lm_constellation_cost) as lm_constellation_cost,
sum(rt_fleet_cost) as rt_fleet_cost, sum(rt_mp_cost) as rt_mp_cost, sum(rt_bp_cost) as rt_bp_cost, 
sum(rt_constellation_cost) as rt_constellation_cost,
sum(ipc_incenter_cost) as ipc_incenter_cost, sum(sc_incenter_cost) as sc_incenter_cost, 
sum(hub_incenter_cost) as hub_incenter_cost,
sum(line_haul_intra_cost) as line_haul_intra_cost, sum(line_haul_inter_cost) as line_haul_inter_cost,
sum(lma_cost) as lma_cost, sum(biker_fuel_cost) as biker_fuel_cost,
sum(consumables_cost) as consumables_cost, sum(other_opex_cost) as other_opex_cost, sum(aws_cost) as aws_cost,
sum(cms_cost) as cms_cost, sum(loss_damage_cost) as loss_damage_cost, sum(total_cost) as total_cost,
sum(total_cost)/sum(wbn_count) as cost_per_shipment,
(sum(gross_revenue)-sum(total_cost))/sum(wbn_count) as margin_per_shipment,
(sum(gross_revenue)-sum(total_cost))*100/sum(gross_revenue) as margin_percentage
from
(select * from data_analytics.bi_heavy_clients_revenue_and_pnl_data
where ad >= (select max(ad) from data_analytics.bi_heavy_clients_revenue_and_pnl_data)
--and lower(client) = lower('FLIPKART - E2E SURFACE')
and month = 4 and year = 2025)
group by 1,2,3
having sum(wbn_count) > 500
order by margin_percentage desc
limit 10
""")


import chainlit as cl
import requests 


@cl.oauth_callback
def oauth_callback( provider_id: str,
  token: str,
  raw_user_data: dict,
  default_user: cl.User,) -> cl.User:
    
    print(token)
    print("user data from Keycloak:", raw_user_data)
    # print(provider_id)
    permission_url = "https://api-stage-ums.delhivery.com/v2/user/has_perm/Financequerybuilder.can_query_finance_data"
    headers = {
        "Authorization": f"Bearer {token}",
        "X-App-Id": "1433",
    }

    try:
        response = requests.get(permission_url, headers=headers)
    
        if response.status_code == 200:
            print("User has permission. Access granted.")
            
            return default_user

        else:
            print("User does not have permission. Access denied.")

            return None
            # raise cl.AuthenticationError("403 Forbidden: You do not have the required permission.")
        
    except requests.exceptions.RequestException as e:
        print(f"Error calling permission API: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None
    
# @cl.on_stop 
# async def on_stop(): 
#     # Cancel ongoing DB query / LLM call 
#     print("⚠️ Task cancelled by user")
#     await cl.Message(content="⚠️ Task cancelled by user.").send()

@cl.on_message
async def main(message: cl.Message):

    my_question = message.content.lower().strip()

    # print(my_question)

    greetings = ["hi", "hello", "hey", "how are you", "what's up", "good morning", "good evening"]

    sql_related_keywords = ["average", "gross yield", "yield", "month over month", "mom change", "shipment", "Volume", "gross", "revenue", "weight bucket", "zone", 
                "client", "client mapping", "b2c", 'heavy',"segment", "current", "last", "affected", "reasons", "shipment status", "mode of transport",
                "top 10", "contributing"]
    
    
    if any(sql_related in my_question for sql_related in sql_related_keywords):

        await cl.Message(content=f"🔍 Question: **{my_question}**").send()
        
        # Generate SQL from question
        sql = vn.generate_sql(my_question, model=openai_model)
        await cl.Message(content="Generating the SQL code for you...").send()
        await cl.Message(content=f"```sql\n{sql}\n```").send()

        sql = sql.replace(";","")
        # Run SQL
        progress_msg = await cl.Message(content="🔄 Running the SQL code on Presto, hold tight!").send()
        

        # df = run_sql(sql)

        async def safe_run_sql(sql, row_limit=5000, timeout_seconds=100):
            try:
                # Add LIMIT to prevent memory issues with large datasets
                if "limit" not in sql.lower() and "top" not in sql.lower():
                    sql = f"{sql} LIMIT {row_limit}"
                
                # Run SQL query with timeout in a separate thread
                loop = asyncio.get_event_loop()
                with ThreadPoolExecutor() as executor:
                    try:
                        sql_df = await asyncio.wait_for(
                            loop.run_in_executor(executor, run_sql, sql),
                            timeout=timeout_seconds
                        )
                    except asyncio.TimeoutError:
                        await cl.Message(content=f"⏰ Query timed out after {timeout_seconds} seconds. Try a more specific query.").send()
                        return None
                
                # Additional safety check for DataFrame size
                if sql_df is not None and len(sql_df) > row_limit:
                    await cl.Message(content=f"⚠️ Query returned {len(sql_df)} rows. Using first {row_limit} for performance.").send()
                    sql_df = sql_df.head(row_limit)
                
                return sql_df
            except Exception as e:
                print(f"Query failed to run : {e}")
                return None

        df = await safe_run_sql(sql)

        # Query result
        if df is None or df.empty:
            await cl.Message(content="❌ Query failed to run: invalid SQL or column not found.").send()
            
        else:
            await cl.Message(content="✅ Query ran successfully!").send()

        print(f'df output shape: {df.shape}')

        # handle data size issue take only first 1000 rows if query send larger output data 
        ANALYSIS_ROW_LIMIT = 1000
        DISPLAY_ROW_LIMIT = 500  # Even smaller limit for UI display
        
        df_for_analysis = df.head(ANALYSIS_ROW_LIMIT) if len(df) > ANALYSIS_ROW_LIMIT else df
        df_for_display = df.head(DISPLAY_ROW_LIMIT) if len(df) > DISPLAY_ROW_LIMIT else df
        
        if len(df) > ANALYSIS_ROW_LIMIT:
            await cl.Message(
                content=f"⚠️ The query returned **{len(df)}** rows. "
                        f"For visualization and analysis, I will use the first **{ANALYSIS_ROW_LIMIT}** rows. "
                        f"Displaying first **{DISPLAY_ROW_LIMIT}** rows in the table below."
            ).send()

       
        elements = [cl.Dataframe(data=df_for_display, display="inline", name="Dataframe")]
        await cl.Message(content="📊 Here are the results of the query: ", elements=elements).send()
       
        
        ### Provide CSV download of full data
        import os
        csv_name = "sql_query_result.csv"
        csv_path = os.path.join(os.getcwd(), csv_name)

        df.to_csv(csv_path, index=False)
        download_button = cl.File(name=csv_name, path=csv_path, display="inline")
        await cl.Message(content="📥 Download the complete table as a CSV file:", elements=[download_button]).send()

        #### Generate plotly chart   
        # option 1   
        plotly_code = vn.generate_plotly_code(my_question, sql, df_for_analysis)
        fig = vn.get_plotly_figure(plotly_code, df_for_analysis)
        elements = [cl.Plotly(name="chart", figure=fig, display="inline")]
        await cl.Message(content="Results Chart:", elements=elements).send()



        # Limit data for summary to prevent token limits and memory issues
        df_for_summary = df_for_analysis
        
        #### Convert to JSON and check size
        # df_summary = df_for_summary.to_json(orient="records", indent=2)
        df_summary = df_for_summary.to_csv(index=False)
        # df_summary = df_for_summary.describe().to_dict()
   
        # else:
        if len(df_for_analysis) > ANALYSIS_ROW_LIMIT:
            await cl.Message(content=f"📊 **Large Dataset**: {len(df_for_analysis)} total records. Using first {ANALYSIS_ROW_LIMIT} for analysis.").send()
      

        # Two-step insights generation
        try:
            # Step 1: Generate tailored analysis prompt
            # await cl.Message(content="🔄 Generating tailored analysis prompt...").send()

            custom_prompt_msg = cl.Message(content="🔄 Generating prompt for **customized insights**")
            await custom_prompt_msg.send()
            
            data_context = {
                "data_shape": df_for_analysis.shape,
                "columns": list(df_for_analysis.columns),
                "sample_data": df_for_analysis.head(5).to_dict('records') if len(df_for_analysis) > 0 else {}
            }

            prompt_generation_request = f"""
            You are an expert logistics finance analyst. Your job is to generate a **single-line analysis prompt** (not multi-step) that will guide an LLM to extract precise insights from logistics financial data.

            # Inputs
            - User Question: "{my_question}"
            - Data Context: "{data_context}"

            # Rules for Prompt generation:
            1. The output must be a **prompt only** — no data points.
            2. Explicitly tie the analysis to the provided columns and context.
            3. If the question is about **yield or cost monthly trends (monthly/overall/client-level)**:
            - direct the LLM to calculate yield = revenue / shipment_count and compare across months.
            4. Factors driving gross yield change / "why did this change happen"
                - Treat each affecting factor as an **independent partition** of the month (e.g., all `bill_zone` rows for a month sum to the month's totals; same for `weight_bucket_group`, `scan_status`, etc.).  
                - If the user **specifies a factor** (e.g., `bill_zone`) → analyze that factor only across months (compare its factor_values month→month).  
                - If the user **does not** specify a factor → analyze each factor **separately** at the month level:
                    - `weight_bucket_group`
                    - `bill_zone`
                    - `scan_status`
                    - `package_type`
                    - `mode_of_transport`
                - To attribute gross yield change from month M1 → M2 for a given factor_value, decompose the gross yield change into **volume change effect** and **yield change effect** in each bucket and rank factor_values by their contribution to the overall change.
                
                            
            5. Be concise, but keep it specific enough to guide the analysis.

            # Output Requirement
            Return only the generated analysis prompt, no commentary, no extra text.
            """

            # Option 1 - Generate the tailored prompt
            # prompt_response = model.generate_content(prompt_generation_request)
            # generated_analysis_prompt = prompt_response.text if prompt_response else None

            # Option 2 
            async def custom_gemini_prompt():

                def sync_call():
                    # prompt_response = model.generate_content(prompt_generation_request, stream=True)
                    text = ""
                    # for chunk in prompt_response:
                    #     if chunk.text:
                    #         text += chunk.text
                    # return text
                
                    ### Azure OpenAI LLM
                    response = client.chat.completions.create(
                            model=openai_model,  # your deployment name
                            messages=[
                                {"role": "user", "content": prompt_generation_request}] )
                    
                    text = response.choices[0].message.content

                    return text


                return await asyncio.to_thread(sync_call)

            # Start Gemini in background
            cust_task = asyncio.create_task(custom_gemini_prompt())

            # Optional: keep frontend alive with "progress pings"
            dots = ""
            while not cust_task.done():
                await asyncio.sleep(2)   # ping every 2s (tune this interval)
                # await msg.update() 
                dots = (dots + ".") if len(dots) < 4 else ""
                custom_prompt_msg.content = f"🔄 Generating prompt for **customized insights**\n\nGenerating{dots}"
                await custom_prompt_msg.update()
            
            generated_analysis_prompt = await cust_task

            # progress_msg = cl.Message(content=f"✨ **Enhanced Gemini Analysis:**\n\n{result}")
            # custom_prompt_msg.content = f"✨ **Custom prompt generated:**\n\n{generated_analysis_prompt}"
            # await custom_prompt_msg.update()

            
            if generated_analysis_prompt:
                await cl.Message(content="✅ **Customized Prompt Generated :**").send()
                await cl.Message(content=f"```\n{generated_analysis_prompt}\n```").send()
                
                # Step 2: Use the generated prompt for analysis
                final_analysis_prompt = f"""
                {generated_analysis_prompt}
                
                # Context
                - Data: {df_summary}

                Be concise, numeric, and decisive. Do not add commentary or fluff but add needed explaination
                Don't show any **change effect formula** in the response to keep the insights concise and meaningful
                and keep only summary of the question asked within 300 words and before responding, please double check that aggregated numbers adds up to the given context data.
                """
                print(f'### Using LLM Generated Prompt:')
                
            else:
                # Fallback to original prompt if generation fails
                await cl.Message(content="⚠️ Using fallback analysis prompt...").send()
                final_analysis_prompt = f"""
                You are a Senior Logistics Finance Analyst for India. Answer the user's question using ONLY the provided data.

                # Inputs
                - Question: {my_question}
                - Data : {df_summary}

                # Definitions
                - Yield = Gross_Amount / wbn_count (revenue per shipment).
                - Currency = INR.
                - Closure_month/Closure_year = time periods (compare MoM).

                # Output Requirements
                - ## Final Answer: Show precise response with supporting data relevant to asked question.

                Be concise, numeric, and decisive. Do not add commentary or fluff but add needed explaination
                Don't show any **change effect formula** in the response to keep the insights concise and meaningful
                and keep only summary of the question asked within 300 words and before responding, please double check that aggregated numbers adds up to the given context data.
                """
                print(f'### Using Original Fallback Prompt:')

            print(f'###### Final Analysis Prompt: \n{final_analysis_prompt} ')
            print(f'###### Gemini Prompt length: {len(final_analysis_prompt)} characters')
            print(f'###### Data summary length: {len(str(df_summary))} characters')
            
            
            # Add timeout handling
            import time
            start_time = time.time()
            
            await cl.Message(content="🤖 Generating enhanced AI analysis...").send()

            
            progress_msg = cl.Message(content="✨ **Enhanced Gemini Analysis:**\n\nThinking")
            await progress_msg.send()

            #### Option 1
            # response = model.generate_content(final_analysis_prompt, stream=True)
            # full_response = ""
            # for chunk in response:
            #     if chunk.text:
            #         await msg.stream_token(chunk.text)
            #         full_response += chunk.text
            # await msg.update()

            # Option2 - Paralley task to send a message in UI and keep UI active while gemini responds
            async def run_gemini():

                def sync_call():
                    # response = model.generate_content(final_analysis_prompt, stream=True)
                    text = ""
                    # for chunk in response:
                    #     if chunk.text:
                    #         text += chunk.text
                    # return text
                
                    #### Open AI LLM 

                    response = client.chat.completions.create(
                            model=openai_model,  # your deployment name
                            messages=[
                                {"role": "user", "content": final_analysis_prompt}] )
                    
                    text = response.choices[0].message.content

                    return text

                return await asyncio.to_thread(sync_call)

            # Start Gemini in background
            task = asyncio.create_task(run_gemini())

            # Optional: keep frontend alive with "progress pings"
            dots = ""
            while not task.done():
                await asyncio.sleep(2)   # ping every 2s (tune this interval)
                # await msg.update() 
                dots = (dots + ".") if len(dots) < 4 else ""
                progress_msg.content = f"✨ **Enhanced Gemini Analysis:**\n\nThinking{dots}"
                await progress_msg.update()
            
            result = await task

            # progress_msg = cl.Message(content=f"✨ **Enhanced Gemini Analysis:**\n\n{result}")
            progress_msg.content = f"✨ **Enhanced Gemini Analysis:**\n\n{result}"
            await progress_msg.update()

            print(f'### Printing Full Response Text:\n{result}')

            end_time = time.time()
            print(f'###### Gemini response time: {end_time - start_time:.2f} seconds')
            
           
        except Exception as e:
            error_msg = str(e)
            print(f"Gemini error: {error_msg}")
            
            if "quota" in error_msg.lower() or "limit" in error_msg.lower():
                await cl.Message(content="⚠️ **API Quota Exceeded** - Please try again later or use a smaller dataset.").send()
            elif "timeout" in error_msg.lower():
                await cl.Message(content="⏱️ **Request Timed Out** - Dataset too large. Try asking for specific metrics.").send()
            else:
                await cl.Message(content=f"⚠️ Could not generate summary: {error_msg}").send()
            response = None

        #Log entry
        try:
            from datetime import datetime
            import pytz 
            ist = pytz.timezone('Asia/Kolkata')
            timestamp = datetime.now(ist).isoformat()
            gemini_summary = result if result else "Summary generation failed"
            log_entry = { "timestamp": timestamp, "user_message": my_question, "gemini_response": gemini_summary}

            # Check if log file exists, create if not
            import os
            if os.path.exists('gemini_responses.csv'):
                df_log = pd.read_csv('gemini_responses.csv')
            else:
                df_log = pd.DataFrame(columns=["timestamp", "user_message", "gemini_response"])
            
            df_log = pd.concat([df_log, pd.DataFrame([log_entry])], ignore_index=True)
            df_log.to_csv('gemini_responses.csv', index=False)
        except Exception as e:
            print(f"Logging failed: {e}")  # Silent fail for logging

        # Prompt user for next question
        await cl.Message(content="🔄 You can ask another question anytime!").send()

    else:
        try:
            # response = model.generate_content(my_question)
            # await cl.Message(content=response.text).send()

            ### Openai LLM 
            response = client.chat.completions.create(
                        model=openai_model,
                          messages=[{"role": "user", "content": my_question}])
            
            answer_text = response.choices[0].message.content
            await cl.Message(content=answer_text).send()

        except Exception as e:
            await cl.Message(content=f"⚠️ Sorry, I couldn't process your request: {str(e)}").send()
        return
    