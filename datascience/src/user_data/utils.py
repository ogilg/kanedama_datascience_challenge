import pandas as pd
import os

def get_month(date):
    return date.astype('datetime64[M]').astype(int) % 12 + 1

business_naf = pd.read_csv(os.path.join('datascience', 'data', 'business_NAF.csv'),sep=';')

def get_job_type(naf_code):
    try:
        job_name = business_naf.loc[business_naf['code'] == naf_code]['name'].values[0]
        return job_name.replace(',',' ').split()[0]
    except:
        return 'Unknown'
