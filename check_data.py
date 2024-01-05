import pandas as pd

df_all = pd.read_csv('./data/alldata.csv')
df_relation1 = pd.read_csv('./data/relation1.csv')
df_relation2 = pd.read_csv('./data/relation2.csv')

for index, row in df_relation1.iterrows():
    api_id = row['API_ID']
    tag_id = row['TAG_ID']
