import json
import pandas as pd
import numpy as np

# input
api_dataset_path = './data/apilist-details.txt'
mashup_dataset_path = './data/mashup-details.txt'


df_API = pd.DataFrame(columns=['API', 'description'])
df_MASHUP = pd.DataFrame(columns=['MASHUP', 'description'])
df_TAG = pd.DataFrame(columns=['TAG'])

df_ALL = pd.DataFrame(columns=['NAME', 'TYPE', 'description', 'y'])

df_relation = pd.DataFrame(columns=['API_ID', 'TAG_ID'])
df_relation2 = pd.DataFrame(columns=['MASHUP_ID', 'API_ID'])

# SET
dic_tags = {}
dic_apis = {}
dic_ctag = {}

sorted_apis = []

tag_dsc = {}

with open('./data/category-details.txt', 'r', encoding='utf-8') as f:
    j = json.load(f)
    for item in j:
        val = item[1]
        if val == None or val == '':
            val = item[0]
        tag_dsc[item[0]] = val

# API的文件
dic_ctag['0'] = 0
index_ctag = 1
with open(api_dataset_path, 'r') as f:
    data = f.read()
    jsonarry = json.loads(data)
    data_len = len(jsonarry)
    print(len(jsonarry))

    for api in jsonarry:
        try:
            category = api['versions'][0]['details']['Primary Category']
        except:
            try:
                category = api['versions'][0]['details']['Related Categories: '].split(',')[0].strip()
            except:
                category = '0'
                print('null')
        if dic_ctag.get(category) == None:
            dic_ctag[category] = index_ctag
            index_ctag = index_ctag + 1

with open(api_dataset_path, 'r') as f:
    data = f.read()
    jsonarry = json.loads(data)
    data_len = len(jsonarry)
    print(len(jsonarry))

    for api in jsonarry:
        title = api['title'].replace('MASTER RECORD', '')
        tags = api['tags']
        description = api['description']
        url = api['url'].replace('\n', '')
        try:
            category = api['versions'][0]['details']['Primary Category']
        except:
            try:
                category = api['versions'][0]['details']['Related Categories: '].split(',')[0].strip()
            except:
                category = '0'
        if len(description) == 0:
            description = ' '.join(title.split('-'))
        df_ALL = df_ALL.append({'NAME':url, 'TYPE': 'API', 'description':description, 'y': category}, ignore_index=True)
        for tag in tags:
            if dic_tags.get(tag) == None:
                dic_tags[tag] = 1
            else:
                dic_tags[tag] = dic_tags[tag] + 1

for key in dic_tags.keys():
    df_ALL = df_ALL.append({'NAME': key, 'TYPE': 'TAG', 'description': tag_dsc.get(key, key), 'y': ''}, ignore_index=True)


# MASHUP 文件
with open(mashup_dataset_path, 'r') as f:
    data = f.read()
    jsonarry = json.loads(data)
    data_len = len(jsonarry)
    print(len(jsonarry))
    print(jsonarry[0])

    for amashup in jsonarry:
        title = amashup['title']
        tags = amashup['tags']
        related_apis = amashup['related_apis']
        description = amashup['description']
        df_ALL = df_ALL.append({'NAME':title, 'TYPE': 'MASHUP', 'description':description, 'y': ''}, ignore_index=True)
        for api in related_apis:
            if dic_apis.get(api) == None:
                dic_apis[api] = 1
            else:
                dic_apis[api] = dic_apis[api] + 1

for key, val in dic_apis.items():
    sorted_apis.append((key, val))

def takeSecond(elem):
    return elem[1]

sorted_apis.sort(key=takeSecond, reverse=True)
df_ALL.to_csv('./data/alldata.csv', index=0)

print(df_ALL[df_ALL['NAME']=='Mashup: Iraq Coalition'].index.tolist())

with open(api_dataset_path, 'r') as f:
    data = f.read()
    jsonarry = json.loads(data)
    data_len = len(jsonarry)
    print(len(jsonarry))

    for api in jsonarry:
        title = api['title']
        tags = api['tags']
        description = api['description']
        url = api['url'].replace('\n', '')
        api_ids = df_ALL[df_ALL['NAME'] == url].index.tolist()[0]
        for tag in tags:
            tag_id = df_ALL[df_ALL['NAME']==tag].index.tolist()[0]
            df_relation = df_relation.append({'API_ID': api_ids, 'TAG_ID': tag_id, 'API_NAME':url, 'TAG_NAME':tag}, ignore_index=True)

with open(mashup_dataset_path, 'r') as f:
    data = f.read()
    jsonarry = json.loads(data)
    data_len = len(jsonarry)
    print(len(jsonarry))
    print(jsonarry[0])

    for amashup in jsonarry:
        title = amashup['title']
        tags = amashup['tags']
        related_apis = amashup['related_apis']
        categories =amashup['categories']
        mashup_id = df_ALL[df_ALL['NAME'] == title].index.tolist()[0]
        mashup_name = df_ALL[df_ALL['NAME'] == title]['NAME'].tolist()[0]
        mashup_description = df_ALL[df_ALL['NAME'] == title]['description'].tolist()[0]

        # mashup也写入relation
        # tag in tags:
        #    index = df_ALL[df_ALL['NAME'] == tag].index
        #    if len(index) == 0:
        #        continue
        #    tag_id = index.tolist()[0]
        #    df_relation = df_relation.append({'API_ID': mashup_id, 'TAG_ID': tag_id, 'API_NAME':mashup_name, 'TAG_NAME':tag}, ignore_index=True)
        ##

        all_id = ''
        all_names = ''
        all_id_array = []
        all_name_array = []
        category_list = []
        for api in related_apis:
            api = api.replace('\n', '')
            api_ids = df_ALL[df_ALL['NAME'] == api].index.tolist()
            all_name_array.append(api)
            if len(api_ids)==0:
                continue
            for one_api in sorted_apis:
                if one_api[0] == api:
                    all_id_array.append((api_ids[0], one_api[1]))
        all_id_array.sort(key=takeSecond, reverse=True)

        for category in categories:
            category_list.append(category)

        for one_api in all_id_array:
            all_id = all_id + str(one_api[0]) + "\t"

        for one_api_name in all_name_array:
            all_names = all_names + one_api_name + "\t"

        df_relation2 = df_relation2.append({'MASHUP_ID':mashup_id, 'API_ID':all_id, 'MASHUP_NAME': mashup_name, 'MASHUP_DESC':mashup_description, 'API_NAME': all_names, 'CATEGORY': (' ').join(category_list)}, ignore_index=True)

df_relation.to_csv('./data/relation1.csv', index=0)
df_relation2.to_csv('./data/relation2.csv', index=0)
print('')