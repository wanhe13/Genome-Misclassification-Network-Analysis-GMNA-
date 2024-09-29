#!/usr/bin/env python
# coding: utf-8


import pandas as pd


tsv_read = pd.read_csv("data/metadata_tsv_2021_06_03/metadata.tsv", sep='\t')

ID2date=dict(zip(list(tsv_read['Accession ID']),list(tsv_read['Collection date'])))
print(len(set(list(tsv_read['Accession ID'])))==len(list(tsv_read['Accession ID'])))
print(len(set(list(tsv_read['Virus name']))))
print(len(list(tsv_read['Virus name'])))

Name_list=[]
for i in list(tsv_read['Virus name']):
    cov,region,strain,year=i.split('/',3)
    Name_list.append(strain)

Name2date=dict(zip(Name_list,list(tsv_read['Collection date'])))
print(len(Name2date))
print(len(list(set(Name_list))))


df_flights = pd.read_csv('data/OAG_country-flight-network_201912-202103.csv',header=0)
print(df_flights)
print(len(list(set(df_flights['Dep Country Code']))))

input_path="sequences.fasta"
output_path = "data/data.txt"


df=pd.read_table('complete_original_country.txt')

index2Name=dict(zip(list(range(len(df))),list(df['strain'])))

print(len(df))

# Get a list of collection dates correponds to genome name when availble
Date=[]
for name in list(df['strain']):
    try:
        Date.append(Name2date[name])
    except:
        Date.append('NA')
        

print('length of date info',len(Date[8]))


#Get a list of indices where the collection date corresponding to the genome name is not available or not in the right form
#i.e. 2020
row_delete=[i for i in range(len(Date)) if len(Date[i])!=10]

print(len(Date),len(row_delete))

df.insert(5, "Date", Date, True)
df_filtered_date=df.drop(row_delete)



#check there is no more 'NA's
'NA' in list(df_filtered_date['Date'])


df_filtered_date=df_filtered_date.sort_values(by=['Date'])


def date_conversion(date):
    month=(int(date[:4])-2019)*12-11+int(date[5:7])
    return month




Date_list=list(df_filtered_date['Date'])
Month_list=[]
for i in Date_list:
    Month_list.append(date_conversion(i))
    



df_filtered_date.insert(6, "Month", Month_list, True)
df_filtered_date.to_csv('data/df_date.csv',index=False)
