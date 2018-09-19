# coding=utf-8

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns


train_df = pd.read_csv(r'C:\Users\msi\Desktop\Zillow\train_2016_v2.csv',parse_dates = ["transactiondate"])
#对于不规范的日期，在read_csv之后用to_datetime()来转换
'''
print(train_df.head())
print(train_df.info())
print(train_df.isnull().sum())
'''

plt.figure(figsize=(15,10))
plt.scatter(train_df.index,train_df['logerror'])
plt.xlabel('index',fontsize=10)
plt.ylabel('logerror',fontsize = 10)
plt.title('Logerror_Distribution',fontsize = 20)
#上面有几个outliers，应该去掉


ulimit = np.percentile(train_df.logerror.values,99)
llimit = np.percentile(train_df.logerror.values,1)
train_df.loc[train_df['logerror'] > ulimit,'logerror'] = ulimit 
train_df.loc[train_df['logerror'] < llimit,'logerror'] = llimit

plt.figure(figsize = (12,8))
sns.distplot(train_df.logerror.values, bins = 50, kde = False)
plt.xlabel('logerror',fontsize = 12)
#nice normal distribution on the log error

f,ax1 = plt.subplots()
train_df['transaction_month'] = train_df['transactiondate'].dt.month
train_df['transaction_month'].value_counts().plot.bar(ax = ax1)
ax1.set_xlabel('YearMonth',fontsize = 16)
ax1.set_ylabel('Occurance',fontsize = 16)
ax1.set_title('Occur_Month_Distribution',fontsize = 20)


print((train_df['parcelid'].value_counts().reset_index())['parcelid'].value_counts())	#第一步计数之后获得 id:counter的表，reset_index()之后两者调换
#有重复出现的id 但是大多数(99%)都是只出现一次的

prop_df = pd.read_csv(r'C:\Users\msi\Desktop\Zillow\properties_2016.csv')
#print(prop_df.head())

#查看其缺失值bar plot
f,ax2 = plt.subplots()
prop_df.isnull().sum().plot.bar(ax = ax2)
ax2.set_title('Feature_Missing',fontsize = 16)
#缺失值很大且缺失属性很多

#Latitude and Longitude
































plt.show()


































































