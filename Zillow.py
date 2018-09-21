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
plt.scatter(train_df.index,np.sort(train_df['logerror'].values))
plt.xlabel('index',fontsize=10)
plt.ylabel('logerror',fontsize = 10)
plt.title('Logerror_Distribution',fontsize = 20)
#上面有几个outliers，应该修改


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
#事件似乎在11,12月发生得很少，年中很多

#print((train_df['parcelid'].value_counts().reset_index())['parcelid'].value_counts())	#第一步计数之后获得 id:counter的表，reset_index()之后两者调换
#有重复出现的id 但是大多数(99%)都是只出现一次的

prop_df = pd.read_csv(r'C:\Users\msi\Desktop\Zillow\properties_2016.csv')
#print(prop_df.head())

#查看其缺失值bar plot
f,ax2 = plt.subplots()
prop_df.isnull().sum().plot.bar(ax = ax2)
ax2.set_title('Feature_Missing',fontsize = 16)
#缺失值很大且缺失属性很多

#Latitude and Longitude
f,ax3 = plt.subplots(figsize = (20,20))
plt.scatter(x = 'latitude',y = 'longitude',data = prop_df)
plt.xlabel('latitude',fontsize = 12)
plt.ylabel('longitude',fontsize = 12)
plt.title('Transaction_Latitude_Longitude',fontsize = 18)
#交易发生地点有较大的密集聚集区域

#将train数据与properties的合在一起
train_df = pd.merge(train_df,prop_df,on = 'parcelid',how = 'left')
#print(train_df.head())

#print(train_df.info())
#hashottuborspa,propertyzoningdesc,propertycountylandusecode,fireplaceflag,taxdelinquencyflag 五个object属性
#这几个属性的缺失值都很多,最多的non-nan就1/3,同时有许多int/float属性缺失也很多

missing_ratio = (train_df.isnull().sum()/train_df.shape[0]).reset_index()
missing_df = pd.DataFrame(data = missing_ratio)
#print(missing_df.loc[missing_df[0] > 0.999,:])
#有四个属性99.9%的事件中都缺失

#尝试用平均值来填充缺失值
mean_values = train_df.mean() 
train_df_new = train_df.fillna(mean_values,inplace = True)

#查看各属性与logerror之间的协方差（相关性）
x_cols = [col for col in train_df_new.columns if col != 'logerror' and train_df_new[col].dtype == 'float64']

feature_corr_Dict = {}
for col in x_cols:
	feature_corr_Dict[col] = list()
	#print(np.corrcoef(train_df_new[col].values,train_df_new['logerror'].values)[0,1])	#corrcoef返回的是一个二维矩阵，行索引为(feat_a,feat_b) 列索引为(feat_a,feat_b)
	feature_corr_Dict[col].append(np.corrcoef(train_df_new[col].values,train_df_new['logerror'].values)[0,1])
corr_df = pd.DataFrame.from_dict(feature_corr_Dict)
f,ax4 = plt.subplots() 
sns.barplot(ax = ax4,data = corr_df,orient = "h")
ax4.set_title("Correlation coefficient of the variables",fontsize = 16)
#大部分属性与其有关 但有几个是为0的  表明其变化与logerror无关

usable_feat = list() 
for col in x_cols:
	if corr_df[col].values.tolist()[0] > 0.02 or corr_df[col].values.tolist()[0] < -0.01:
		usable_feat.append(col)
#print(usable_feat)

f,ax5 = plt.subplots()
temp_df = train_df_new[usable_feat]
corr_mat = pd.DataFrame(data = temp_df).corr(method = 'spearman')
sns.heatmap(corr_mat,ax = ax5,vmax = 1.)
ax5.set_title('Important variables correlation map',fontsize = 16)

#print(corr_df.head())
#corr_df = corr_df.sort_values(by = '')
#print(feature_corr_Dict)



























plt.show()


































































