# coding=utf-8 

import numpy as np 
import tensorflow as tf 
import seaborn as sns 
import pandas as pd 
import matplotlib.pyplot as plt 
from plotly import tools 
import plotly.plotly as py 
import plotly.figure_factory as ff 
import plotly.graph_objs as go 
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot 
import matplotlib as mpl
from matplotlib.patches import Circle, Rectangle, Arc
from sklearn import mixture
from sklearn import ensemble
from sklearn import model_selection
from sklearn.metrics import accuracy_score as accuracy
from sklearn.metrics import log_loss
import time
import itertools
import operator
from scipy.stats import skew
from scipy.stats.stats import pearsonr


data = pd.read_csv(r'C:\Users\msi\Desktop\HousePrize\train.csv',low_memory = False)

sns.distplot(data['SalePrice'])

#查看房价与时间的关系
#1.房价均价与年份
fig,ax = plt.subplots()
Year_MeanPrice  = data.groupby('YrSold',as_index = False).mean()
df_YM = pd.DataFrame(data = Year_MeanPrice[['YrSold','SalePrice']])
df_YM.groupby('YrSold').mean().plot.bar(ax = ax)
ax.set_title('Year_MeanPrice',fontsize = 16)
#plt.show()
#年均房价在2006-2010之间保持水平小振幅，故估计房价与年份无关

#2.房价均价与月份
f,ax1 = plt.subplots()
Month_MeanPrice = data.groupby('MoSold',as_index = False).mean() 
df_MM = pd.DataFrame(data = Month_MeanPrice[['MoSold','SalePrice']])
df_MM.groupby('MoSold').mean().plot.bar(ax = ax1)
ax1.set_title('Month_MeanPrice',fontsize = 16)
#9月份的均价最高，4月份的均价最低

#3.房价均价与各年月份
f,ax2 = plt.subplots(1,5,figsize = (18,8)) 
YearMonth_MeanPrice = data.groupby(['YrSold','MoSold'],as_index = False).mean() 
df_YMM = pd.DataFrame(data = YearMonth_MeanPrice[['YrSold','MoSold','SalePrice']])
for i in range(0,5):
	year = 2006 + i 
	df_YMM.loc[df_YMM['YrSold'] == year,:].groupby('MoSold').mean().plot.bar(ax = ax2[i])
	ax2[i].set_title(str(year) + '\'Months MeanPrice',fontsize = 16)

#每年房价均价最高大致都出现在一年的后半段，每年开头的房价都是先往下走再上升

#4.房价与重塑-购买时长的关系
f,ax3 = plt.subplots(1,2,figsize = (25,8)) 
data['TimeDiff'] = np.nan 
data['TimeDiff'] = ((data['YearRemodAdd'] - data['YearBuilt'])/10).astype(int)
TimeDiff_MeanPrice = data.groupby('TimeDiff',as_index = False).mean()
TD_MP = pd.DataFrame(data = TimeDiff_MeanPrice[['TimeDiff','SalePrice']])
TD_MP.groupby('TimeDiff').mean().plot.bar(ax = ax3[0])
ax3[0].set_title('TimeDiff_MeanPrice',fontsize = 16)

TimeDiff_List = list(set(data['TimeDiff'].values.tolist()))
#print(TimeDiff_List)
data['TimeDiff'].value_counts().plot.pie( autopct = "%1.2f%%",ax = ax3[1],shadow = True, fontsize = 12)
ax3[1].set_title("TimeDiff_HousesPercent",fontsize = 16)

#因为重装修-购买的时长中0年的占了70%，其余的从6%开始逐渐随着时长递减，故房价与装修时长的关系中只有0,1,2的可以稍微值得观察
#可以发现随着时长增加，房价减少，但是数量较少，不太可信

#4.房价均价与所处位置的关系
f,ax4 = plt.subplots(1,2,figsize = (25,8))
LocationofCity_MeanPrice = data.groupby('MSZoning',as_index = False).mean() 
LCMP = pd.DataFrame(data = LocationofCity_MeanPrice[['MSZoning','SalePrice']])
LCMP.groupby('MSZoning').mean().plot.bar(ax = ax4[0])
ax4[0].set_title('LocationofCity_MeanPrice',fontsize = 16)

LocationofCity_List = list(set(data['MSZoning'].values.tolist()))
data['MSZoning'].value_counts().plot.pie(autopct = "%1.2f%%",ax = ax4[1],shadow = True, fontsize = 12)
ax4[1].set_title('LocationofCit_Percent',fontsize = 16)
#FV的均价最高，其次是RL,RH,RM,C   但是FV与RM的房子占比最小（RH>RL>C>RM>FV），FV可能是富人区。 RH可能是城市大部分地区，RL可能是城市地理环境较好的地区

#5.房价与房子大小的关系
f,ax5 = plt.subplots()
data['HouseSize'] = np.nan 
data['HouseSize'] = data['1stFlrSF'] = data['2ndFlrSF'] + data['TotalBsmtSF']
sns.lineplot(x = 'HouseSize',y = 'SalePrice', data = data,ax = ax5)
#房价随着住房面积增大而增大。 但是到了4500之后就会开始下降。 可能与该房子的地理位置有关

f,ax6 = plt.subplots(1,2,figsize = (20,8)) 
Location_HighPrice = data.loc[data['HouseSize'] > 4459, ['MSZoning','SalePrice']]
#print(Location_HighPrice)
LHP = pd.DataFrame(data = Location_HighPrice)
LHPL = list(set(LHP['MSZoning'].values.tolist())) 
LHP['MSZoning'].value_counts().plot.pie(autopct = "%1.2f%%",ax = ax6[0],shadow = True, fontsize = 12)
ax6[0].set_title('HighPrice_Location')
Location_HighMeanPrice = LHP.groupby(['MSZoning'],as_index = False).SalePrice.mean()
LHMP = pd.DataFrame(data = Location_HighMeanPrice)
LHMP.groupby('MSZoning').mean().plot.bar(ax = ax6[1])
ax6[1].set_title('Location_HighMeanPrice',fontsize = 16)

#可以发现Housesize大于4459的都是RL区，但是一个是70W多售价，另外两个是20W往下  可能跟其配置有关

#6.配置与房子均价的关系
#配置：Basement CentralAir Electrical FullBath PoolArea GarageFinish FireplaceQu KitchenQual 

#6.1 电力与房子均价	Electrical 
f,ax7 = plt.subplots(1,2) 
Elec_MeanPrice = data.groupby('Electrical',as_index = False).mean() 
EMP = pd.DataFrame(data = Elec_MeanPrice[['Electrical','SalePrice']])
EMP.groupby('Electrical').mean().plot.bar(ax = ax7[0])
ax7[0].set_title("Elec_MeanPrice",fontsize = 16)

#ElectricalList = list(set(data['Electrical'].values.tolist()))
data['Electrical'].value_counts().plot.pie(autopct = "%1.2f%%", ax = ax7[1], shadow = True, fontsize = 12)
ax7[1].set_title("ElectricalList",fontsize = 16)

#SBrkr电力系统的房子均价最高 Mix的最低   同时SBrkr占比91%

#6.2 Pool Area与房子均价
f,ax9 = plt.subplots(1,2)
sns.lineplot(x = 'PoolArea', y = 'SalePrice', data = data, ax = ax9[0])
ax9[0].set_title('PoolArea_Price',fontsize = 16)
#光看游泳池大小与房子均价规律不明显，应配上地理位置来看

sns.lineplot(x = 'PoolArea', y = 'SalePrice', hue = 'MSZoning',data = data, ax = ax9[1])
ax9[1].set_title('Location_Pool_Price',fontsize = 16)
#只有RL地区有泳池，而且前面提及RL可能是地理环境较好的区域（均价第二高），可能是别墅区

#6.3 车库与房子均价
f,ax10 = plt.subplots(1,3,figsize = (16,8)) 
GarageFinish_MeanPrice = data.groupby('GarageFinish',as_index = False).mean() 
GFMP = pd.DataFrame(data = GarageFinish_MeanPrice[['GarageFinish','SalePrice']])
GFMP.groupby('GarageFinish').mean().plot.bar(ax = ax10[0])
ax10[0].set_title('GarageFinish_MeanPrice',fontsize = 16)
#RFn的均价最高 共有三种Finish 
sns.lineplot(x = 'GarageArea', y = 'SalePrice',hue = 'MSZoning', data = data , ax = ax10[1])
ax10[1].set_title("GarageArea_SalePrice")
#C的房价并不随着车库大小增大而增大，同时车库面积大于1000的普遍为RL区的 而且越大越便宜 FV区的几乎成线性增加，可能房子配置比较一致
sns.stripplot(x = 'GarageType',y = 'SalePrice', jitter = True, edgecolor = "blue", data = data,size = 5, ax = ax10[2],split = False)
ax10[2].set_title("GarageType_SalePrice",fontsize = 16)
#Attchd的高售价更多，而且Attchd的房子数量较多。



#7.风格与房价的关系	HouseStyle
f,ax8 = plt.subplots(1,2,figsize = (20,8)) 
HouseStyle = data.groupby('HouseStyle',as_index = False).mean() 
HSMP = pd.DataFrame(data = HouseStyle[['HouseStyle','SalePrice']])
HSMP.groupby('HouseStyle').mean().plot.bar(ax = ax8[0])
ax8[0].set_title("HouseStyle_MeanPrice")

#HouseStyleList = list(set(HSMP['HouseStyle'].values.tolist()))
data['HouseStyle'].value_counts().plot.pie(autopct = "%1.2f%%", ax = ax8[1],shadow = True, fontsize = 12)
ax8[1].set_title("HouseStylePercent",fontsize = 16)
#print(data['HouseStyle'].value_counts())
#1.5Unf 均价最低      2.5Fin均价最高  2Story的是第二高    
#房屋风格分布: 1Story占接近50% 2Story占30% 1.5Fin占10%

#8.浴室、卧室与房价的关系
f,ax11 = plt.subplots(1,2) 
sns.lineplot(x = 'HouseSize', y= 'SalePrice', hue = 'FullBath', data = data, ax = ax11[0])
sns.barplot(x = 'MSZoning', y = 'HouseSize',hue = 'FullBath',data = data, ax = ax11[1])
ax11[0].set_title("FullBath_HouseSize_Price",fontsize = 16)
ax11[1].set_title("Location_Size_FullBath",fontsize = 16)
#浴室与房价的关系结论待思考

#全质量与房价的关系
f,ax12 = plt.subplots()
sns.lineplot(x = 'OverallQual', y = 'SalePrice', data = data,ax = ax12)
ax12.set_title("AllQual_SalePrice")
#大致成线性

#heatmap学习
f,ax13 = plt.subplots(figsize = (20,10))
corrmat = data.corr() 	#返回Pearson矩阵
sns.heatmap(corrmat,vmax = .8,square = True, ax = ax13)
ax13.set_title("Corr_Heatmap",fontsize = 16)
#可以看出信息之间的相关性

#选出corr与价格最有关的属性
k = 10 
cols = corrmat.nlargest(k,'SalePrice')['SalePrice'].index 	#与之最相关的属性的列表
cm = np.corrcoef(data[cols].values.T)		#Pearson矩阵
sns.set(fontsize = 1.25)
hm = sns.heatmap(cm,cbar=True,annot = True, square = True, fmt = '.2f',annot_kws = {'size':10},yticklabels = cols.values,xticklabels = cols.values)

#'OverallQual', 'GrLivArea' and 'TotalBsmtSF' are strongly correlated with 'SalePrice'.
#GarageCar与GarageArea是孪生数据（即corr比较大的 >0.8)

#查看与价格相关的几个关系之间的pairplot
sns.set() 
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(data[cols],size = 3)

#统计缺失值，删掉缺失太多的属性
total_null = data.isnull().sum().sort_values(ascending=False)	#各属性的缺失值排序列表
percent = (data.isnull().sum()/data.isnull().count()).sort_values(ascending = False)	
missing = pd.concat([total_null,percent],axis = 1, keys = ['Total','Percent'])
missing.head()

#删除有缺失的属性	drop
data = data.drop((missing['Total'] > 1).index,1)
data = data.drop(data.loc[data['Electrical'].isnull()].index)
data.isnull().sum().max()

#删除异常值
#先将数值进行标准化  
SalePrice_Scaled = StandardScaler().fit_transform(data['SalePrice'][:,np.newaxis])	#标准化
low_range = SalePrice_Scaled[SalePrice_Scaled[:,0].argsort()][:10]
high_range = SalePrice_Scaled[SalePrice_Scaled[:,0].argsort()][-10:]
print("outer range (low) of the distribution:")
print(low_range)
print("\nouter range (high) of the distribution")
print(high_range)

data.sort_values(by = 'GrLivArea',ascending = False)[:2]
data = data.drop(data[data['Id'] == 1299].index)
data = data.drop(data[data['Id'] == 524].index)

data['HasBsmt'] = pd.Series(len(data['TotalBsmtSF']),index = data.index)
data['HasBsmt'] = 0 
data.loc[data['TotalBsmtSF'] > 0,'HasBsmt'] = 1 
data.loc[data['HasBsmt']==1,'TotalBsmtSF'] = np.log(data['TotalBsmtSF'])

sns.displot(data[data['TotalBsmtSF'] > 0]['TotalBsmtSF'],fit = norm)
fig = plt.figure()
res = stas.probplot(data[data['TotalBsmtSF'] > 0]['TotalBsmtSF'],plot = plt)

data['SalePrice'] - np.log(data['SalePrice'])

plt.scatter(data['GrLivArea'],data['SalePrice'])

Bsmt_SalePrice = data.loc[data['TotalBsmtSF'] > 0,['TotalBsmtSF','SalePrice']]
Bsmt_SalePrice = pd.DataFrame(data = Bsmt_SalePrice)
plt.scatter(Bsmt_SalePrice['TotalBsmtSF'],Bsmt_SalePrice['SalePrice'])


#可用线性拟合
test = pd.read_csv(r'C:\Users\msi\Desktop\HousePrize\test.csv',low_memory = False)

all_data = pd.concat((data.loc[:,'MSSubClass':'SaleCondition'],test.loc[:,'MSSubClass':'SaleCondition']))

matplotlib.rcParams['figure.figsize'] = (12.0,6.0)
prices = pd.DataFrame({"price":data['SalePrice'],"log(price+1)":np.log1p(data["SalePrice"])})
prices.hist()

data['SalePrice'] = np.log1p(data['SalePrice'])
numeric_feats = all_data.dtypes[all_data.dtypes != 'object'].index #属性名称

skewed_feats = data[numeric_feats].apply(lambda x : skew(x.dropna()))	#去除nan后的偏差
skewed_feats = skewed_feats[skewed_feats > 0.75]	
skewed_feats = skewed_feats.index 

all_data[skewed_feats] = np.log1p(all_data[skewed_feats])
all_data = all_data.get_dummies(all_data)
all_data = all_data.fillna(all_data.mean())	#用平均值来填补mean 

X_train = all_data[:data.shape[0]]
X_test = all_data[data.shape[0]:]
y = data.SalePrice

from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.model_selection import cross_val_score

def rmse_cv(model):
	rmse = np.sqrt(-cross_val_score(model,X_train,y,scoring="neg_mean_squared_error",cv = 5))	#交叉验证 cv表示分成几份，返回的是scoring参数指定的值列表（平方损失函数值的负数）
	return(rmse)

model_ridge = Ridge() 	#岭回归对病态数据的拟合比最小二乘法的好
alphas = [0.05,0.1,0.3,1,3,5,10,15,30,50,75]	#正则化参数，参数越大则越不容易过拟合
cv_ridge = [rmse_cv(Ridge(alpha = alpha)).mean() for alpha in alphas]
cv_ridge = pd.Series(cv_ridge,index = False)
cv_ridge.plot()
plt.xlabel("alpha")
plt.ylabel("rmse")

#alpha = 10大概是正确的地方
print(cv_ridge.min())

model_lasso = LassoCV(alphas = [1,0.1,0.001,0.0005]).fit(X_train,y)
print(rmse_cv(model_lasso).mean())
#平均值要优于岭回归
#Lasso可以通过将不重要的属性系数=0来帮我们筛选特征

#xgboost 
import xgboost as xgb 
dtrain = xgb.DMatrix(X_train,label = y)
dtest = xgb.DMatrix(X_test)

params = {'max_depth':2,'eta':0.1}
model = xgb.cv(params,dtrain,num_boost_round = 500, early_stopping_rounds = 100)
model.loc[30:,["test-rmse-mean", "train-rmse-mean"]].plot()

model_xgb = xgb.XGBRegressor(n_estimators = 360, max_depth = 2, learning_rate = 0.1)
model_xgb.fit(X_train,y)

xgb_preds = np.expm1(model_xgb.predict(X_test))
lasso_preds = np.expm1(model_lasso.predict(X_test))
predictions = pd.DataFrame({"xgb":xgb_preds,"lasso":lasso_preds})
predictions.plot(x = "xgb",y = "lasso",kind = "scatter")

#将两者结合
preds = 0.7 * lasso_preds + 0.3 * xgb_preds 

solution = pd.DataFrame({"id":test.Id, "SalePrice":preds})
solution.to_csv("ridge_sol.csv", index = False)

from keras.layers import Dense
from keras.models import Sequential
from keras.regularizers import l1
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

X_train = StandardScaler().fit_transform(X_train)	#标准化
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y, random_state = 3)	#交叉验证 返回X_train,X_test,y_train,y_test

#Doubts
model = Sequential()
model.add(Dense(1,input_dim = X_train.shape[1],W_regularizer = l1(0.001)))
model.compile(loss = "mse", optimizer = "adam")

#print(model.summary())

hist = model.fit(X_tr, y_tr, validation_data = (X_val, y_val))
pd.Series(model.predict(X_val)[:,0]).hist()

plt.show()

   
