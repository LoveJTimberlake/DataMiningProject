# coding=utf-8

import numpy as np 
import pandas as pd 
%matplotlib inline 
import matplotlib.pyplot as plt 
import seaborn as sns 
color = sns.color_palette() 
sns.set_style('darkgrid')
from scipy import stats
from scipy.stats import norm,skew
from subprocess import check_output
import warnings 
def ignore_warn(*args,**kwargs):
	pass 
warnings.warn = ignore_warn

pd.set_option('display.float_format',lambda x : '{:.3f}'.format(x)) #限制其中的数据都只有三位小数

train = pd.read_csv(r'C:\Users\msi\Desktop\HousePrize\train.csv',low_memory = False)
test = pd.read_csv(r'C:\Users\msi\Desktop\HousePrize\test.csv',low_memory = False)

Train_ID = train['Id']
Test_ID = test['Id']
train.drop('Id',axis = 1,inplace = True)
test.drop('Id',axis = 1,inplace = True)

#找出异常值并删除
fig,ax = plt.subplots() 
ax.scatter(x= train['GrLivArea'],y = train['SalePrice'])	#而两者高度符合线性
plt.ylabel('SalePrice',fontsize=16)
plt.xlabel('GrLivArea',fontsize=16)
plt.title('Area_SalePrice',fontsize = 20)

train.drop(train[(train['GrLivArea'] > 4000) & (train['SalePrice'] < 300000)].index)

#SalePrice分布
sns.displot(train['SalePrice'],fit = norm)	#用
(mu,sigma) = norm.fit(data['SalePrice'])
print("mu:{mu},sigma:{sigma}".format(mu,sigma))
fig = plt.figure()
res = stas.probplot(data['SalePrice'],plot = plt)

#将其变为log,从而更好符合用线性拟合 使其分布原本偏离正态分布的数据可以在Log1p之后更加符合正态分布
train['SalePrice'] = np.log1p(train['SalePrice'])
test['SalePrice'] = np.log1p(test['SalePrice'])
sns.displot(train['SalePrice'],fit = norm)	#发现极度符合

fig = plt.figure()
res = stats.probplot(train['SalePrice'],plot = plt)

ntrain = train.shape[0]
ntest = test.shape[0]
y_train = train['SalePrice'].values 
all_data = pd.concat((train,test).reset_index(drop=True))	#新属性作为列添加，旧属性则直接加在下面
all_data.drop('SalePrice',axis = 1,inplace = True)


#对缺失的data处理
all_data_na = (all_data.isnull().sum()/all_data.shape[0]) * 100 
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index()).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing':all_data_na})
#missing_data.head()
for col in ['PoolQC','MiscFeature','Alley','Fence','FireplaceQu','GarageType', 'GarageFinish', 'GarageQual', 'GarageCond','BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2','MasVnrType','MSSubClass']:
	data[col] = data[col].fillna('None')

all_data['LotFrontage'] = all_data.groupby('Neighborhood')['LotFrontage'].transform(lambda x : x.fillna(x.median()))

for col in ('GarageYrBlt',"MasVnrArea", 'GarageArea', 'GarageCars','BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
	all_data[col] = all_data[col].fillna(0)

for col in ('MSZoning','Electrical','KitchenQual','Exterior1st','Exterior2nd','SaleType')
	all_data[col] = all_data[col].fillna(data[col].mode()[0])

all_data = all_data.drop('Utilities',axis = 1)

all_data["Functional"] = all_data['Functional'].fillna("Typ")

all_data_na = all_data.isnull().sum() / all_data.shape[0]
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending = False)

#将一些非数值的离散型的特征用OneHotEnCoder将其转换为numeric（类别值无大小之分的） 若是有大小之分的则用LabelEncoder即可
all_data['MSSubClass'] = all_data['MSSubClass'].astype(str)
all_data['OverallCond'] = all_data['OverallCond'].astype(str)
all_data['YrSold'] = all_data['YrSold'].astype(str)
all_data['MoSold'] = all_data['MoSold'].astype(str)

from sklearn.preprocessing import LabelEncoder
cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold')

for c in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(all_data[c].values)) 
    all_data[c] = lbl.transform(list(all_data[c].values))

all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
all_data.drop(['TotalBsmtSF','1stFlrSF','2ndFlrSF'],axis = 1)

numeric_feats = all_data.dtypes[all_data.dtype != "object"].index 
skewed_feats = all_data[numeric_feats].apply(lambda x : skew(x.dropna())).sort_values(ascending = False)
skewness = pd.DataFrame(data = skewed_feats)
#skewness.head()

#对于skewness过大的数据采用box-cox函数转换
#首先，应该对所有变量进行正态分布检验，如果变量呈严重的偏态分布，
#则最好用BOX-COX transformation,通常根据Lamba值会有倒数、取对数等不同的转换形式，
#一般而言，取对数的形式较常见。  依据就是科布道格拉斯生产函数
skewness = skewness[abs(skewness) > 0.75]

from scipy.special import boxcox1p
skewed_features = skewness.index
lam = 0.15		#lam=0时则等于log1p
for feat in skewed_features:
    #all_data[feat] += 1
    all_data[feat] = boxcox1p(all_data[feat], lam)
    
#all_data[skewed_features] = np.log1p(all_data[skewed_features])

all_data = pd.get_dummies(all_data)
train = all_data[:ntrain]
test = all_data[ntrain:]


from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb

n_folds = 5 
def rmsle_cv(model):
	kf = kFold(n_folds,shuffle = True, randome_state = 42).get_n_splits(train.values)
	rmse = np.sqrt(-cross_val_score(model,train.values,y_train,scoring="neg_mean_squared_error",cv = kf))
	return(rmse)

#线性回归 Lasso 
lasso = make_pipeline(RobustScaler(),Lasso(alpha = 0.0005, randome_state = 1))
score = rmsle_cv(lasso)
#Elasitc Net Regression 
ENet = make_pipeline(RobustScaler(),ElasticNet(alpha = 0.0005,l1_ratio = .9, random_state = 3))	
score = rmsle_cv(ENet)
#Kernel Ridge Regression
KRR = KernelRidge(alpha = 0.6,kernel = 'ploynomial',degree = 2, coef0 = 2.5)
score = rmsle_cv(KRR)
#Gradient Boosting Regression 	使用huber loss来抵消异常值对其的影响 增强鲁棒性
GBoost = GradientBoostingRegressor(n_estimators = 3000,learning_rate = 0.05, max_depth = 4,max_features = 'sqrt',min_samples_leaf = 15, min_samples_split = 10, loss = 'huber',randome_state = 5)
score = rmsle_cv(GBoost)
#XGBoost
model_xgb = xgb.XGBRegressor(colsample_bytree = 0.4603, gamma = 0.0468, learning_rate = 0.05, max_depth = 3, min_child_weight = 1.7817, n_estimators = 2200, reg_alpha = 0.4640, reg_lambda = 0.8571, subsample = 0.5213, silent = 1, random_state = 7, nthread = -1)
score = rmsle_cv(model_xgb)
#LightGBM
model_lgb = lgb.LGBMRegressor(objective = 'regression',num_leaves = 5, learning_rate = 0.05, n_estimators = 720, max_bin = 55, bagging_fraction = 0.8, bagging_freq = 5,feature_fraction = 0.2319, feature_fraction_seed = 9, bagging_seed = 9, min_data_in_leaf = 6, min_sum_hessian_in_leaf = 11)
score = rmsle_cv(model_lgb)

#叠加模型

class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
	def __init__(self,models):
		self.models = models

	def fit(self,X,y):
		self.models = [clone(x) for x in self.models]

		for model in self.models_:
			model.fit(X,y)

		return self 

	def predict(self,X):
		predictions = np.column_stack([model.predict(X) for model in self.models_])

		return np.mean(predictions,axis = 1)

averaged_models = AveragingModels(models = (ENet,GBoost,KRR,lasso))
score = rmsle_cv(averaged_models)
print(" Averaged base models score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

# Averaged base models score: 0.1091 (0.0075)

#在叠加模型上加一个元模型
#1.先将训练数据集划分成两个不相交的数据集 train and holdout
#2.先训练train上的数据
#3.在holdout上测试该模型
#4.使用(3)中的预测作为输入，正确的响应为输出来训练一个高级学习器（meta-model)
'''
The first three steps are done iteratively . 
If we take for example a 5-fold stacking , we first split the training data into 5 folds. 
Then we will do 5 iterations. In each iteration, we train every base model on 4 folds and predict on the remaining fold (holdout fold).
So, we will be sure, after 5 iterations , that the entire data is used to get out-of-folds predictions that we will then use as new feature to train our meta-model in the step 4.
For the prediction part , We average the predictions of all base models on the test data and used them as meta-features on which, 
the final prediction is done with the meta-model.
'''

class StackingAveragedModels(BaseEstimator,RegressorMixin,TransformerMixin):
	def __init__(self,base_models,meta_model,n_folds = 15):
		self.base_models = base_models;
		self.meta_model = meta_model;
		self.n_folds = n_folds

	def fit(self,X,y):
		self.base_models_ = [list() for x in self.base_models]
		self.meta_model_ = clone(self.meta_model)
		kflod = KFold(n_splits = self.n_folds, shuffle = True, random_state = 156)

		out_of_fold_predictions = np.zeros((X.shape[0],len(self.base_models)))
		for i,model in enumerate(self.base_models):
			for train_index, holdout_index in kflod.split(X,y):
				instance = clone(model)
				self.base_models_[i].append(instance)
				instance.fit(X[train_index],y[train_index])
				y_pred = instance.predict(X[holdout_index])
				out_of_fold_predictions[holdout_index,i] = y_pred 

		self.meta_model_.fit(out_of_fold_predictions,y)
		return self 

	def predict(self,X):
		meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_ ])
		return self.meta_model_.predict(meta_features)


#将lasso作为meta-model
stacked_averaged_models = StackingAveragedModels(base_models = (ENet,GBoost,KRR),meta_model = lasso)
score = rmsle_cv(stacked_averaged_models)

#集成学习
def rmsle(y,y_pred):
	return np.sqrt(mean_squared_error(y,y_pred))

stacked_averaged_models.fit(train.values, y_train)
stacked_train_pred  = stacked_averaged_models.predict(train.values)
stacked_pred = np.expm1(stacked_averaged_models.predict(test.values))
#print(rmsle(y_train,stacked_train_pred))

#XGBoost
model_xgb.fit(train, y_train)
xgb_train_pred = model_xgb.predict(train)
xgb_pred = np.expm1(model_xgb.predict(test))
#print(rmsle(y_train, xgb_train_pred))

#LightGBM
model_lgb.fit(train, y_train)
lgb_train_pred = model_lgb.predict(train)
lgb_pred = np.expm1(model_lgb.predict(test.values))
#print(rmsle(y_train, lgb_train_pred))

ensemble = stacked_pred*0.70 + xgb_pred*0.15 + lgb_pred*0.15
print(rmsle(y_train,stacked_train_pred*0.70 +
               xgb_train_pred*0.15 + lgb_train_pred*0.15 ))


plt.show()


sub = pd.DataFrame()
sub['Id'] = test_ID
sub['SalePrice'] = ensemble
sub.to_csv('submission.csv',index=False)













































