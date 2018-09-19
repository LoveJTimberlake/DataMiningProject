# coding=utf-8

import pandas as pd
import numpy as numpy
import matplotlib.pyplot as plt 
import warnings 
import seaborn as sns
plt.style.use('fivethirtyeight')
warnings.filterwarnings('ignore')


#机器学习模块导入
from sklearn.linear_model import LogisticRegression 
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split 
from sklearn import metrics 
from sklearn.metrics import confusion_matrix 
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

titanic = pd.read_csv(r"C:\Users\msi\Desktop\Titanic\train.csv")
titanic.head()

titanic.isnull().sum()		# show total number of NULL value
#也可用titanic.info()显示相关信息

'''
PassengerId      0
Survived         0
Pclass           0
Name             0
Sex              0
Age            177
SibSp            0
Parch            0
Ticket           0
Fare             0
Cabin          687
Embarked         2
total          891
dtype: int64
'''

#1.Check the data distribution generally 


'''
fig = plt.figure()
ax1 = flg.add_subplot(2,2,1)
ax2 = flg.add_subplot(2,2,2)
'''
f,ax = plt.subplots(1,2,figsize=(18,8))		#创建一个新的Figure,并返回一个subplot对象的数组ax。将画的图分为两个subplot, 一行两列排列。 图像大小是18*8  ax为子图索引字典
titanic['Survived'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow = True)	#pie是圆饼图
#plt.pie(titanic['Survived'].value_counts(),explode = [0,0.1],autopct = '%1.1f%%', shadow = True)		另一个方法
ax[0].set_title('Survived')
ax[0].set_ylabel('')
sns.countplot('Survived',data = titanic, ax = ax[1])		#countplot是直方图 第一个参数是要统计的数据属性值，第二个是数据集，第三个是指作图的结果表现的子图位置
ax[1].set_title('Survived')
#plt.show()



'''
PassengerId 乘客ID 		int型数据 
Surviveed 是否生还 		0或者1 
Pclass 乘客的社会等级 	1或者2或者3 
Name 乘客名字 			string 
Sex 乘客性别 			male或者female 
Age 乘客年龄 			float一位小数 
SibSP 在船上的家庭人员数 int型数据 
Parch 在船上的双亲数 	int型数据 
Ticket 船票编号			string 
Fare 船票费用 			float四位小数 
Cabin 船舱号 			string 
Embarked 上船港口 		C或者Q或者S
'''

#查看生还率与港口的关系
ff,ax1 = plt.subplots(1,2,figsize = (18,8))
sns.countplot('Embarked',data = titanic,order = ['Q','C','S'],ax = ax1[0])
ax1[0].set_title('total number : Embarked')
sns.countplot('Embarked',hue = 'Survived', order = ['Q','C','S'], data = titanic, ax = ax1[1])	#hue表示分类序组
ax1[1].set_title('Embarked: Survived or not')
#plt.show()


#查看性别与生还率的关系
fff,ax2 = plt.subplots(1,2,figsize = (18,8))
titanic[['Sex','Survived']].groupby(['Sex']).mean().plot.bar(ax = ax2[0])		#.mean()对所有的值求均值 bar()是作柱状图
ax2[0].set_title('Survived and Sex')
sns.countplot('Sex',hue = 'Survived', data = titanic, ax = ax2[1])
ax2[1].set_title('Sex and Survived')
#plt.show()

#查看社会等级与生还率的关系
pd.crosstab(titanic.Pclass,titanic.Survived,margins = True).style.background_gradient(cmap='summer_r')	#传递两个因子计算其交叉表，先传递两个参数数组作为因子的数据，margins为边距
ffff,ax3 = plt.subplots(1,2,figsize=(18,8))
titanic['Pclass'].value_counts().plot.bar(color = ['#CD7F32','#FFDF00','#D3D3D3'],ax = ax3[0])
ax3[0].set_title('Number of Passengers By Pclass')
ax3[0].set_ylabel('Count')
sns.countplot('Pclass', hue = 'Survived',data = titanic, ax = ax3[1])
ax[1].set_title('Pclass:Survived vs Dead')
#plt.show()

#查看年龄和阶级以及年龄和性别与生还率的关系     用violin图，适合表示1个连续性变量与2个离散变量（y轴是连续性变量（第二个参数），x轴是离散变量（第一个参数），hue是离散变量）
f5,ax4 = plt.subplots(1,2,figsize=(18,8))
sns.violinplot("Pclass","Age",hue = "Survived", data = titanic, split = True, ax = ax4[0])
ax4[0].set_title('Pclass and Age vs Survived')
ax4[0].set_yticks(range(0,110,10))
sns.violinplot('Sex',"Age",hue = "Survived", data = titanic, split = True, ax = ax4[1])
ax4[1].set_title('Sex and Age vs Survived')
ax4[1].set_yticks(range(0,110,10))
#plt.show()

#船上亲戚数量及船上双亲数与生存率的关系
titanic['Family_Size'] = titanic['SibSp'] + titanic['Parch']
f6,ax5 = plt.subplots(1,2,figsize = (18,8))
sns.countplot('Family_Size',data = titanic, ax = ax5[0])
ax5[0].set_title('count of Family_Size')
sns.countplot('Family_Size',hue = 'Survived',data = titanic, ax = ax5[1])
ax5[1].set_title('Survived and Family_Size')
#plt.show()

#船票费与生存率关系		用区间分布图
f7,ax6 = plt.subplots(1,3,figsize=(20,8))
sns.distplot(titanic[titanic['Pclass']==1].Fare,ax = ax6[0])	#dataset[attribute] == x 返回的是(number,bool)列表，dataset[dataset[attribute] == x ]返回的是内条件为真的全行字典
ax6[0].set_title("Fare in Pclass 1")
sns.distplot(titanic[titanic['Pclass']==2].Fare,ax = ax6[1])
ax6[1].set_title('Fare in Pclass 2')
sns.distplot(titanic[titanic['Pclass']==3].Fare, ax = ax6[2])
ax6[2].set_title('Fare in Pclass 3')

#协方差（变量之间的关系）
f8,ax7 = plt.subplots()
relt = titanic.corr()
sns.heatmap(relt,vmax = .8,ax = ax7)

#plt.show()



#2.数据清洗与特征工程

#港口S的生还率特别低，女性当中的生还率较高，社会阶级越高生还率越高，阶级越高，年龄大的生还率高，阶级越低，年龄20-30的生还率高
#船上有1-3个亲属的人生还率逐渐提高，与船费应该无关
#故有关因素如上所示

#缺失值补充：
'''
PassengerId      0
Survived         0
Pclass           0
Name             0
Sex              0
Age            177
SibSp            0
Parch            0
Ticket           0
Fare             0
Cabin          687
Embarked         2
total          891
'''
#港口值缺失数较少，按比例随机分配港口/直接赋予众数
#Cabin缺失太多，没有发现与之有较大关联的因子 故忽略
#Age可根据船上亲属数（小孩子在船上必有亲戚），名字也可以用来预测年龄
titanic['F_Name'] = 0
for i in titanic:
	titanic['F_Name'] = titanic.Name.str.extract('([A-Za-z]+)\.')	
pd.crosstab(titanic.F_Name,titanic.Sex).T.style.background_gradient(cmap = 'summer_r')	
titanic['F_Name'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don'],['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr'],inplace=True)

#print(titanic[['F_Name','Age']].groupby(['F_Name']).mean())		#对各种名字对应的年龄求平均数并用于填补空缺年龄
titanic.loc[(titanic.Age.isnull())&(titanic.F_Name == 'Mr'),'Age'] = 33
titanic.loc[(titanic.Age.isnull())&(titanic.F_Name == 'Mrs'),'Age'] = 36
titanic.loc[(titanic.Age.isnull())&(titanic.F_Name == 'Master'),'Age'] = 5
titanic.loc[(titanic.Age.isnull())&(titanic.F_Name == 'Miss'),'Age'] = 22
titanic.loc[(titanic.Age.isnull())&(titanic.F_Name == 'Other'),'Age'] = 46

#通过热力图查看多余标签(已去掉Cabin与将SibSp,Parch合在一起)
sns.heatmap(titanic.corr(),annot=True,cmap='RdYlGn',linewidths=0.2)		#corr是协方差
fig=plt.gcf()
fig.set_size_inches(10,8)
#plt.show()

#将年龄进行分段
titanic['Age_band'] = 0 
titanic.loc[titanic['Age'] <= 16, 'Age_band'] = 0 
titanic.loc[(titanic['Age'] > 16) & (titanic['Age'] <= 32),'Age_band'] = 1
titanic.loc[(titanic['Age'] > 32) & (titanic['Age'] <= 48),'Age_band'] = 2
titanic.loc[(titanic['Age'] > 48) & (titanic['Age'] <= 64),'Age_band'] = 3
titanic.loc[titanic['Age'] > 64, 'Age_band'] = 4

#统计年龄段人数
titanic['Age_band'].value_counts().to_frame().style.background_gradient(cmap='summer')

'''
Age_band	Num
1			382
2			325
0			104
3			69
4			11
'''

#家庭人数在1-3之间的生还率更高，因被加入因子
titanic['Alone or Many_Parents'] = 0
titanic.loc[titanic.Family_Size == 0, 'Alone or Many_Parents' ] = 1 
titanic.loc[titanic.Family_Size > 4 , 'Alone or Many_Parents' ] = 1 

#船费与社会阶段相关，将船费分段并变成离散型变量
titanic['Fare_Range'] = pd.qcut(titanic['Fare'],4)		#切成四个阶段使其频率不会差太多
titanic.groupby(['Fare_Range'])['Survived'].mean().to_frame().style.background_gradient(cmap='summer_r')
titanic['Fare_cat'] = 0
titanic.loc[titanic['Fare'] <= 7.91, 'Fare_cat'] = 0 
titanic.loc[(titanic['Fare'] > 7.91) & (titanic['Fare'] <= 14.454),'Fare_cat'] = 1
titanic.loc[(titanic['Fare'] > 14.454) & (titanic['Fare'] <= 31),'Fare_cat'] = 2
titanic.loc[(titanic['Fare'] > 31) & (titanic['Fare'] <= 513),'Fare_cat'] = 3

#预测模型构建，将string变量变为int
titanic['Sex'].replace(['male','female'],[0,1],inplace = True)
titanic['Embarked'].replace(['S','C','Q'],[0,1,2],inplace = True)		#应该变成三个变量
titanic['F_Name'].replace(['Mr','Mrs','Miss','Master','Other'],[0,1,2,3,4],inplace=True)	#应该直接删掉 因为其只与性别年龄有关

titanic.drop(['Name','Age','Ticket','Fare','Cabin','Fare_Range','F_Name','SibSp','Parch','PassengerId'],axis=1,inplace=True)
titanic.loc[titanic.Embarked.isnull(),'Embarked'] = 0 

sns.heatmap(titanic.corr(),annot=True,cmap='RdYlGn',linewidths = 0.2, annot_kws = {'size':20})
fig=plt.gcf() 
fig.set_size_inches(18,15)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
#plt.show()


train,test = train_test_split(titanic,test_size = 0.3, random_state = 0, stratify = titanic['Survived'])	#以stratify指定的参数值保证拆分数据集之后各个数据集该参数的分布均一样
train_x = train[train.columns[1:]]		#train.columns[1:]是一个列属性索引名的迭代器
train_y = train[train.columns[:1]]		#train_y = train['Survived']
test_x = test[test.columns[1:]]
test_y = test[test.columns[:1]]
x = titanic[titanic.columns[1:]]
y = titanic['Survived']

#print(train.isnull().sum())
#print(test.isnull().sum())

#径向支持向量机
model1 = svm.SVC(kernel='rbf', C = 1, gamma = 0.1)
model1.fit(train_x,train_y)
prediction1 = model1.predict(test_x)
print("RBF SVM Accuracy",metrics.accuracy_score(prediction1,test_y))


#线性支持向量机
model2 = svm.SVC(kernel = 'linear',C = 0.1 , gamma = 0.1)
model2.fit(train_x,train_y)
prediction2 = model2.predict(test_x)
print("Linear SVM Accuracy",metrics.accuracy_score(prediction2,test_y))

#逻辑回归
model3 = LogisticRegression()
model3.fit(train_x,train_y)
prediction3=model3.predict(test_x)
print("LR Accuracy:", metrics.accuracy_score(prediction3,test_y))

#决策树
model4 = DecisionTreeClassifier()
model4.fit(train_x,train_y)
prediction4=model4.predict(test_x)
print("DT Accuracy:",metrics.accuracy_score(prediction4,test_y))

#KNN
model5 = KNeighborsClassifier()
model5.fit(train_x,train_y)
prediction5 = model5.predict(test_x)
print("KNN Accuracy:",metrics.accuracy_score(prediction5,test_y))

#朴素贝叶斯
model6 = GaussianNB()
model6.fit(train_x,train_y)
prediction6 = model6.predict(test_x)
print("Bayes Accuracy:", metrics.accuracy_score(prediction6,test_y))

#随机森林
model7 = RandomForestClassifier()
model7.fit(train_x,train_y)
prediction7 = model7.predict(test_x)
print("RF Accuracy:",metrics.accuracy_score(prediction7,test_y))

#XGBoost
model_xgb = xgb.XGBClassifier(silent = 0,learning_rate = 0.05,max_depth = 3, gamma = 0.005, subsample = 1, max_delta_step = 0.5,n_estimators = 100,seed = 1000,colsample_bytree = 1)
model_xgb.fit(train_x,train_y)
prediction8 = model_xgb.predict(test_x)
print("XGBoost Accuracy",metrics.accuracy_score(prediction8,test_y))

#K叠交叉验证法
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score 
from sklearn.model_selection import cross_val_predict 
kfold = KFold(n_splits = 10, random_state = 22)		#划分为10个数据集
xyz = [] 
accuracy = [] 
std = [] 
classfiers = ['Linear Svm','Radial Svm','Logistic Regression','KNN','Decision Tree','Naive Bayes','Random Forest']
models = [svm.SVC(kernel = 'linear'),svm.SVC(kernel = 'rbf'),LogisticRegression(),KNeighborsClassifier(n_neighbors=9),DecisionTreeClassifier(),GaussianNB(),RandomForestClassifier(n_estimators=100)]

for i in models:
	model = i
	cv_result = cross_val_score(model,x,y,cv = kfold,scoring="accuracy")
	cv_result = cv_result
	xyz.append(cv_result.mean())
	std.append(cv_result.std())
	accuracy.append(cv_result)

new_models_dataframe2 = pd.DataFrame({'CV Mean':xyz, 'Std':std},index = classfiers)

plt.subplots(figsize = (12,6))
box = pd.DataFrame(accuracy,index = [classfiers])
box.T.boxplot()

new_models_dataframe2['CV Mean'].plot.barh(width=0.8)
plt.title('Avg CV Mean Accuracy')
flg = plt.gcf()
fig.set_size_inches(8,5)
#plt.show()



#混淆矩阵法
f9,ax9 = plt.subplots(3,3,figsize = (12,10))
y_pred = cross_val_predict(svm.SVC(kernel = 'rbf'),x,y,cv = 10)
sns.heatmap(confusion_matrix(y,y_pred),ax = ax9[0,0], annot = True, fmt = '2.0f')
ax9[0,0].set_title('Matrix for rbf-SVM')
y_pred = cross_val_predict(svm.SVC(kernel = 'linear'),x,y,cv= 10)
sns.heatmap(confusion_matrix(y,y_pred),ax = ax9[0,1],annot = True, fmt = '2.0f')
ax9[0,1].set_title("Matrix for Linear-SVM")


#模型调参
#支持向量机

from sklearn.model_selection import GridSearchCV
C=[0.05,0.1,0.2,0.3,0.25,0.4,0.5,0.6,0.7,0.8,0.9,1]
gamma=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
kernel = ['rbf','linear']
hyper = {'kernel':kernel,'C':C,'gamma':gamma}
gd = GridSearchCV(estimator=svm.SVC(), param_grid = hyper, verbose = True)
gd.fit(x,y)
print(gd.best_score_)
print(gd.best_estimator_)


#随机森林
n_estimators = range(100,1000,100)
hyper = {'n_estimators':n_estimators}
gd = GridSearchCV(estimator = RandomForestClassifier(random_state = 0), param_grid = hyper,verbose = True)
gd.fit(x,y)
print(gd.best_score_)
print(gd.best_estimator_)

#stack model




























