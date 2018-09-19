# coding=utf-8


import tensorflow as tf 
import pandas as pd 
import seaborn as sns 
import numpy as np 
import matplotlib.pyplot as plt 
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import make_pipeline,Pipeline,FeatureUnion
from imblearn.pipeline import make_pipeline as imbalanced_make_pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from imblearn.metrics import classification_report_imbalanced
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report
from collections import Counter
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.base import BaseEstimator, TransformerMixin                                                                                                                                                                                                                                                                                                                            
from sklearn.utils import check_array
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy import sparse
OrderedDict
from plotly import tools 
import plotly.plotly as py 
import plotly.figure_factory as ff 
import plotly.graph_objs as go 
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot 
init_notebook_mode(connected = True )

from imblearn.over_sampling import SMOTE 	#SMOTE对少数类样本进行分析并合成新的样本加入dataset，来防止过拟合  Smote_Algo.py中有
from collections import Counter

import time 

df = pd.read_csv(r"C:\Users\msi\Desktop\loan.csv", low_memory = False)

original_df = df.copy() 

#df.head()
#df.info()		属性 数量 not-null/null 类型

#先概览整体数据分布：贷款数目，借出金额，投资金额

fig,ax = plt.subplots(1,3,figsize=(16,5))

loan_amount  = df["loan_amount"].values 		#变成值列表 
funded_amount = df["funded_amount"].values 
investor_funds = df["investor_funds"].values

sns.displot(loan_amount, ax = ax[0], color = "#7522F")		#第一个参数是一个数组/列表
ax[0].set_title("Loan Applied by the Borrower", fontsize = 14)	#fontsize字体大小
sns.displot(funded_amount, ax = ax[1], color = "2F8FF7")
ax[1].set_title("Amount Funded by the Lender",fontsize = 14)
sns.displot(investor_funds,ax =ax[2], color = "#2EAD46")
ax[2].set_title("Total committed by Investors", fontsize=14)

#上面三图观察发现三者分布相似


#年份与贷款之间的关系
dt_series = pd.to_datetime(df['issue_d'])		#http://pandas.pydata.org/pandas-docs/stable/generated/pandas.to_datetime.html#pandas.to_datetime
df['year'] = dt_series.dt.year	# 新创年份属性		#对to_datetime对象调用其字典


plt.figure(figsize = (12,8))
sns.barplot('year','loan_amount',data = df,palette = 'tab10')		#第一个为x，第二个为y 
plt.title("Issuance of Loans", fontsize = 14)	
plt.xlabel('year',fontsize = 14)
plt.y_label("Average loan amount issued", fontsize = 14)

#贷款数量逐年升高


df["loan_status"].value_counts() 
'''
Current                                                601779
Fully Paid                                             207723
Charged Off                                             45248
Late (31-120 days)                                      11591
Issued                                                   8460
In Grace Period                                          6253
Late (16-30 days)                                        2357
Does not meet the credit policy. Status:Fully Paid       1988
Default                                                  1219
Does not meet the credit policy. Status:Charged Off       761
Name: loan_status, dtype: int64
'''

#对贷款好坏进行二分值变化
bad_loan= ["Charged Off", "Default", "Does not meet the credit policy. Status:Charged Off", "In Grace Period", "Late (16-30 days)", "Late (31-120 days)"]

df['loan_condition'] = np.nan

def loan_condition(status):
	if status in bad_loan:
		return "Bad Loan"
	else:
		return "Good Loan"

df['loan_condition'] = df['loan_status'].apply(loan_condition)

#查看好坏贷款分布与好坏贷款随年份变化的关系

f,ax = plt.subplots(1,2, figsize = (16,8))
colors = ["#3791D7", "#D72626"]
labels = "Good Loans","Bad Loans"

plt.suptitle('Information on Loan Conditions', fontsize=20)
df['loan_condition'].value_counts().plot.pie(explode = [0,0.1], autopct = "%1.2f%%", ax =ax[0],shadow = True, colors = colors, labels = labels, fontsize = 12, startangle = 70)

ax[0].set_ylabel(r"% of Condition of Loans", fontsize = 4)

palette = ["#3791D7", "#E01E1B"]
sns.barplot(x = "year",y = "loan_amount",hue = "loan_condition", data = df,palette = palette, estimator = lambda x : len(x) / len(df) * 100)		#hue 表示根据其属性值决定不一样的数据用不同的颜色
ax[1].set_ylabel("%",fontsize = 4)

#近几年的bad_loan仅有7.6%,同时随着年份增长，good_loan越来越多而bad_loan有减少趋势

#Loan and Region 
#Translate different places into five parts

#df['addr_state'].unique() 

west = ['CA', 'OR', 'UT','WA', 'CO', 'NV', 'AK', 'MT', 'HI', 'WY', 'ID']
south_west = ['AZ', 'TX', 'NM', 'OK']
south_east = ['GA', 'NC', 'VA', 'FL', 'KY', 'SC', 'LA', 'AL', 'WV', 'DC', 'AR', 'DE', 'MS', 'TN' ]
mid_west = ['IL', 'MO', 'MN', 'OH', 'WI', 'KS', 'MI', 'SD', 'IA', 'NE', 'IN', 'ND']
north_east = ['CT', 'NY', 'PA', 'NJ', 'RI','MA', 'MD', 'VT', 'NH', 'ME']

df['region'] = np.nan 

def finding_regions(state):
	if state in west:
		return 'West'
	elif state in south_west:
		return 'SouthWest'
	elif state in south_east:
		return 'SouthEast'
	elif state in mid_west:
		return 'MidWest'
	elif state in north_east:
		return 'NorthWest'

df['region'] = df['addr_state'].apply(finding_regions)


#统计每个区域随年份变化的贷款数量
df['complete_date'] = pd.to_datetime(df['issue_d'])
group_dates = df.groupby(['complete_date', 'region'], as_index = False).sum()			#as_index=False 使得聚合结果不会出现多级index

group_dates['issue_d'] = [month.to_period('M') for month in group_dates['complete_date']]		#变为以年-月为索引的表格		month = "2017-01-01" to_period["M"]->"2017-01"
group_dates = group_dates.groupby(['issue_d','region'], as_index = False).sum()		#以年月与区域为分组的对每个年月每个地域的所有数据进行求和
group_dates['loan_amount'] = group_dates['loan_amount']/1000 

df_dates = pd.DataFrame(data = group_dates[['issue_d','region','loan_amount']])
plt.style.use('dark_background')
cmap = plt.cm.Set3
by_issued_amount = df_dates.groupby(['issue_d','region']).loan_amount.sum()
by_issued_amount.unstack().plot(stacked = False, colormap = cmap, legend = True, figsize = (15,6))		#unstack() 将行索引变为列索引 默认是最里面那层 If the index is not a MultiIndex, the output will be a Series
plt.set_title('Loans issued by Region', fontsize = 16)		# 又等于 plt.title('Loans issued by Region', fontsize = 16)

#东北地区的贷款能力最高，西南的则最低


#将工作年限属性变为int属性 emp_length
employment_length = ['10+ years', '< 1 year','1 year','3 years','8 years', '9 years','4 years','5 years', '6 years','2 years','7 years','n/a']

df['emp_length_int'] = np.nan 
df.loc[df['emp_length'] == '10+ years', "emp_length_int"] = 10		#df.loc[:, 'x'] = n
df.loc[df['emp_length'] == '9 years', "emp_length_int"] = 9
df.loc[df['emp_length'] == '8 years', "emp_length_int"] = 8
df.loc[df['emp_length'] == '7 years', "emp_length_int"] = 7
df.loc[df['emp_length'] == '6 years', "emp_length_int"] = 6
df.loc[df['emp_length'] == '5 years', "emp_length_int"] = 5
df.loc[df['emp_length'] == '4 years', "emp_length_int"] = 4
df.loc[df['emp_length'] == '3 years', "emp_length_int"] = 3
df.loc[df['emp_length'] == '2 years', "emp_length_int"] = 2 
df.loc[df['emp_length'] == '1 years', "emp_length_int"] = 1 
df.loc[df['emp_length'] == '< 1 years', "emp_length_int"] = 0.5
df.loc[df['emp_length'] == 'n/a', "emp_length_int"] = 0


#各州各变量随年份变化(利率，工作长度，平均债与收入比，平均年收入)
sns.set_style('whitegrid')
figure , axx = plt.subplots(2,2) 		
cmap = plt.cm.inferno 

by_interest_rate = df.groupby(['year','region']).interest_rate.mean() 
by_interest_rate.unstack().plot(kind = 'area', stacked = True, colormap = cmap, grid = False, legend = False, ax = axx[0], figsize = (16,12))
axx[0].set_title('Average Interest Rate by Region', fontsize=14)

by_employment_length = df.groupby(['year', ' region']).emp_length_int.mean() 
by_employment_length.unstack().plot(kind = 'area', stacked = True, colormap = cmap, grid = False, legend=False, ax =axx[1], figsize = (16,12))
axx[1].set_title('Average Employment Length by Region', fontsize=14)
	
by_dti = df.groupby(['year','region']).dti.mean() 
by_dti.unstack().plot(kind = 'area', stacked = True, colormap=cmap, grid = False, legend = False, ax = axx[2],figsize=(16,12))
axx[2].set_title('Average Debt-to-Income by Region', fontsize=14)

by_income = df.groupby(['year','region']).annual_income.mean() 
by_income.unstack().plot(kind='area', stacked = Trye, colormap=cmap, grid = False, ax = axx[3], figsize = (16,12))
axx[3].set_title('Average Annual Income by Region', fontsize=14)
axx[3].legend(bbox_to_anchor = (-1.0,-0.5,1.8,0.1), loc = 10, prop = {'size':12}, ncol = 5, mode = "expand", borderaxespad = 0.)

#每个州在四个图的位置分布一样,利率均减少但是其他都增加


#对bad loan的数据进行具体分析
#先求bad_loan中各还债情况的百分比
badloans_df = df.loc[df['loan_condition'] == "Bad Loan"]

loan_status_cross = pd.crosstab(badloans_df['region'],badloans_df['loan_status']).apply(lambda x : x/ x.sum() * 100 )
number_of_loanstatus = pd.crosstab(badloans_df['region'], badloans_df['loan_status'])

loan_status_cross['Charged Off'] = loan_status_cross['Charged Off'].apply(lambda x : round(x,2)) 
loan_status_cross['Default'] = loan_status_cross['Default'].apply(lambda x : round(x,2))
loan_status_cross['Does not meet the credit policy. Status:Charged Off'] = loan_status_cross['Does not meet the credit policy. Status:Charged Off'].apply(lambda x: round(x, 2))
loan_status_cross['In Grace Period'] = loan_status_cross['In Grace Period'].apply(lambda x : round(x,2)) 
loan_status_cross['Late (16-30 days)'] = loan_status_cross['Late (16-30 days)'].apply(lambda x: round(x, 2))
loan_status_cross['Late (31-120 days)'] = loan_status_cross['Late (31-120 days)'].apply(lambda x: round(x, 2))
 
number_of_loanstatus['Total'] = number_of_loanstatus.sum(axis = 1) 	#对每一行前面的属性值求和
#print(number_of_loanstatus)


#先将各贷款还款情况做成对应的表
#先将其值变为List便于迭代
charged_off = loan_status_cross['Charged Off'].values.tolist() 
default = loan_status_cross['Default'].values.tolist() 
not_meet_credit = loan_status_cross['Does not meet the credit policy. Status:Charged Off'].values.tolist()
grace_period = loan_status_cross['In Grace Period'].values.tolist() 
short_pay = loan_status_cross['Late (16-30 days)'] .values.tolist()
long_pay = loan_status_cross['Late (31-120 days)'].values.tolist()
	
charged = go.Bar(
	x=['MidWest', 'NorthEast', 'SouthEast', 'SouthWest', 'West'],
    y= charged_off,													#传入x与y的为其值的list
    name='Charged Off',
    marker=dict(
        color='rgb(192, 148, 246)'
    ),
    text = '%'
)

defaults = go.Bar(
	x=['MidWest', 'NorthEast', 'SouthEast', 'SouthWest', 'West'],
    y=default,
    name='Defaults',
    marker=dict(
        color='rgb(176, 26, 26)'
    ),
    text = '%'
)

grace = go.Bar(
    x=['MidWest', 'NorthEast', 'SouthEast', 'SouthWest', 'West'],
    y= grace_period,
    name='Grace Period',
    marker = dict(
        color='rgb(147, 147, 147)'
    ),
    text = '%'
)

short_pays = go.Bar(
    x=['MidWest', 'NorthEast', 'SouthEast', 'SouthWest', 'West'],
    y= short_pay,
    name='Late Payment (16-30 days)', 
    marker = dict(
        color='rgb(246, 157, 135)'
    ),
    text = '%'
)

long_pays = go.Bar(
    x=['MidWest', 'NorthEast', 'SouthEast', 'SouthWest', 'West'],
    y= long_pay,
    name='Late Payment (31-120 days)',
    marker = dict(
        color = 'rgb(238, 76, 73)'
        ),
    text = '%'
)

data = [charged, defaults, credit_policy, grace, short_pays, long_pays]
layout = go.layout(
	barmode = 'stack',
	title =  '% of Bad Loan Status by Region',
	xaxis = dict(title = 'US Regions')
	)

fig = go.Figure(data = data, layout = layout)
iplot(fig, filename = 'stacked-bar')

#平均利率
ir = df['interest_rate'].mean() 
#平均年收入
ai = df['annual_income'].mean()

#西部与东南部的坏贷款占总比较多


#three key metrics: Loans issued by state (Total Sum), 
#Average interest rates charged to customers and average annual income of all customers by state.
#to see states that give high returns at a descent risk

by_loan_amout = df.groupby(['region','addr_state'], as_index = False).loan_amount.sum()
by_interest_rate = df.groupby(['region','addr_state'], as_index = False).interest_rate.mean()
by_income = df.groupby(['region','addr_state'], as_index = False).annual_income.mean()

states = by_loan_amout['addr_state'].values.tolist() 
average_loan_amounts = by_loan_amout['loan_amount'].values.tolist() 
average_interest_rates = by_interest_rate['interest_rate'].values.tolist() 
average_annual_income = by_income['annual_income'].values.tolist()

from collections import OrderedDict

metrics_data = OrderedDict([('state_codes', states),
                            ('issued_loans', average_loan_amounts),
                            ('interest_rate', average_interest_rates),
                            ('annual_income', average_annual_income)])

metrics_df = pd.DataFrame.from_dict(metrics_data)		# dict -> dataframe 
metrics_df = metrics_df.round(decimals = 2)

#做出USA的州际图

for col in metrics_df.columns:
	metrics_df[col] = metrics_df[col].astype(str)		#转化为str

scl = [[0.0, 'rgb(210, 241, 198)'],[0.2, 'rgb(188, 236, 169)'],[0.4, 'rgb(171, 235, 145)'],\
            [0.6, 'rgb(140, 227, 105)'],[0.8, 'rgb(105, 201, 67)'],[1.0, 'rgb(59, 159, 19)']]

#设置鼠标移到州对应位置时的显示内容
metrics_df['text'] = metrics_df['state_codes'] + '<br>' +\
'Average loan interest rate: ' + metrics_df['interest_rate'] + '<br>'+\
'Average annual income: ' + metrics_df['annual_income'] 

data = [ dict(
        type='choropleth',
        colorscale = scl,
        autocolorscale = False,
        locations = metrics_df['state_codes'],
        z = metrics_df['issued_loans'], 
        locationmode = 'USA-states',
        text = metrics_df['text'],
        marker = dict(
            line = dict (
                color = 'rgb(255,255,255)',
                width = 2
            ) ),
        colorbar = dict(
            title = "$s USD")
        ) ]

layout = dict(
    title = 'Lending Clubs Issued Loans <br> (A Perspective for the Business Operations)',
    geo = dict(
        scope = 'usa',
        projection=dict(type='albers usa'),
        showlakes = True,
        lakecolor = 'rgb(255, 255, 255)')
)

fig = dict(data = data, layout = layout)
iplot(fig, filename = 'd3-cloropleth-map')	 #一张图由数据集及其属性还有外框架及其属性构成

#create income categories in order to detect patters 
#Low income category , Medium income category , High income category  (0,100000) (100000,200000) (200000,)

df['income_category'] = np.nan 
lst = [df] 

for col in lst:
	col.loc[col['annual_income'] <= 100000, 'income_category'] = 'Low'
	col.loc[(col['annual_income'] >= 100000) & (col['annual_income'] <= 200000), 'income_category'] = 'Medium'
	col.loc[col['annual_income'] > 200000, 'income_category'] = 'High'

df['loan_condition_int'] = np.nan  
for col in lst:
	col.loc[df['loan_condition'] == 'Good Loan', 'loan_condition_int'] = 0 
	col.loc[df['loan_condition'] == 'Bad Loan', 'loan_condition_int'] = 1

df['loan_condition_int'] = df['loan_condition_int'].astype(str) 

fig, ax = plt.subplots(2,2, figsize(14,6))

sns.violinplot(x="income_category", y="loan_amount", data=df, palette="Set2", ax=ax[0] )
sns.violinplot(x="income_category", y="loan_condition_int", data=df, palette="Set2", ax=ax[1])
sns.boxplot(x="income_category", y="emp_length_int", data=df, palette="Set2", ax=ax[2])
sns.boxplot(x="income_category", y="interest_rate", data=df, palette="Set2", ax=ax[3])

#低收入的人的贷款变为坏贷款的可能性更高

#Accessing Risks 

#债务/收入的比例是一个重要指标
#平均工作年限也是一个指标
#以州为单位进行考察

by_condition = df.groupby('addr_state')['loan_condition'].value_counts() / df.groupby('addr_state')['loan_condition'].count()		#各州好坏贷款的比例
by_emp_length = df.groupby(['region','addr_state'], as_index = False).emp_length_int.mean().sort_values(by = 'addr_state')		#以addr_state这一列的值为关键字进行排序

loan_condition_bystates = pd.crosstab(df['addr_state'], df['loan_condition'])

cross_condition = pd.crosstab(df['addr_state'], df['loan_condition'])
percentage_loan_contributor = pd.crosstab(df['addr_state'], df['loan_condition']).apply(lambda x : x / x.sum() * 100)	#各州好坏贷款数占好坏贷款总量的各比例
condition_ratio = cross_condition['Bad Loan'] / cross_condition['Good Loan']
by_dti = df.groupby(['region', 'addr_state'], as_index = False).dti.mean()		# dti : Debt to income
state_codes = sorted(states)		# 292 line 

default_ratio = condition_ratio.values.tolist() 
average_dti = by_dti['dti'].values.tolist() 
average_emp_length = by_emp_length['emp_length_int'].values.tolist() 
number_of_badloans = loan_condition_bystates['Bad Loan'].values.count() 
percentage_ofall_badloans = percentage_loan_contributor['Bad Loan'].values.tolist() 

risk_data = OrderedDict([('state_codes', state_codes),
                         ('default_ratio', default_ratio),		#Bad loan number / good loan number
                         ('badloans_amount', number_of_badloans),		#addr_state and loan_condition's crosstab's bad_loan number
                         ('percentage_of_badloans', percentage_ofall_badloans),			#each state's bad_loan's percentage of all bad_loan
                         ('average_dti', average_dti),				#each state's average debt to income
                         ('average_emp_length', average_emp_length)])			

risk_df = pd.DataFrame.from_dict(risk_data)
risk_df = risk_df.round(decimals=3)

'''
	state_codes	default_ratio	badloans_amount	percentage_of_badloans	average_dti	average_emp_length
0	AK	0.074	151	0.224	13.656	6.256
1	AL	0.097	993	1.473	17.821	6.607
2	AR	0.083	507	0.752	19.681	6.404
3	AZ	0.084	1581	2.345	19.284	5.793
4	CA	0.088	10518	15.599	18.683	5.967
'''

for col in risk_df.columns:
	risk_df[col] = risk_df[col].astype(str)

scl = [[0.0, 'rgb(202, 202, 202)'],[0.2, 'rgb(253, 205, 200)'],[0.4, 'rgb(252, 169, 161)'],\
            [0.6, 'rgb(247, 121, 108  )'],[0.8, 'rgb(232, 70, 54)'],[1.0, 'rgb(212, 31, 13)']]

risk_df['text'] = risk_df['state_codes'] + '<br>' +\
'Number of Bad Loans: ' + risk_df['badloans_amount'] + '<br>' + \
'Percentage of all Bad Loans: ' + risk_df['percentage_of_badloans'] + '%' +  '<br>' + \
'Average Debt-to-Income Ratio: ' + risk_df['average_dti'] + '<br>'+\
'Average Length of Employment: ' + risk_df['average_emp_length'] 

data = [ dict(
        type='choropleth',
        colorscale = scl,
        autocolorscale = False,
        locations = risk_df['state_codes'],
        z = risk_df['default_ratio'], 
        locationmode = 'USA-states',
        text = risk_df['text'],
        marker = dict(
            line = dict (
                color = 'rgb(255,255,255)',
                width = 2
            ) ),
        colorbar = dict(
            title = "%")
        ) ]

layout = dict(
    title = 'Lending Clubs Default Rates <br> (Analyzing Risks)',
    geo = dict(
        scope = 'usa',
        projection=dict(type='albers usa'),
        showlakes = True,
        lakecolor = 'rgb(255, 255, 255)')
)


#分析整体的风险水平，以及有多少贷款是坏的类型	(被收到的客户的信用评分等级）。
#信用评分的等级越低，投资者的风险越高

f, axxx = plt.subplots(1,2)
cmap = plt.cm.coolwarm 

by_credict_score = df.groupby(['year','grade']).loan_amount.mean()
by_credict_score.unstack().plot(legend = False, ax =axxx[0], figsize = (14,4), colormap = cmap)
axxx[0].set_title('Loans issued by Credit Score', fontsize=14)

by_inc = df.groupby(['year','grade']).interest_rate.mean() 
by_inc.unstack().plot(ax =axxx[1], figsize = (14,4), colormap = cmap) 
axxx[1].set_title('Interest Rates by Credit Score', fontsize=14)
axxx[1].legend(bbox_to_anchor=(-1.0, -0.3, 1.7, 0.1), loc=5, prop={'size':12},ncol=7, mode="expand", borderaxespad=0.)



fig = plt.figure(figsize = (16,12))
ax1 = fig.add_subplot(221)
ax2= fig.add_subplot(222)
ax3 = fig.add_subplot(212) 

cmap = plt.cm.coolwarm_r 

loans_by_region = df.groupby(['grade','loan_condition']).size()		# groupby().size() 以grade与loan_condition为分组值的各组大小
loans_by_region.unstack().plot(kind = 'bar', stacked = True, colormap = cmap, ax = ax1, grid = False)
ax1.set_title('Type of Loans by Grade', fontsize=14)

loans_by_grade = df.groupby(['sub_grade','loan_condition']).size() 
loans_by_grade.unstack().plot(kind = 'bar', stacked = True, colormap = cmap, ax = ax2, grid = False)
ax2.set_title('Type of Loans by Sub_Grade', fontsize = 14)

by_interest = df.groupby(['year','loan_condition']).size() 
by_interest.unstack().plot(ax = ax3, colormap = cmap)
ax3.set_title('Average Interest rate by Loan Condition', fontsize=14)
ax3.set_ylabel('Interest Rate (%)', fontsize=12)


#确认决定坏贷款的主要因素 (low credit score, high dti )
'''
去年的坏账种类有减少的趋势，除了迟缴（可能表明经济复苏）。

抵押贷款是房屋所有权栏目中的变量，该栏使用了被认为是坏账的贷款中最高的借款金额。

有抵押贷款的人有轻微的增加。

拥有抵押贷款的人（取决于抵押贷款中的其他因素）更可能要求
'''
numeric_variables = df.select_dtypes(exclude = ['object'])


df_correlations = df.corr() 
trace = go.Heatmap(z=df_correlations.values,
                   x=df_correlations.columns,
                   y=df_correlations.columns,
                  colorscale=[[0.0, 'rgb(165,0,38)'], 
                              [0.1111111111111111, 'rgb(215,48,39)'], 
                              [0.2222222222222222, 'rgb(244,109,67)'], 
                              [0.3333333333333333, 'rgb(253,174,97)'], 
                              [0.4444444444444444, 'rgb(254,224,144)'], 
                              [0.5555555555555556, 'rgb(224,243,248)'], 
                              [0.6666666666666666, 'rgb(171,217,233)'], 
                              [0.7777777777777778, 'rgb(116,173,209)'], 
                              [0.8888888888888888, 'rgb(69,117,180)'], 
                              [1.0, 'rgb(49,54,149)']],
            colorbar = dict(
            title = 'Level of Correlation',
            titleside = 'top',
            tickmode = 'array',
            tickvals = [-0.52,0.2,0.95],
            ticktext = ['Negative Correlation','Low Correlation','Positive Correlation'],
            ticks = 'outside'
        )
                  )

layout = {'title':"Correlation Heatmap"}	
data = [trace]

fig = dict(data = data, layout = layout)
iplot(fig, filename = 'labelled-heatmap')



title = 'Bad Loans: Loan Statuses'

labels = bad_loan # All the elements that comprise a bad loan.

#len(labels)
colors = ['rgba(236, 112, 99, 1)', 'rgba(235, 152, 78, 1)', 'rgba(52, 73, 94, 1)', 'rgba(128, 139, 150, 1)',
         'rgba(255, 87, 51, 1)', 'rgba(255, 195, 0, 1)']

mode_size = [8,8,8,8,8,8]

line_size = [2,2,2,2,2,2]

x_data = [
    sorted(df['year'].unique().tolist()),
    sorted(df['year'].unique().tolist()),
    sorted(df['year'].unique().tolist()),
    sorted(df['year'].unique().tolist()), 
    sorted(df['year'].unique().tolist()),
    sorted(df['year'].unique().tolist()),
]

# type of loans
charged_off = df['loan_amount'].loc[df['loan_status'] == 'Charged Off'].values.tolist()
defaults = df['loan_amount'].loc[df['loan_status'] == 'Default'].values.tolist()
not_credit_policy = df['loan_amount'].loc[df['loan_status'] == 'Does not meet the credit policy. Status:Charged Off'].values.tolist()
grace_period = df['loan_amount'].loc[df['loan_status'] == 'In Grace Period'].values.tolist()
short_late = df['loan_amount'].loc[df['loan_status'] == 'Late (16-30 days)'].values.tolist()
long_late = df['loan_amount'].loc[df['loan_status'] == 'Late (31-120 days)'].values.tolist()

y_data = [
    charged_off,
    defaults,
    not_credit_policy,
    grace_period,
    short_late,
    long_late,
]

p_charged_off = go.Scatter(
    x = x_data[0],
    y = y_data[0],
    name = 'A. Charged Off',
    line = dict(
        color = colors[0],
        width = 3,
        dash='dash')
)

p_defaults = go.Scatter(
    x = x_data[1],
    y = y_data[1],
    name = 'A. Defaults',
    line = dict(
        color = colors[1],
        width = 3,
        dash='dash')
)

p_credit_policy = go.Scatter(
    x = x_data[2],
    y = y_data[2],
    name = 'Not Meet C.P',
    line = dict(
        color = colors[2],
        width = 3,
        dash='dash')
)

p_graced = go.Scatter(
    x = x_data[3],
    y = y_data[3],
    name = 'A. Graced Period',
    line = dict(
        color = colors[3],
        width = 3,
        dash='dash')
)

p_short_late = go.Scatter(
    x = x_data[4],
    y = y_data[4],
    name = 'Late (16-30 days)',
    line = dict(
        color = colors[4],
        width = 3,
        dash='dash')
)

p_long_late = go.Scatter(
    x = x_data[5],
    y = y_data[5],
    name = 'Late (31-120 days)',
    line = dict(
        color = colors[5],
        width = 3,
        dash='dash')
)




data=[p_charged_off, p_defaults, p_credit_policy, p_graced, p_short_late, p_long_late]

layout = dict(title = 'Types of Bad Loans <br> (Amount Borrowed Throughout the Years)',
              xaxis = dict(title = 'Year'),
              yaxis = dict(title = 'Amount Issued'),
              )

fig = dict(data=data, layout=layout)

iplot(fig, filename='line-mode')


plt.figure(figsize = (18,18))

bad_df = df.loc[df['loan_condition'] == "Bad Loan"]
plt.subplot(211)
g = sns.boxplot(x='home_ownership', y='loan_amount', hue='loan_condition',
               data=bad_df, color='r')

g.set_xticklabels(g.get_xticklabels(),rotation=45)
g.set_xlabel("Type of Home Ownership", fontsize=12)
g.set_ylabel("Loan Amount", fontsize=12)
g.set_title("Distribution of Amount Borrowed \n by Home Ownership", fontsize=16)

plt.subplot(212)
g1 = sns.boxplot(x='year', y='loan_amount', hue='home_ownership',
               data=bad_df, palette="Set3")
g1.set_xticklabels(g1.get_xticklabels(),rotation=45)
g1.set_xlabel("Type of Home Ownership", fontsize=12)
g1.set_ylabel("Loan Amount", fontsize=12)
g1.set_title("Distribution of Amount Borrowed \n through the years", fontsize=16)


plt.subplots_adjust(hspace = 0.6, top = 0.8)



###########特征工程与神经网络应用


#copy dataframe
complete_df = df.copy() 

#处理缺失值
#新建属性列
for col in ('dti_joint', 'annual_inc_joint', 'il_util', 'mths_since_rcnt_il', 'open_acc_6m', 'open_il_6m', 'open_il_12m',
           'open_il_24m', 'inq_last_12m', 'open_rv_12m', 'open_rv_24m', 'max_bal_bc', 'all_util', 'inq_fi', 'total_cu_tl',
           'mths_since_last_record', 'mths_since_last_major_derog', 'mths_since_last_delinq', 'total_bal_il', 'tot_coll_amt',
           'tot_cur_bal', 'total_rev_hi_lim', 'revol_util', 'collections_12_mths_ex_med', 'open_acc', 'inq_last_6mths',
           'verification_status_joint', 'acc_now_delinq'):
    complete_df[col] = complete_df[col].fillna(0)

#获取上一个、下一个支付日期以及最后日期的信用额度
complete_df["next_pymnt_d"] = complete_df.groupby("region")["next_pymnt_d"].transform(lambda x: x.fillna(x.mode))		#transform内的函数用于转化原位置的数值变为最终的数值
complete_df["last_pymnt_d"] = complete_df.groupby("region")["last_pymnt_d"].transform(lambda x: x.fillna(x.mode))
complete_df["last_credit_pull_d"] = complete_df.groupby("region")["last_credit_pull_d"].transform(lambda x: x.fillna(x.mode))
complete_df["earliest_cr_line"] = complete_df.groupby("region")["earliest_cr_line"].transform(lambda x: x.fillna(x.mode))

# Get the mode on the number of accounts in which the client is delinquent
complete_df["pub_rec"] = complete_df.groupby("region")["pub_rec"].transform(lambda x: x.fillna(x.median()))

# Get the mean of the annual income depending in the region the client is located.
complete_df["annual_income"] = complete_df.groupby("region")["annual_income"].transform(lambda x: x.fillna(x.mean()))

# Get the mode of the  total number of credit lines the borrower has 	信贷额度
complete_df["total_acc"] = complete_df.groupby("region")["total_acc"].transform(lambda x: x.fillna(x.median()))

# Mode of credit delinquencies in the past two years.		信用违约
complete_df["delinq_2yrs"] = complete_df.groupby("region")["delinq_2yrs"].transform(lambda x: x.fillna(x.mean()))

#去掉冗余属性
complete_df.drop(['issue_d', 'income_category', 'region', 'year', 'emp_length', 'loan_condition',
                 'earliest_cr_line', 'last_pymnt_d', 'next_pymnt_d', 'last_credit_pull_d', 
                 'verification_status_joint', 'emp_length_int', 'total_rec_prncp', 'funded_amount', 'investor_funds', 
                 'sub_grade', 'complete_date', 'loan_status', 'interest_payments', 
                 'initial_list_status', 'out_prncp', 'out_prncp_inv', 'total_pymnt',
               'total_pymnt_inv', 'total_rec_int', 'total_rec_late_fee', 'recoveries',
               'collection_recovery_fee', 'last_pymnt_amnt',
               'collections_12_mths_ex_med', 'mths_since_last_major_derog',
               'policy_code', 'application_type', 'annual_inc_joint', 'dti_joint',
               'acc_now_delinq', 'tot_coll_amt', 'tot_cur_bal', 'open_acc_6m',
               'open_il_6m', 'open_il_12m', 'open_il_24m', 'mths_since_rcnt_il',
               'total_bal_il', 'il_util', 'open_rv_12m', 'open_rv_24m', 'max_bal_bc',
               'all_util', 'total_rev_hi_lim', 'inq_fi', 'total_cu_tl', 'inq_last_12m'], axis=1, inplace=True)

#complete_df.isnull().sum().max() 

complete_df['loan_condition_int'].value_counts() / len(complete_df['loan_condition_int']) * 100 	#好坏贷款



#数据划分

stratified = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 42)		#划分数据集的函数 n_splits是train/test的对数 test_size是划分后训练数据所占比例 ramdom_state是将样本随机打乱
for train_set, test_set in stratified.split(complete_df, complete_df["loan_condition_int"]):		#doubt
    stratified_train = complete_df.loc[train_set]
    stratified_test = complete_df.loc[test_set]

'''
print('Train set ratio \n', stratified_train["loan_condition_int"].value_counts()/len(df))
print('Test set ratio \n', stratified_test["loan_condition_int"].value_counts()/len(df))
'''

train_df = stratified_train
test_df = stratified_test

# Shuffle the data
train_df = train_df.sample(frac=1).reset_index(drop=True)
test_df = test_df.sample(frac=1).reset_index(drop=True)

# Train set (Normal training dataset)
X_train = train_df.drop('loan_condition_int', axis=1)
y_train = train_df['loan_condition_int']

# Test Dataset
X_test = test_df.drop('loan_condition_int', axis=1)
y_test = test_df['loan_condition_int']

class CategoricalEncoder(BaseEstimator, TransformerMixin):
	#输入是一个用分类离散的值来代表特征的矩阵
	 """Encode categorical features as a numeric array.
    The input to this transformer should be a matrix of integers or strings,
    denoting the values taken on by categorical (discrete) features.
    The features can be encoded using a one-hot aka one-of-K scheme
    (``encoding='onehot'``, the default) or converted to ordinal integers
    (``encoding='ordinal'``).
    This encoding is needed for feeding categorical data to many scikit-learn
    estimators, notably linear models and SVMs with the standard kernels.
    Read more in the :ref:`User Guide <preprocessing_categorical_features>`.
    Parameters
    ----------
    encoding : str, 'onehot', 'onehot-dense' or 'ordinal'
        The type of encoding to use (default is 'onehot'):
        - 'onehot': encode the features using a one-hot aka one-of-K scheme
          (or also called 'dummy' encoding). This creates a binary column for
          each category and returns a sparse matrix.
        - 'onehot-dense': the same as 'onehot' but returns a dense array
          instead of a sparse matrix.
        - 'ordinal': encode the features as ordinal integers. This results in
          a single column of integers (0 to n_categories - 1) per feature.
    categories : 'auto' or a list of lists/arrays of values.
        Categories (unique values) per feature:
        - 'auto' : Determine categories automatically from the training data.
        - list : ``categories[i]`` holds the categories expected in the ith
          column. The passed categories are sorted before encoding the data
          (used categories can be found in the ``categories_`` attribute).
    dtype : number type, default np.float64
        Desired dtype of output.
    handle_unknown : 'error' (default) or 'ignore'
        Whether to raise an error or ignore if a unknown categorical feature is
        present during transform (default is to raise). When this is parameter
        is set to 'ignore' and an unknown category is encountered during
        transform, the resulting one-hot encoded columns for this feature
        will be all zeros.
        Ignoring unknown categories is not supported for
        ``encoding='ordinal'``.
    Attributes
    ----------
    categories_ : list of arrays
        The categories of each feature determined during fitting. When
        categories were specified manually, this holds the sorted categories
        (in order corresponding with output of `transform`).
    Examples
    --------
    Given a dataset with three features and two samples, we let the encoder
    find the maximum value per feature and transform the data to a binary
    one-hot encoding.
    >>> from sklearn.preprocessing import CategoricalEncoder
    >>> enc = CategoricalEncoder(handle_unknown='ignore')
    >>> enc.fit([[0, 0, 3], [1, 1, 0], [0, 2, 1], [1, 0, 2]])
    ... # doctest: +ELLIPSIS
    CategoricalEncoder(categories='auto', dtype=<... 'numpy.float64'>,
              encoding='onehot', handle_unknown='ignore')
    >>> enc.transform([[0, 1, 1], [1, 0, 4]]).toarray()
    array([[ 1.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  0.],
           [ 0.,  1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.]])
    See also
    --------
    sklearn.preprocessing.OneHotEncoder : performs a one-hot encoding of
      integer ordinal features. The ``OneHotEncoder assumes`` that input
      features take on values in the range ``[0, max(feature)]`` instead of
      using the unique values.
    sklearn.feature_extraction.DictVectorizer : performs a one-hot encoding of
      dictionary items (also handles string-valued features).
    sklearn.feature_extraction.FeatureHasher : performs an approximate one-hot
      encoding of dictionary items or strings.
    """

    '''
	Onehot code 用n个比特的值来表示n个状态
	假如有三种颜色特征：红、黄、蓝。只是想让机器区分它们，并无大小比较之意。
	因为有三种颜色状态，所以就有3个比特。即红色：1 0 0 ，黄色: 0 1 0，蓝色：0 0 1 
	如此一来每两个向量之间的距离都是根号2，在向量空间距离都相等，所以这样不会出现偏序性
	基本不会影响基于向量空间度量算法的效果。

	编码方法：读取数据查看每个特征维度上各有几个不同的值并从小到大排列(int),然后对应的编码方式是采用长度为不同值个数的比特数且1从高位到低位依次排列的编码

	常用于解决类别数据的离散值问题，但当类别数据种类过多时，需要用PCA来减少维度
	当采用基于树的方法处理数据时，不需要进行归一化(随机森林，bagging, boosting)

    '''


    def __init__(self,encoding = 'onebot', categories = 'auto', dtype = np.float64, handle_unknown = 'error'):
    	self.encoding = encoding 
    	self.categories = categories 
    	self.dtype = dtype 
    	self.handle_unknown = handle_unknown 

    def fit(self, X, y = None):		#X is array with data to determine the categories of each feature
    	if self.encoding not in ['onebot','onebot-dense','ordinal']:
    		template = ("encoding should be either 'onehot', 'onehot-dense' "
                        "or 'ordinal', got %s")
    		raise ValueError(template % self.handle_unknown )

    	if self.handle_unknown not in ['error', 'ignore']:
    		template = ("handle_unknown should be either 'error' or "
                        "'ignore', got %s")
            raise ValueError(template % self.handle_unknown)

        if self.encoding == 'ordinal' and self.handle_unknown == 'ignore':
            raise ValueError("handle_unknown='ignore' is not supported for"
                             " encoding='ordinal'")

        X = check_array(X,dtype = np.object, accept_sparse = 'csc', copy = True)		#doubts
        n_samples, n_features = X.shape 
        self._label_encoders_ = [LabelEncoder() for _ in range(n_features)]	#LabelEncoder对

        for i in range(n_features):
        	le = self._label_encoders_[i] 		#第i个属性的解码方法
        	Xi = X[:,i]			#所有samples的第i个属性列
        	if self.categories == 'auto':
        		le.fit(Xi)		#读取Xi的数据来学习如何对该属性数据进行编码
        	else:
        		valid_mask = np.in1d(Xi, self.categories[i])	#in1d(a,b) 查找序列a与b之间相同的值并返回bool/false
        		if not np.all(valid_mask):		#np.all(s)表示对可迭代对象s中的各个对象进行比较，若都相等则返回True
        			if self.handle_unknown == 'error':
        				diff = np.unique(Xi[~valie_mask])	#求set
        				msg = ("Found unknown categories {0} in column {1}"
                               " during fit".format(diff, i))
        				raise ValueError(msg)
        		le.classes_ = np.array(np.sort(self.categories[i]))	#已经排序的类型列表

        		#le_classes_ 是表示LabelEncoder.fit(data)之后的类型值，transform(data)将data值变为对应的类型值

        self.categories_ = [le.classes_ for le in self._label_encoders_]

        return self 

    def transform(self,X):
    	X = check_array(X,accept_sparse = 'csc', dtype = np.object, copy = True)
    	n_samples, n_features = X.shape 
    	X_int = np.zeros_like(X,dtype= np.int)
    	X_mask = np.ones_like(X,dtype = np.bool)

    	for i in range(n_features):
    		valid_mask = np.in1d(X[:,i], self.categories_[i])

    		if not np.all(valid_mask):
    			if self.handle_unknown == 'error':
    				diff = np.unique(X[~valid_mask,i])
    				msg = ("Found unknown categories {0} in column {1}"
                           " during transform".format(diff, i))
                    	raise ValueError(msg)

            	else :
            		X_mask[:,i]= valid_mask
            		X[:,i][~valid_mask] = self.categories_[i][0]		#~valid_mask取反
        	X_int[:,i]= self._label_encoders_[i].transform(X[:,i])

        if self.encoding == 'ordinal':
        	return X_int.astype(self.dtype, copy = False)

        mask = X_mask.ravel()		#返回X_mask数组的视图（引用）
        n_values = [cats.shape[0] for cats in self.categories_]
        n_values = np.array([0] + n_values)
        indices = np.cumsum(n_values)		#按轴向计算和的函数

        column_indices = (X_int + indices[:-1]).ravel()[mask]	#
        row_indices = np.repeat(np.arange(n_samples, dtype = np.int32), n_features)[mask]
        data = np.ones(n_samples * n_features)[mask]

        out = sparse.csc_matrix((data, (row_indices, column_indices)),
                                shape=(n_samples, indices[-1]),			#indices记录了(a,b) b是属性个数		data[k] = out[row_indices[k],column_indices[k]]
                                dtype=self.dtype).tocsr()		#构建稀疏矩阵	Convert this matrix to Compressed Sparse Row format.

        if self.coding == 'onehot-dense':
        	return out.toarray()
        else:
        	return out 


class DataFrameSelector(BaseEstimator,TransformerMixin):		#由于Scikit-Learn不能处理DataFrame，故写这个类来选择数字类型或目录类型的列
	def __init__(self, attribute_name):
		self.attribute_name = attribute_name
	def fit(self,X,y = None):
		return self 
	def transform(self,X):
		return X[self.attribute_name]


numeric = X_train.select_dtypes(exclude = ["object"])
categorical = X_train.select_dtypes(["object"])

numeric_pipeline = Pipeline([				#流水线处理
    ('selector', DataFrameSelector(numeric.columns.tolist())),
    ('scaler', StandardScaler()),
])

categorical_pipeline = Pipeline([
    ('selector', DataFrameSelector(categorical.columns.tolist())), 
    ('encoder', CategoricalEncoder(encoding="onehot-dense")),
])

combined_pipeline = FeatureUnion(transformer_list=[
    ('numeric_pipeline', numeric_pipeline),
    ('categorical_pipeline', categorical_pipeline)
])
	
X_train = combined_pipeline.fit_transform(X_train)			#fit_transform()  先fit data 然后 transform it to 对应的类型标签(int)
X_test = combined_pipeline.fit_transform(X_test)


log_reg = LogisticRegression() 
log_reg.fit(X_train, y_train)

normal_ypred = log_reg.predict(X_test) 
print(accuracy_score(y_test, normal_ypred))

def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)
    
def fetch_batch(epoch, batch_index, batch_size, instances=X_train.shape[0]):
    np.random.seed(epoch * n_batches + batch_index)
    indices = np.random.randint(instances, size=batch_size)  
    X_batch = X_train[indices] 
    y_batch = y_train[indices]
    return X_batch, y_batch
    
reset_graph()


# Directory to access tensorboard
# now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
# root_logdir = "tf_logs"
# LOGDIR = "{}/run-{}/".format(root_logdir, now)

# Neural Network with TensorFlow 

n_inputs = X_train.shape[1]
hidden1_amount = 66
hidden2_amount = 66
n_outputs = 2

# Placeholders
X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int32, shape=(None), name="y")

# Architectural Structure
with tf.name_scope('dnn'):
    hidden1 = tf.layers.dense(X, hidden1_amount, activation=tf.nn.relu, name="first_layer")
    hidden2 = tf.layers.dense(hidden1, hidden2_amount, activation=tf.nn.relu, name="second_layer")
    logits = tf.layers.dense(hidden2, n_outputs, name="outputs")
    
# Loss Functions
with tf.name_scope("loss"):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,
                                                                  logits=logits)
    loss = tf.reduce_mean(cross_entropy, name="loss")
    
# optimizer
learning_rate = 0.01

with tf.name_scope("train"):
    optimizer = tf.train.AdamOptimizer(learning_rate)
    best_op = optimizer.minimize(loss)
    
# Evaluating
with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    
init = tf.global_variables_initializer()


batch_size = 250
n_batches = int(np.ceil(X_train.shape[0]/ batch_size))
n_epochs = 10

with tf.Session() as sess:
    init.run()
    
    for epoch in range(n_epochs):
        for batch_index in range(n_batches):
            X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
            sess.run(best_op, feed_dict={X: X_batch, y: y_batch})
        loss_val, acc_val = sess.run([loss, accuracy], feed_dict={X: X_test, y: y_test})
        print(epoch+1, "Loss:{:.5f}\t Accuracy:{:.3f}%".format(loss_val, acc_val * 100))













































