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


df = pd.read_csv(r"C:\Users\msi\Desktop\Kobe Prediction\data.csv", low_memory = False)

'''
print(df.head())
print(df.info())
'''

fig, ax = plt.subplots()

Action_Types = df["action_type"].values 
Shot_Types = df["combined_shot_type"].values 

df["action_type"].value_counts().plot.pie( autopct = '%1.1f%%', ax = ax, shadow = True)
ax.set_title("Kobe's Action Distribution")

'''
Wrong Code with problems
fig, ax1 = plt.subplots() 

df['Area_Shoot_Total'] = 1 
Area_Shoot = df.groupby(['shot_zone_area','shot_made_flag'], as_index = False).count()
Area_Goal = pd.DataFrame(Area_Shoot) 
Area_Goal['Area_Shoot_Percent'] = np.nan
print(Area_Goal.groupby('shot_zone_area')['Area_Shoot_Total'].sum())
Area_Goal['Area_Shoot_Percent'] = Area_Goal['Area_Shoot_Total'] / Area_Goal.groupby(['shot_zone_area'])['Area_Shoot_Total'].sum()
#Area_Goal.apply(lambda x : round(x,2))
print(Area_Goal)
'''

'''
#求出Kobe在各个区域的命中率
Area_Shoot = df.copy() 
Area_Shoot['Goal_Percent'] = 1
Area_GoalTotal = Area_Shoot.loc[(Area_Shoot['shot_made_flag'] == float(1)) | (Area_Shoot['shot_made_flag'] == float(0))]
Area_Goal= Area_GoalTotal.groupby(['shot_zone_area','shot_made_flag'])['shot_made_flag'].count()
t = pd.DataFrame(Area_Goal)

t = t.rename(columns={'shot_made_flag':'Shot_Counts'})
t.index.droplevel()
t['Goal_Percent'] = np.nan 
t['Goal_Percent'] = t['Shot_Counts'] / t.groupby(['shot_zone_area'])['Shot_Counts'].sum() 
t['Goal_Percent'].apply( lambda x : round(x,2))
'''
'''
print(t)
                                      Shot_Counts  Goal_Percent
shot_zone_area        shot_made_flag
Back Court(BC)        0.0                      71      0.986111
                      1.0                       1      0.013889
Center(C)             0.0                    5356      0.474444
                      1.0                    5933      0.525556
Left Side Center(LC)  0.0                    2149      0.638823
                      1.0                    1215      0.361177
Left Side(L)          0.0                    1889      0.603129
                      1.0                    1243      0.396871
Right Side Center(RC) 0.0                    2458      0.617433
                      1.0                    1523      0.382567
Right Side(R)         0.0                    2309      0.598342
                      1.0                    1550      0.401658

print(t.index)
MultiIndex(levels=[['Back Court(BC)', 'Center(C)', 'Left Side Center(LC)', 'Left Side(L)', 'Right Side Center(RC)', 'Right Side(R)'], [0.0, 1.0]],
           labels=[[0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5], [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]],
           names=['shot_zone_area', 'shot_made_flag'])
'''



#plt.show()


data = df[df['shot_made_flag'].notnull()].reset_index()
#print(data.head())

data['Standard_Game_Date'] = pd.to_datetime(data['game_date'])
data['Week'] = data['Standard_Game_Date'].dt.dayofweek 
data['Year'] = data['Standard_Game_Date'].dt.dayofyear

data['SecondsFromPeriodEnd'] = 60 * data['minutes_remaining'] + data['seconds_remaining']
data['SecondsFromPeriodStart'] = 12 * 60 - data['SecondsFromPeriodEnd']
data['SecondsFromGameStart'] = (data['period']-1) * 12 * 60 + data['SecondsFromPeriodStart']

#print(data.loc[:10, ['period','SecondsFromPeriodStart','SecondsFromPeriodEnd','SecondsFromGameStart']])

#Location and shots 

#http://scikit-learn.org/stable/modules/mixture.html
numGaussians = 13 
gaussianMixtureModel = mixture.GaussianMixture(n_components = numGaussians, covariance_type = 'full', init_params = 'kmeans',n_init = 50, verbose = 0, random_state = 5)
gaussianMixtureModel.fit(data.loc[:,['loc_x','loc_y']])

data['ShotLocationCluster'] = gaussianMixtureModel.predict(data.loc[:,['loc_x','loc_y']])
print(data.head())

def draw_court(ax = None, color = 'black', lw = 2, outer_lines = False):
	if ax is None:
		ax = plt.gca() 

	hoop = Circle((0,0), radius = 7.5, linewidth = lw, color = color, fill = False)
	backboard = Rectangle((-30,-7.5), 60, -1, linewidth = lw, color = color)
	outer_box = Rectangle((-80,-47.5),160,190, linewidth = lw, color = color, fill = False)
	inner_box = Rectangle((-60,-47.5), 120, 190, linewidth = lw, color = color,fill = False)
	top_free_throw = Arc((0,142.5), 120,120,theta1 = 0, theta2 = 180, linewidth = lw, color = color, fill=False)
	bottom_free_throw = Arc((0,0),80,80, theta1 = 0, theta2 = 180, linewidth = lw, color = color)
	restricted = Arc((0,0), 80,80, theta1 = 0, theta2 = 180, linewidth = lw, color = color)

	corner_three_a = Rectangle((-220,-47.5),0,140,linewidth = lw, color = color)
	corner_three_b = Rectangle((-220,-47.5), 0, 140, linewidth = lw, color = color)
	three_arc = Arc((0,0),475,475, theta1 = 22, theta2 = 158, linewidth = lw, color = color)
	center_outer_arc = Arc((0,422.5), 120,120, theta1 = 180, theta2 = 0, linewidth = lw, color = color)
	center_inner_arc = Arc((0,422.5), 40,40, theta1 = 180, theta2 = 0, linewidth = lw, color = color)
	court_elements = [hoop, backboard, outer_box, inner_box, top_free_throw,bottom_free_throw, restricted, corner_three_a,corner_three_b, three_arc, center_outer_arc,center_inner_arc]

	if(outer_lines):
		outer_lines = Rectangle((-250,-47.5), 500, 470, linewidth = lw, color = color, fill = False)
		court_elements.append(outer_lines)

	for element in court_elements:
		ax.add_patch(element)

	return ax 


def Draw2DGaussians(gaussianMixtureModel,ellipseColors, ellipseTextMessages):
	fig , h = plt.subplots() 
	for i,(mean, covarianceMatrix) in enumerate(zip(gaussianMixtureModel.means_ , gaussianMixtureModel.covariances_ )):
		v,w = np.linalg.eigh(covarianceMatrix)		#计算矩阵的特征值v与特征向量w
		v = 2.5 * np.sqrt(v) #用标准偏差来代替标准方差

		u = w[0] / np.linalg.norm(w[0])	#第二范数
		angle = np.arctan(u[1]/u[0])	#两个特征向量之间的夹角
		angle = angle * 180 / np.pi;	#转换维度
		currEllipse = mpl.patches.Ellipse(mean,v[0], v[1], 180 + angle, color = ellipseColors[i])
		currEllipse.set_alpha(0.5)
		h.add_artist(currEllipse)
		h.text(mean[0] + 7, mean[1] -1 , ellipseTextMessages[i], fontsize = 13, color = 'blue')


plt.rcParams['figure.figsize'] = (13,10) 
plt.rcParams['font.size'] = 15 

ellipseTextMessages = [str(100*gaussianMixtureModel.weights_[x])[:4] + '%' for x in range(numGaussians)]
ellipseColors = ['red','green','purple','cyan','magenta','yellow','blue','orange','silver','maroon','lime','olive','brown','darkblue']
Draw2DGaussians(gaussianMixtureModel, ellipseColors,ellipseTextMessages)
draw_court(outer_lines = True)
plt.ylim(-60,440)
plt.xlim(270,-270)
plt.title('shot attempts')
plt.show()
#We can see that Kobe is making more attempts from the left side of the court (or right side from his point of view). this is probably because he's right handed.

#计算每个投篮点类的准确度
#先计算每个类的总投篮次数
VariableCategories = data['ShotLocationCluster'].value_counts().index.tolist()
clusterAccuracy = {} 
for category in VariableCategories:
	ShotsAttempted = np.array(data['ShotLocationCluster'] == category).sum()
	
























