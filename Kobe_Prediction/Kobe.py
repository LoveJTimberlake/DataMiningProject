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

fig, ax1 = plt.subplots() 

#求出Kobe在各个区域的命中率
Area_Shoot = df.copy() 
Area_GoalTotal = Area_Shoot.loc[(Area_Shoot['shot_made_flag'] == 1) or (Area_Shoot['shot_made_flag'] == 0)]
Area_Goal = Area_GoalTotal.groupby('shot_zone_area')
Area_Goal.plot.pie( autopct = '%1.1f%%', ax = ax1, shadow = True)

plt.show()






