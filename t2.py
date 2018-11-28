#coding=gbk
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

columns = ['时间','小区名','小区房屋出租数量','楼层','总楼层','房屋面积'
           ,'房屋朝向#1','房屋朝向#2','房屋朝向#3','房屋朝向#4','房屋朝向#5','房屋朝向#6','房屋朝向#7','房屋朝向#8'
           ,'卧室数量','厅的数量','卫的数量','区','位置','地铁线路',
           '地铁站点','距离','月租金']
train=pd.read_csv('./train2.csv',names=columns,encoding='gbk')

#对缺失的“地铁线路”和“地铁站点”填充0，对缺失的“距离”填充1
train=train.fillna({'地铁线路':0,'地铁站点':0,'距离':1})

#删除“小区名”字段
train.drop('小区名',axis=1,inplace=True)

#训练集各字段信息查看
#print(train.info())
#print(train.describe())

# #绘制月租金分布图
# plt.figure(figsize=(12,6))
# plt.subplot(211)
# plt.title('月租金分布图')
# sns.distplot(train['月租金'])
# plt.subplot(212)
# plt.scatter(range(train.shape[0]),np.sort(train['月租金'].values))
# plt.show()

#绘制各个特征的分布柱状图
train.hist(figsize=(20,15),bins=50,grid=False)
plt.show()







