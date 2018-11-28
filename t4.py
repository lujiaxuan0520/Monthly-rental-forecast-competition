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

# #绘制各个特征的分布柱状图
# train.hist(figsize=(20,15),bins=50,grid=False)
# plt.show()

#连续变量
continuous_cols=['小区房屋出租数量','总楼层','房屋面积','卧室数量','厅的数量',
                 '卫的数量','距离']
# #计算连续变量与月租金的皮尔逊相关系数，并且进行排序，绘制图像
# for col in continuous_cols:
#     sns.jointplot(x=col,y='月租金',data=train,alpha=0.3,size=4)
# plt.figure(figsize=(12,6))
# train.corr()['月租金'][continuous_cols].sort_values(ascending=False).plot(
#     'barh',figsize=(12,6),title='月租金与连续变量的相关性'
# )
# plt.show()

#对分类变量进行one-hot编码（dummy编码），房屋朝向已经转为one-hot形式，无需再做
categorial_cols=['时间','楼层','区','位置','地铁线路','地铁站点']
for col in categorial_cols:
    dummies=pd.get_dummies(train[col],drop_first=False)
    dummies=dummies.add_prefix("{}#".format(col))
    train.drop(col,axis=1,inplace=True)
    train=train.join(dummies)
# #查看一下进行dummy编码以后的数据集各字段信息
#     print(train.info())

#此处可以进行数据的归一化处理（待优化）

#以8:2分割训练集与验证集
from sklearn.cross_validation import train_test_split
np.random.seed(21)
target=train['月租金']
train.drop('月租金',axis=1,inplace=True)
train_data,val_data,train_y,val_y=train_test_split(
    train,target,train_size=0.8,random_state=21
)

#保存训练集与验证集
train_data.to_pickle('train_data_all.pickle')
val_data.to_pickle('val_data.pickle')
train_y.to_pickle('train_y_all.pickle')
val_y.to_pickle('val_y.pickle')

print(train_data.columns.size)





