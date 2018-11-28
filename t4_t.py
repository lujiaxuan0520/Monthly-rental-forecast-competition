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
test=pd.read_csv('./test2.csv',names=columns,encoding='gbk')

#对缺失的“地铁线路”和“地铁站点”填充0，对缺失的“距离”填充1
test=test.fillna({'地铁线路':0,'地铁站点':0,'距离':1})

#删除“小区名”字段
test.drop('小区名',axis=1,inplace=True)

#对分类变量进行one-hot编码（dummy编码），房屋朝向已经转为one-hot形式，无需再做
categorial_cols=['时间','楼层','区','位置','地铁线路','地铁站点']
for col in categorial_cols:
    dummies=pd.get_dummies(test[col],drop_first=False)
    dummies=dummies.add_prefix("{}#".format(col))
    test.drop(col,axis=1,inplace=True)
    test=test.join(dummies)

test.to_pickle('test_data.pickle')

print(test.columns.size)