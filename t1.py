#coding=gbk
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

columns = ['时间','小区名','小区房屋出租数量','楼层','总楼层','房屋面积','房屋朝向',
           '居住状态','卧室数量','厅的数量','卫的数量','出租方式','区','位置','地铁线路',
           '地铁站点','距离','装修情况','月租金']
train=pd.read_csv('./train.csv',names=columns,encoding='gbk')
#原始训练集各字段信息查看
print(train.info())

