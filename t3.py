#coding=gbk
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def mapping(s):
    if s=="东":
        return 0
    if s=="南":
        return 1
    if s=="西":
        return 2
    if s=="北":
        return 3
    if s=="东南":
        return 4
    if s=="东北":
        return 5
    if s=="西南":
        return 6
    if s=="西北":
        return 7

if __name__=='__main__':
    columns = ['时间','小区名','小区房屋出租数量','楼层','总楼层','房屋面积','房屋朝向'
               ,'卧室数量','厅的数量','卫的数量','区','位置','地铁线路',
               '地铁站点','距离','月租金']
    train=pd.read_csv('./train1.csv',names=columns,encoding='gbk')

    arr=train['房屋朝向']
    #针对房屋朝向的字符串数据进行分类，分成8类：东 南 西 北 东南 东北 西南 西北
    #保存在数组directions中
    directions=np.zeros((arr.size,8))

    for item in range(arr.size):
        str=arr[item]
        x=str.split()
        for i in x:
            index=mapping(i)
            directions[item][index]=1
    np.savetxt('train_1_directions.csv',directions,delimiter=',')


