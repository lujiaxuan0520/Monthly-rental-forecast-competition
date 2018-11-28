#coding=gbk
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib as mpl
import random
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

if __name__=='__main__':
    #读取训练集与测试集
    train_data = pd.read_pickle('train_data_all.pickle')
    train_y = pd.read_pickle('train_y_all.pickle')
    test_data=pd.read_pickle('test_data.pickle')
    # 读取特征选择后的特征
    filter_feature = pd.read_pickle('filter_feature_top100.pickle')
    filter_feature = filter_feature.tolist()

    train=train_data.as_matrix(columns=filter_feature)
    test=test_data.as_matrix(columns=filter_feature)

    print('f')