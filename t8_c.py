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
import xgboost as xgb
import random
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

#使用回归来预测（针对测试集）
def regression_model(reg,train_data,train_y,test_data,input_feature):
    reg.fit(train_data.as_matrix(columns=input_feature),train_y)
    predict_y=reg.predict(test_data.as_matrix(columns=input_feature))
    return reg,predict_y

if __name__=='__main__':
    #读取训练集与测试集
    train_data = pd.read_pickle('train_data_all.pickle')
    train_y = pd.read_pickle('train_y_all.pickle')
    test_data=pd.read_pickle('test_data.pickle')
    # 读取特征选择后的特征
    filter_feature = pd.read_pickle('filter_feature_top200.pickle')
    filter_feature = filter_feature.tolist()

    # # 使用随机森林回归
    # rf_gre = RandomForestRegressor()
    # rf_gre, predict_y_rf = regression_model(rf_gre, train_data, train_y, test_data, filter_feature)

    # 做epoch次随机森林回归
    epoch = 5
    # 构建一个存放结果的矩阵
    res =np.zeros((56283,epoch))
    for i in range(epoch):
        #使用调整参数后的随机森林回归
        xgbo = xgb.XGBRegressor(max_depth=60, learning_rate=0.1, n_estimators=500, silent=False, objective='reg:gamma')
        xgbo,predict_y_rf=regression_model(xgbo,train_data, train_y, test_data, filter_feature)
        res[:,i]=predict_y_rf.reshape(56283)
    #构建一个存放平均值的矩阵
    avg=np.zeros((56283,1))
    for i in range(56283):
        avg[i]=res[i].mean()
    #将预测结果导出
    np.savetxt('avg_predict_result.csv',avg,delimiter=',')
    np.savetxt('all_predict_result.csv',res,delimiter=',')



