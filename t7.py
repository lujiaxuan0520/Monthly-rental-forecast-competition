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
from sklearn.ensemble import RandomForestRegressor
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

#使用回归来预测（针对测试集）
def regression_model(reg,train_data,train_y,test_data,input_feature):
    reg.fit(train_data.as_matrix(columns=input_feature),train_y)
    predict_y=reg.predict(test_data.as_matrix(columns=input_feature))
    return reg,predict_y

if __name__=='__main__':
    #读取训练集与测试集
    train_data = pd.read_pickle('train_data.pickle')
    train_y = pd.read_pickle('train_y.pickle')
    test_data=pd.read_pickle('test_data.pickle')
    # 读取特征选择后的特征
    filter_feature = pd.read_pickle('filter_feature.pickle')
    filter_feature = filter_feature.tolist()

    # # 使用随机森林回归
    # rf_gre = RandomForestRegressor()
    # rf_gre, predict_y_rf = regression_model(rf_gre, train_data, train_y, test_data, filter_feature)

    #使用调整参数后的随机森林回归
    rf_reg=RandomForestRegressor(n_estimators=40,max_features='auto',max_depth=50)
    rf_reg,predict_y_rf=regression_model(rf_reg,train_data, train_y, test_data, filter_feature)

    #将预测结果导出
    np.savetxt('predict_result.txt',predict_y_rf)


