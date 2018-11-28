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
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import  GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn import neighbors
import matplotlib as mpl
import xgboost as xgb
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

#使用回归来预测
def regression_model(reg,train_data,train_y,val_data,val_y,input_feature):
    reg.fit(train_data.as_matrix(columns=input_feature),train_y)
    predict_y=reg.predict(val_data.as_matrix(columns=input_feature))
    RMSE=mean_squared_error(val_y,predict_y)**0.5
    return reg,RMSE,predict_y

if __name__=='__main__':
    # 读取训练集与验证集
    train_data = pd.read_pickle('train_data.pickle')
    val_data = pd.read_pickle('val_data.pickle')
    train_y = pd.read_pickle('train_y.pickle')
    val_y = pd.read_pickle('val_y.pickle')
    #读取特征选择后的特征,filter_feature仅仅是一个array
    filter_feature = pd.read_pickle('filter_feature_top200.pickle')
    filter_feature = filter_feature.tolist()
    # filter_feature=['房屋面积','总楼层','距离','区#12','小区房屋出租数量','区#10','卧室数量','位置#43','卫的数量','位置#137','区#6','厅的数量',
    #                 '位置#78','地铁站点#72.0','地铁线路#5.0','楼层#1','楼层#2','位置#148','楼层#0','房屋朝向#2','房屋朝向#5','区#11','区#1',
    #                 '房屋朝向#1','位置#102','地铁站点#107.0','位置#71','位置#9','房屋朝向#4','房屋朝向#7']

    np.random.seed(21)
    # #使用岭回归
    # ridge=Ridge()
    # ridge,RMSE,predict_y_ridge=regression_model(ridge,train_data,train_y,val_data,val_y,filter_feature)
    # print('RMSE for ridge_regression: %s'%RMSE)
    # #使用svm回归
    # svr=svm.SVR()
    # svr,RMSE,predict_y_svm=regression_model(svr, train_data, train_y, val_data, val_y, filter_feature)
    # print('RMSE for SVM_regression: %s' % RMSE)
    # #使用knn回归
    # knn = neighbors.KNeighborsRegressor()
    # knn, RMSE, predict_y_knn = regression_model(knn, train_data, train_y, val_data, val_y, filter_feature)
    # print('RMSE for KNN_regression: %s' % RMSE)
    # #使用决策树回归
    # dt_gre=DecisionTreeRegressor()
    # dt_gre, RMSE, predict_y_dt = regression_model(dt_gre, train_data, train_y, val_data, val_y, filter_feature)
    # print('RMSE for DecisionTree_regression: %s' % RMSE)

    #集成方法
    # #使用随机森林回归
    # rf_gre=RandomForestRegressor(n_estimators=100)#括号里调参数,n_estimators表示决策树的个数，100-200为宜.参数还可以有max_features=2, min_samples_split=4, min_samples_leaf=2
    # rf_gre, RMSE, predict_y_rf = regression_model(rf_gre, train_data, train_y, val_data, val_y, filter_feature)
    # print('RMSE for RandomForest_regression: %s' % RMSE)
    # #使用GBRT回归
    # gbrt = GradientBoostingRegressor(n_estimators=100)#参数中还可以learning_rate=0.1, max_depth=1, random_state=0, loss='quantile'
    # gbrt, RMSE, predict_y_gbrt = regression_model(gbrt, train_data, train_y, val_data, val_y, filter_feature)
    # print('RMSE for GBRT_regression: %s' % RMSE)
    # #使用Adaboost回归，基于决策树来做
    # ada_tree_backing = DecisionTreeRegressor()
    # ada =AdaBoostRegressor(ada_tree_backing,n_estimators=100,learning_rate=0.01, loss='square')
    # ada, RMSE, predict_y_ada = regression_model(ada, train_data, train_y, val_data, val_y, filter_feature)
    # print('RMSE for Adaboost_regression: %s' % RMSE)
    # 使用XGBoost回归,XGBoost的速度会比之前的集成方法快很多。learning_rate ＝ 0.1 或更小，tree_depth ＝ 2～8；
    # 可以调的超参数组合：树的个数和大小 (n_estimators and max_depth)；学习率和树的个数 (learning_rate and n_estimators)；行列的 subsampling rates (subsample, colsample_bytree and colsample_bylevel)
    xgbo = xgb.XGBRegressor(max_depth=60, learning_rate=0.1, n_estimators=100, silent=False, objective='reg:gamma')
    xgbo, RMSE, predict_y_xgbo = regression_model(xgbo, train_data, train_y, val_data, val_y, filter_feature)
    print('RMSE for XGBoost_regression: %s' % RMSE)


    # #调节随机森林的超参数
    # np.random.seed(21)
    # tune_params={'n_estimators':[50,80,120,150,180,200],
    #             'max_features':['auto','sqrt'],
    #              'max_depth':[50,80,100,120,150]}
    # rf_reg=RandomForestRegressor()
    # grid_search=GridSearchCV(estimator=rf_reg,param_grid=tune_params,verbose=1,n_jobs=-1)
    # grid_search.fit(train_data,train_y)
    # print(grid_search.best_params_)
