from keras import Sequential, models
from keras.layers.core import Dense, Dropout, Activation
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error



train_data = pd.read_pickle('train_data.pickle')
val_data = pd.read_pickle('val_data.pickle')
train_y = pd.read_pickle('train_y.pickle')
train_y = pd.DataFrame(train_y)
val_y = pd.read_pickle('val_y.pickle')
filter_feature = pd.read_pickle('filter_feature.pickle')
filter_feature = filter_feature.tolist()
train_data = train_data.as_matrix(columns=filter_feature)
val_data = val_data.as_matrix(columns=filter_feature)

# model = models.load_model('my_model.h5')
model = Sequential() #建立模型
model.add(Dense(input_dim=61, output_dim=240)) #添加输入层、隐藏层的连接
model.add(Activation('relu')) #以Relu函数为激活函数
model.add(Dense(input_dim=240, output_dim=100)) #添加隐藏层、隐藏层的连接
model.add(Activation('relu')) #以Relu函数为激活函数
model.add(Dense(input_dim=100, output_dim=30)) #添加隐藏层、隐藏层的连接
model.add(Activation('relu')) #以Relu函数为激活函数
model.add(Dense(input_dim=30, output_dim=10)) #添加隐藏层、输出层的连接
model.add(Activation('relu')) #以sigmoid函数为激活函数
model.add(Dense(input_dim=10, output_dim=1)) #添加隐藏层、输出层的连接
model.add(Activation('relu')) #以sigmoid函数为激活函数
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(train_data, train_y, nb_epoch=100, batch_size=40) #训练模型，nb_epoch为总迭代次数，batch_size为每次训练的数量
model.save('my_model.h5')
# res = model.predict(val_data)
# print(res.shape)
# np.savetxt('test.txt', res, fmt='%0.8f')
# test_data = np.loadtxt('test.txt')
# predict_res = pd.Series(test_data)
# # print(predict_res)
# # print(val_y.shape)
# RMSE = mean_squared_error(val_y, predict_res) ** 0.5
# print(RMSE)