import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import plot_model
from keras.utils.vis_utils import model_to_dot
from IPython.display import SVG
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score,r2_score
r = np.vectorize(round)
np.set_printoptions(precision=0)

red = pd.read_csv('./winequality-red.csv',sep=';')
white = pd.read_csv('./winequality-white.csv',sep=';')
red['type'] = 1
white['type'] = 0
wines = red.append(white,ignore_index=True)
y = wines['quality']
X = wines.drop('quality',axis=1)
X = X.drop('type',axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
X_train = np.matrix(X_train)
X_test = np.matrix(X_test)

model = Sequential()
print(X.columns)
model.add(Dense(200,activation='relu',input_shape = (11,)))
model.add(Dense(100,activation='relu'))
model.add(Dense(1))

model.summary()
##printing model
SVG(model_to_dot(model).create(prog='dot', format='svg'))
plot_model(model, to_file='model.png', show_shapes=True)
#fitting
model.compile(loss='mse',optimizer='rmsprop')
                   
model.fit(X_train, y_train,epochs=55, batch_size=1, verbose=1)
y_pred = r(model.predict(X_test))

#x = np.append(np.matrix(y_test).transpose(),np.matrix(y_pred),axis=1)
x = np.matrix(y_test).transpose() == np.matrix(y_pred)
y_pred2 = r(model.predict(X_train))
q = np.matrix(y_train).transpose() == np.matrix(y_pred2)
print(np.sum(q),q.shape)
print(np.sum(x),x.shape)
x=x.astype(int)
np.savetxt('test.out', x, delimiter=',',fmt ='%.0f')

