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
from sklearn.metrics import f1_score

red = pd.read_csv('./winequality-red.csv',sep=';')
white = pd.read_csv('./winequality-white.csv',sep=';')
red['type'] = 1
white['type'] = 0
wines = red.append(white,ignore_index=True)
X = wines.ix[:,0:11]
y = wines['type']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

model = Sequential()

model.add(Dense(12,activation='relu',input_shape = (11,)))
model.add(Dense(1,activation='sigmoid'))

model.summary()
##printing model
SVG(model_to_dot(model).create(prog='dot', format='svg'))
plot_model(model, to_file='model.png', show_shapes=True)
##fitting
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
                   
model.fit(X_train, y_train,epochs=10, batch_size=1, verbose=1)
y_pred = model.predict(X_test)
y_pred = y_pred > 0.5
y_pred = y_pred.astype(int)
print(f1_score(y_test,y_pred))

