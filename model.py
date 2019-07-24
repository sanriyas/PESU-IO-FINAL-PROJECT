'''
Output:

Accuracy : 0.8375027673234448
Confusion Matrix: 
 [[16688   899]
 [ 2771  2227]]

'''

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

#reading and cleaning the dataset
dataset = pd.read_csv('dataset.csv',usecols=[2,3,4,7,8,9,10,11,12,13,14,15,16,19,20,21,23],na_values='nan')
dataset=dataset.dropna()
dataset['WindGustDir']=LabelEncoder().fit_transform(dataset['WindGustDir'])
dataset['WindDir9am']=LabelEncoder().fit_transform(dataset['WindDir9am'])
dataset['WindDir3pm']=LabelEncoder().fit_transform(dataset['WindDir3pm'])
dataset['RainToday'].replace({'No': 0, 'Yes': 1},inplace = True)
dataset['RainTomorrow'].replace({'No': 0, 'Yes': 1},inplace = True)

#splitting into data and output
x=dataset[['MinTemp','MaxTemp','Rainfall','WindGustDir','WindDir9am','WindDir3pm','WindSpeed9am','WindSpeed3pm','Humidity9am','Humidity3pm','Pressure9am','Pressure3pm','Temp9am','Temp3pm','RainToday']]
y=dataset[['RainTomorrow']]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=2)

model= LogisticRegression()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
acc= accuracy_score(y_test,y_pred)
print('Accuracy :',acc)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print("Confusion Matrix: \n",cm)

'''
#Another model (Accuracy: 83.45)
from keras.models import Sequential
from keras.layers import Dense
model=Sequential()
model.add(Dense(15, input_dim=15, activation='relu'))
model.add(Dense(8, activation='relu')) 
model.add(Dense(1, activation='sigmoid')) 
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])
model.fit(x_train, y_train, epochs = 5,batch_size = 100 , validation_data=(x_test, y_test))
acc=model.evaluate(x_test, y_test)
print("\nAccuracy: ",acc)

'''
