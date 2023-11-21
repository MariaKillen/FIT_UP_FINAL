''''#importing libraries
import pandas as pd
import numpy as np
import pickle

df=pd.read_csv("bodyPerformance.csv")

#Feature Engineering
#df = df.drop('gender',axis=1)
df.replace("A", 1 , inplace = True)
df.replace("B", 2 , inplace = True)
df.replace("C", 3 , inplace = True)
df.replace("D", 4 , inplace = True)
#print(df)
from sklearn import preprocessing
#labelencodingg
LE=preprocessing.LabelEncoder()
#fitting it to our dataset
df.gender=LE.fit_transform(df.gender)


#Loading the data
#method 1 - using iloc X->Independent variable Y->dependent var
X = df.iloc[:,:-1].values #2D array
Y = df.iloc[:,-1].values #2D array

#print(len(X[0]))
#print(Y)
#spliting the data  into training and testing set
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)

#Standard Scaling - Normalizing the data - X_train
# Importing StandardScaler from scikit-learn
from sklearn.preprocessing import StandardScaler
sst = StandardScaler()
X_train=sst.fit_transform(X_train) #nomalizing

#Training the model

from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier()
model.fit(X_train,Y_train)
#print(len(X_train[0]))
#Testing
#Predicting
y_pred = model.predict(sst.transform(X_test))
#print(y_pred)
#print("its of row one")
#y_pred=model.predict(sst.transform([X[3]]))
print(y_pred)
'''
import pandas as pd
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
# Load the data
df = pd.read_csv("Book1110.csv")
#df.head()
#print(df)
# Feature Engineering (you can add more feature engineering here)
df.rename({'class': 'Classi'}, axis=1, inplace=True)
df.replace({"A": 1, "B": 2, "C": 3, "D": 4}, inplace=True)
# Label Encoding
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
df['gender'] = LE.fit_transform(df['gender'])
#Loading the data
#method 1 - using iloc X->Independent variable Y->dependent var
X = df.iloc[:,:-1].values #2D array
Y = df.iloc[:,-1].values #2D array
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)
# Standard Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=0)
model.fit(X_train, Y_train)
# Evaluate the model
Y_train_pred = model.predict(X_train)
Y_test_pred = model.predict(X_test)
pickle.dump(model,open('model.pkl','wb')) #we are Serializing our model by creating model.pkl and writing into it by 'wb'
model=pickle.load(open('model.pkl','rb')) #Deserializing - reading the file - "rb"
print("Sucess loaded")


#Execute this file only once and create the pkl file.