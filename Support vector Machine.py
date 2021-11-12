# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 22:41:00 2021

@author: acer
"""

                        
                        # SUPPORT VECTOR MACHINE -

# Support Vector Machine

import pandas as pd
import numpy as np
dataset=pd.read_csv("letterdata.csv")
dataset.shape

dataset.info()

dataset.head()

dataset.tail()

dataset.describe()

x=dataset.drop("letter", axis=1)
y=dataset["letter"]
print(y)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.20, random_state=0)

x_test.shape

from sklearn.svm import SVC

classifier=SVC(kernel='rbf', random_state=0)   #kernel thats change "sigmoid", "poly", "linear" & "rbf"=radial basis fun.

classifier.fit(x_train, y_train)

y_pred=classifier.predict(x_test)

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test, y_pred))

print(accuracy_score(y_test, y_pred)*100)






# K-NN(Nearest Neighbors)

import pandas as pd
import numpy as np

dataset=pd.read_csv("wine.csv")

dataset

dataset.shape

dataset.isnull().sum()

dataset.head()

dataset.tail()

dataset.info()

dataset.describe()

x=dataset.drop("class", axis=1)
y=dataset["class"]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.20, random_state=0)
x_train.shape

len(x_train)

len(x_test)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()

x_test_stand=sc.fit_transform(x_test)
x_train_stand=sc.fit_transform(x_train)

x_train_stand

from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier(n_neighbors=11)
classifier.fit(x_train, y_train)
y_pred=classifier.predict(x_test)

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test, y_pred))

print(accuracy_score(y_test, y_pred)*100)

# Apply Scaling

from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

x_train_stand=sc.fit_transform(x_train)

x_test_stand=sc.fit_transform(x_test)

x_train_stand

from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier(n_neighbors=11)

classifier.fit(x_train_stand, y_train)

y_pred=classifier.predict(x_test_stand)

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test, y_pred))

print(accuracy_score(y_test, y_pred)*100)





#----------------------------------------------------------------------------------------------------------



# SVM : Support Vector Machine
# Supervised ML
# Classification and Regression
# No.of features (column) > no. of record ...high dimensional spaces
# memory efficient 
# diasdavtages: Overfitting Problems.
# SVC : Support Vector Classifier

# Where we use SVM Algorithm?
# Ans: that time neaural data analysis at that time be used it.


# importing the dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv('Social_Network_Ads.csv')
print(df)

# seperate out the input & output
x=df.iloc[:, 2 :-1].values
print(x)

y=df.iloc[:, -1 :].value_counts()
print(y)



x[0:5]
y[0:5]

#standardization  (standard scaler)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
xx=sc.fit_transform(x)
print(xx)

from sklearn.model_selection  import train_test_split
x_train, x_test, y_train, y_test=train_test_split(xx,y, test_size=0.20, random_state=0)
x_train
x_test
y_train
y_test


#fitting the data
from sklearn.svm import SVC
classifier=SVC(kernel='linear')
classifier.fit(x_train, y_train)


#predict
y_pred=classifier.predict(x_test)
print(y_pred)


#model Evalution
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
print('This is Confusion matrix:',  confusion_matrix(y_test, y_pred))

print(accuracy_score(y_test, y_pred)*100)

print(classification_report(y_test, y_pred))



























































































import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data=pd.read_csv('Social_Network_Ads.csv')
print(data)

x=data.iloc[:, 2 : -1].values
print(x)

y=data.iloc[:, -1 :].values
print(y)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
xx=sc.fit_transform(x)
print(xx)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(xx,y,test_size=0.20, random_state=2)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

from sklearn.svm import SVC
classifier=SVC(kernel='linear')
model=classifier.fit(x_train, y_train)
print(model)

y_pred=classifier.predict(x_test)
print(y_pred)



from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(confusion_matrix(y_test, y_pred))

print(accuracy_score(y_test, y_pred)*100)

print(classification_report(y_test, y_pred))



import pandas as pd
import numpy as np
data=pd.read_csv('Social_Network_Ads.csv')
print(data)

x=data.iloc[:,2 : -1 ]
print(x)


y=data.iloc[:, -1 :]
print(y)


xx=data.iloc[:, 1 : 2]
print(xx)


from sklearn.preprocessing import LabelEncoder
le= LabelEncoder()
model=le.fit_transform(xx)
print(model)

ff=pd.DataFrame(model)
print(ff)
df=ff.rename(columns={0:'Gender'})
print(df)

jj=pd.concat([x, df], axis=1)
print(jj)


print(x_test.shape)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(jj,y, test_size=0.20, random_state=2)
print(x_train.shape)


print(y_train.shape)

print(y_test.shape)


from sklearn.svm import SVC
classifier=SVC(kernel='linear')
aa=classifier.fit(x_train, y_train)
print(aa)

y_pred=classifier.predict(x_test)
print(y_pred)

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test, y_pred))

print(accuracy_score(y_test, y_pred)*100)



import pandas as pd
import numpy as np
data=pd.read_csv('wine.csv')
print(data)



x=data.iloc[:, 1 : -1]
print(x)

y=data.iloc[:, 0 :1]
print(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y, test_size=0.27, random_state=0)
print(x_train.shape)


print(y_train.shape)

print(y_test.shape)

print(x_test.shape)

from sklearn.svm import SVC
classifier=SVC(kernel='rbf')
model=classifier.fit(x_train, y_train)
print(model)

y_pred=classifier.predict(x_test)
print(y_pred)

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

print(classification_report(y_test,y_pred))

print(accuracy_score(y_test, y_pred)*100)

print(confusion_matrix(y_test, y_pred))



import pandas as pd
import numpy as np
import seaborn as sn
data=pd.read_csv('Advertising.csv')
print(data)


x=data.iloc[:, 1 : -1]
print(x)

y=data.iloc[:, -1 :]
print(y)

sn.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.show()

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
modelx=sc.fit_transform(x)
print(modelx)


from sklearn.preprocessing import LabelEncoder
le= LabelEncoder()
modely=le.fit_transform(y)
print(modely)

print(len(modely))

aa=modely.reshape(-1)
print(aa)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x, aa, test_size=0.20, random_state=1000)
print(x_train.shape)

print(x_test.shape)

print(y_train.shape)

print(y_test.shape)





from sklearn.svm import SVC
classifier=SVC(kernel='rbf')
model=classifier.fit(x_train,y_train)
print(model)

y_pred=classifier.predict(x_test)
print(y_pred)

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test, y_pred))

print(accuracy_score(y_test, y_pred)*100)





import pandas as pd
import numpy as np
data=pd.read_csv('wisc_bc_data.csv')
print(data)


x=data.iloc[:, 2 : -1]
print(x)

y=data.iloc[:, -1 :]
print(y)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
jj=sc.fit_transform(y)
print(jj)

from sklearn.preprocessing import Binarizer
bi=Binarizer(threshold=3.5)
oo=bi.fit_transform(jj)
print(oo)


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
kk=data.apply(le.fit_transform)
print(kk)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(kk,oo, test_size=0.33, random_state=7)
print(x_train.shape)

print(x_test.shape)

print(y_train.shape)

print(y_test.shape)

from sklearn.svm import SVC
classifier=SVC(kernel='rbf')
model=classifier.fit(x_train, y_train)
print(model)

y_pred=model.predict(x_test)
print(y_pred)


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print( accuracy_score(y_test, y_pred)*100)

print(confusion_matrix(y_test,y_pred))

print(classification_report(y_test, y_pred))

















import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
data=pd.read_csv('petrol_consume.csv')
print(data)


x=data.iloc[:, : -1]
x

y=data.iloc[:, -1 :]
y


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.20, random_state=7)
x_train.shape

x_test.shape

y_train.shape

y_test.shape


from sklearn.svm import SVC
svm=SVC(kernel='rbf')

