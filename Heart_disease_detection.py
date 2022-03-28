import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#loading the csv datas to a panda dataframe
heart_data = pd.read_csv(r"C:\Users\atulk\Downloads\heart.csv")

#print 5 rows of heart_data
print(heart_data.tail())

#number of rows and columns
print(heart_data.shape)

#getting some info about the dataset
print(heart_data.info())

#checking for missing values
heart_data.isnull().sum()

#Statistical measures about the data
print(heart_data.describe())

#checking the distribution of traget variable
print(heart_data['target'].value_counts())

#Splitting the features and target
X = heart_data.drop(columns='target',axis=1)
Y = heart_data['target']

#Splitting into training and test data
X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size=0.2,stratify=Y,random_state=2)

# print(X.shape,X_train.shape,X_test.shape)
#Model Training(Logistic regression model)

model = LogisticRegression()

#training the model with training data
model.fit(X_train,Y_train)

#model evaluation Accuracy score
#accuracy on training data
X_train_prediction = model.predict(X_train)
training_accuracy = accuracy_score(X_train_prediction,Y_train)

#accuracy on test data
X_test_prediction = model.predict(X_test)
test_accuracy = accuracy_score(X_test_prediction,Y_test)

#Building the predictive model
# input_data = (62,0,0,138,294,1,1,106,0,1.9,1,3,2)    #target =0   does not have disease
input_data = (58,0,0,100,248,0,0,122,0,1,1,0,2)        #target = 1   has disease

#Chnge input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

#reshape the np array for 1 data point
input_data_reshape = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshape)

if(prediction[0]==0):
    print("Person does not have a heart disease")
else:
    print("Person has heart disease")

