# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#import modules
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


df=pd.read_csv("loan_pred.csv") 

#finding null values
df.isnull().sum()


#visualising the catagorical datas
sns.countplot(df['Gender'])
sns.countplot(df['Married'])
sns.countplot(df['Dependents'])
sns.countplot(df['Education'])
sns.countplot(df['Self_Employed'])


#filling null values
# for int values
df['LoanAmount']=df['LoanAmount'].fillna(df['LoanAmount'].median())
#we dont use mean because of outliers


df['Gender']=df['Gender'].fillna(df['Gender'].mode()[0])
df['Dependents']=df['Dependents'].fillna(df['Dependents'].mode()[0])
df['Married']=df['Married'].fillna(df['Married'].mode()[0])
df['Credit_History']=df['Credit_History'].fillna(df['Credit_History'].mode()[0])

sns.countplot(df['Loan_Amount_Term'])
#since 360 appears a lot , fill using mode
df['Loan_Amount_Term']=df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0])

df['Self_Employed']=df['Self_Employed'].fillna(df['Self_Employed'].mode()[0])
df.isnull().sum()


#visualising the int datas
sns.distplot(df['ApplicantIncome'])
sns.boxplot(y="ApplicantIncome",data=df,hue="Education")


sns.distplot(df['CoapplicantIncome'])


sns.distplot(df['LoanAmount'])
#since it is right skewed , apply log tranformation
df['LoanAmount_log']=np.log(df['LoanAmount'])
sns.distplot(df['LoanAmount_log'])

#correlation matrix
plt.figure(figsize=(15,15))
sns.heatmap(df.corr(),annot=True,cmap="BuPu")

cols=['LoanAmount','Loan_ID']
df=df.drop(columns=cols,axis=1)

#label encoding categorical data
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
cols=["Married","Education","Self_Employed","Property_Area","Loan_Status","Dependents","Gender"]
for col in cols:
    df[col]=le.fit_transform(df[col])


X = df.drop(columns=['Loan_Status'], axis=1)
y = df['Loan_Status']

#splitting data into test and train set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test =train_test_split(X,y,test_size=0.25 , random_state=42) 

# Training the Logistic Regression model on the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))
#from logistic regression i got 81.96% acc


#Actual test set
data=pd.read_csv("test_set.csv") 
data.isnull().sum()


#filling null values
# for int values
data['LoanAmount']=data['LoanAmount'].fillna(data['LoanAmount'].median())
#we dont use mean because of outliers


data['Gender']=data['Gender'].fillna(data['Gender'].mode()[0])
data['Dependents']=data['Dependents'].fillna(data['Dependents'].mode()[0])
data['Married']=data['Married'].fillna(data['Married'].mode()[0])
data['Credit_History']=data['Credit_History'].fillna(data['Credit_History'].mode()[0])

sns.countplot(data['Loan_Amount_Term'])
#since 360 appears a lot , fill using mode
data['Loan_Amount_Term']=data['Loan_Amount_Term'].fillna(data['Loan_Amount_Term'].mode()[0])

data['Self_Employed']=data['Self_Employed'].fillna(data['Self_Employed'].mode()[0])
data.isnull().sum()

#visualising the int datas

sns.distplot(data['LoanAmount'])
#since it is right skewed , apply log tranformation
data['LoanAmount_log']=np.log(data['LoanAmount'])
sns.distplot(data['LoanAmount_log'])


#correlation matrix
plt.figure(figsize=(15,15))
sns.heatmap(data.corr(),annot=True,cmap="BuPu")

cols=['LoanAmount','Loan_ID']
data=data.drop(columns=cols,axis=1)

#label encoding categorical data
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
cols=["Married","Education","Self_Employed","Property_Area","Dependents","Gender"]
for col in cols:
    data[col]=le.fit_transform(data[col])
    

# Predicting the Test set results
y_pred = classifier.predict(data)

print(y_pred)

#creating submission file
og_test_set=pd.read_csv("test_set.csv")
sol=pd.read_csv("submit.csv")
sol["Loan_Status"]=y_pred
sol["Loan_ID"]=og_test_set["Loan_ID"]
sol["Loan_Status"].replace(0,'N',inplace=True)
sol["Loan_Status"].replace(1,'Y',inplace=True)

#converting to csv file
pd.DataFrame(sol,columns=["Loan_ID","Loan_Status"]).to_csv("final_answer.csv")
