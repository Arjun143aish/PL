#Importing necessary libraries
import os
import pandas as pd
import numpy as np

#Selecting folder
os.chdir("C:\\Users\\user\\Documents\\Python\\Interview Preparation\\case study")

#reading xlsx and sheet name
FullRaw = pd.read_excel("datasets_48024_87370_Bank_Personal_Loan_Modelling.xlsx",sheet_name ='Data')

#Checking for null values
FullRaw.isnull().sum()

#Encoding using description

FullRaw['Family'].unique()

def mapping(word):
    word_dict = {1:'one',2:'two',3:'three',4:'four'}
    return word_dict[word]

FullRaw['Family'] = FullRaw['Family'].apply(lambda x:mapping(x))


FullRaw['Education'].unique()
Condition = [FullRaw['Education'] ==1,FullRaw['Education'] ==2,
             FullRaw['Education'] ==3]
Choice = ['Undergrad','Graduate','Professional']
FullRaw['Education'] = np.select(Condition,Choice)

FullRaw['Securities Account'] = np.where(FullRaw['Securities Account'] == 1,'Yes','No')
FullRaw['CD Account'] = np.where(FullRaw['CD Account'] == 1,'Yes','No')
FullRaw['Online'] = np.where(FullRaw['Online'] == 1,'Yes','No')
FullRaw['CreditCard'] = np.where(FullRaw['CreditCard'] == 1,'Yes','No')

FullRaw.drop(['ID'],axis =1, inplace =True)

import seaborn as sns
Corrdf = FullRaw.corr()
sns.heatmap(Corrdf,xticklabels= Corrdf.columns,
            yticklabels= Corrdf.columns,cmap = 'coolwarm_r')

Categorical_vars = (FullRaw.dtypes == 'object')
dummydf = pd.get_dummies(FullRaw.loc[:,Categorical_vars],drop_first =True)

FullRaw2 = pd.concat([FullRaw.loc[:,~Categorical_vars],dummydf],axis =1)

from sklearn.model_selection import train_test_split

Train,Test = train_test_split(FullRaw2,test_size = 0.3, random_state =123)

Train_X = Train.drop(['Personal Loan'],axis =1)
Train_Y = Train['Personal Loan']
Test_X= Test.drop(['Personal Loan'],axis =1)
Test_Y = Test['Personal Loan']


from sklearn.ensemble import RandomForestClassifier

M1 = RandomForestClassifier(random_state=123).fit(Train_X,Train_Y)

Test_Pred = M1.predict(Test_X)

from sklearn.metrics import confusion_matrix

Con_Mat = confusion_matrix(Test_Pred,Test_Y)

sum(np.diag(Con_Mat))/Test_Y.shape[0]*100

from sklearn.model_selection import GridSearchCV

n_trees = [100,150,175]
n_split = [50,75,100]
n_depth = [3,4,5]

my_param_grid = {'n_estimators': n_trees,'min_samples_split': n_split,'max_depth': n_depth}

Grid = GridSearchCV(RandomForestClassifier(random_state=123,criterion = 'entropy'),
                    param_grid = my_param_grid,cv =5,scoring='accuracy').fit(Train_X,Train_Y)

Grid.best_score_


import pickle

pickle.dump(M1,open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))




