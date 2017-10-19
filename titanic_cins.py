#----- Cinsdikici Titanic Kaggle Problem ------
# CopyRight: Muhammed Cinsdikici
# Version  : 2017.10.19-16:30
# Brief Exp: Takes Trainin Data and revision the data matrix
#            a. Replaces "NaN" elements
#            b. Expand Categories with subcategories
#            c. Make meaningles text values replacing with binary values
# Reference:
# (1) https://www.kaggle.com/helgejo/an-interactive-data-science-tutorial
# (2) https://www.kaggle.com/c/titanic#tutorials
# (3) https://www.kaggle.com/c/titanic/data

import pandas as pd
import numpy as np
import seaborn as sn
import os


# Modelling Algorithms
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier

# Modelling Helpers
from sklearn.preprocessing import Imputer , Normalizer , scale
from sklearn.cross_validation import train_test_split , StratifiedKFold
from sklearn.feature_selection import RFECV

# Visualisation
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns


cwd = os.getcwd() #Current Working Directory

#----- Classic Text File Read at below
#datafd = open("../Kaggle/Titanic/Data/train.csv")
#data_train = datafd.read()
#datafd.close()

#Read CSV text file into data_train as matrix
data_train = pd.read_csv("../../Kaggle/Titanic/Data/train.csv")
data_test = pd.read_csv("../../Kaggle/Titanic/Data/test.csv")
# ignore_index makes the index value continou.. In this data
# there is 890 training datarecord. If we use ignore_index=True
# than index continous 891-->1309. If we omit it, appended index
# starts with 0 again and end with 417.
data_full = data_train.append(data_test,ignore_index = True)
"""
Variable Description
------------------------------------
Survived: Survived (1) or died (0)
Pclass: Passenger's class
Name: Passenger's name
Sex: Passenger's sex
Age: Passenger's age
SibSp: Number of siblings/spouses aboard
Parch: Number of parents/children aboard
Ticket: Ticket number
Fare: Fare
Cabin: Cabin
Embarked: Port of embarkation
"""

# print the data_train header 5 lines from matrix
# print(data_train.head())

# data_train statistics for countable parameters
# mean,std,min,Q1,Q2,Q3 of each parameter
print(data_train.describe())

# Plot target visualization
#print (data_train["Sex"])
sn.barplot(x="Sex", y="Survived", data=data_train)

#------Correlation Matrix -----
# Look which parameters are correlated with it
# We see Survival is correlated with 
# Fare(0.257) and Parch(0.081)
corr= data_train.corr()
print (corr)

#------Barplot both variables -----
# Barplot the variable with target
# Both method plots same

# sn.barplot(x="Embarked", y="Survived", data=data_train)
# sn.barplot(x=data_train.Embarked, y=data_train.Survived)

#------Degeri NaN olan Kategorilerin Degerlerini Duzenlemek--------
# data_train icinde degeri "nan" olan
# degiskenleri mean() degerlerle degistirmek
#yeni_data_train = pd.DataFrame()
#yeni_data_train["Age"]=data_train.Age.fillna(data_train.Age.mean())
#print(yeni_data_train)

#both line below make the same operation.
#data_train["Age"]=data_train.Age.fillna(data_train.Age.mean())
data_train.Age=data_train.Age.fillna(data_train.Age.mean())

data_train["Cabin"]=data_train.Cabin.fillna("U")


#------Butunlesik Kategorileri Ayri Ayri ele almak-------
# pd.get_dummies metodu sayesinde "unique" tüm deerler bir 
# kategori olacak _ekilde ç1kart1lmaktad1r. Elde edilen Dataframe
# yeni bir dataframe'dir.

#embark--> liman isimlerini ayri ayri 
# parametre haline getirmek.
embarked = pd.get_dummies(data_train.Embarked , prefix='Embarked' )
print(embarked)
data_train=pd.concat([data_train,embarked],axis=1)
data_train=data_train.drop("Embarked",axis=1)

#pclass --> ticketlar ayri ayri
# parametre haline getiriliyor.
pclass = pd.get_dummies(data_train.Pclass , prefix='Pclass' )
print(pclass)
data_train=pd.concat([data_train,pclass],axis=1)
data_train=data_train.drop("Pclass",axis=1)

#Cabin --> Oda 0simlerinin ilk harflerine gore kategorize edilip
# parametre haline getiriliyor.
kabin = pd.DataFrame()
kabin.Cabin= data_train['Cabin'].map( lambda c: c[0] )
kabin = pd.get_dummies(kabin.Cabin, prefix='Cabin')
print(kabin)
data_train=pd.concat([data_train,kabin],axis=1)
data_train=data_train.drop("Cabin",axis=1)

#------ Conversion of Category names to Binary data -----
#Convert Sex category values male,female into binary 1,0
# a column of Dataframe is "Series"
# numpy's where function is used to find values "male" in Series.
data_train.Sex  = pd.Series(np.where(data_train.Sex=="male",1,0),name="Sex")


#------ To see each element in Trainin data in classical way ------
# to see each Name in the training data
for i in range(0,len(data_train.Name)):
    print(i,".ci isim",data_train.Name[i])

#------ To calculate & Visualise correlation of each parameters in Training data
# Look which parameters are correlated mostly with Survived
# First take correlation matrix of data_Train
corr= data_train.corr()
#print (corr) 

# to see which parameters -or indexes- 
# corrolated with Survived where corr > 0
corr_result=pd.Series(np.where(corr.Survived > 0,corr.Survived,0),corr.index)
print (corr_result, max(corr_result))
# Than plot the most corrolated parameters with Survived.
sn.barplot(y=corr_result.values,x=corr_result.index)

#---- Passengers Prefixes of Names are categorized and added as new category-----
title = pd.DataFrame()
title["Title"]= data_train.Name.map(lambda name: name.split(',')[1].split('.')[0].strip())
Title_Dictionary = {
                    "Capt":       "Officer",
                    "Col":        "Officer",
                    "Major":      "Officer",
                    "Jonkheer":   "Royalty",
                    "Don":        "Royalty",
                    "Sir" :       "Royalty",
                    "Dr":         "Officer",
                    "Rev":        "Officer",
                    "the Countess":"Royalty",
                    "Dona":       "Royalty",
                    "Mme":        "Mrs",
                    "Mlle":       "Miss",
                    "Ms":         "Mrs",
                    "Mr" :        "Mr",
                    "Mrs" :       "Mrs",
                    "Miss" :      "Miss",
                    "Master" :    "Master",
                    "Lady" :      "Royalty"
                    }
title.Title = title.Title.map(Title_Dictionary)
title = pd.get_dummies(title.Title)

#-----Bilet iceriginde biletin ait oldugu seriyi kategorilestirmek
def clean_ticket(bilet):
    bilet = bilet.replace('.','')
    bilet = bilet.replace('/','')
    bilet = bilet.split()
    bilet = map (lambda t: t.strip(), bilet)
    bilet = list(filter (lambda t: not t.isdigit(),bilet))
    if len(bilet)>0:
        return bilet[0]
    else:
        return "XXX"

bilet = pd.DataFrame()
bilet["Ticket"] = data_train.Ticket.map(clean_ticket)
bilet = pd.get_dummies(bilet.Ticket,prefix="Ticket")
bilet.shape
print(bilet.head())



