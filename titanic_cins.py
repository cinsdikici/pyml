#----- Cinsdikici Titanic Kaggle Problem ------
# CopyRight: Muhammed Cinsdikici
# Version  : 2017.10.25-18:45
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
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score


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
data_test  = pd.read_csv("../../Kaggle/Titanic/Data/test.csv")
data_tumset = data_train.append(data_test,ignore_index=True)
# ignore_index makes the index value continou.. In this data
# there is 890 training datarecord. If we use ignore_index=True
# than index continous 891-->1309. If we omit it, appended index
# starts with 0 again and end with 417.
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

"""
Difference Between Data_Train & Data_Test
-----------------------------------------
Some of the parameter values does not included in Data_Train
It should be problem for recalling the system that we are
going to build to test.

In Data_Train these values (Ticket Category) are not available
------------------
1. Ticket_A
2. Ticket_AQ3
3. Ticket_AQ4
4. Ticket_LP
5. Ticket_SCA3
6. Ticket_STONOQ

In Data_Test these values (Ticket category and a Cabin Category) are not available
------------------
0. Cabin_T
1. Ticket_AS
2. Ticket_CASATON
3. Ticket_Fa
4. Ticket_LINE
5. Ticket_PPP
6. Ticket_SCOW
7. Ticket_SOP
8. Ticket_SP
9. Ticket_SWPP

In Data_Tumset, All Categories are constructed. 66 Categories are
existing in the final form of Data_Tumset.
"""

# print the data_train header 5 lines from matrix
# print(data_train.head())

# data_train statistics for countable parameters
# mean,std,min,Q1,Q2,Q3 of each parameter
print(data_train.describe())

# Plot target visualization
#print (data_train["Sex"])
sn.barplot(x="Sex", y="Survived", data=data_train)


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

data_test.Age=data_test.Age.fillna(data_test.Age.mean())
data_test["Cabin"]=data_test.Cabin.fillna("U")


# Dikkat TumSet'in NaN elemanlari TumSet'in mean'i ile degistirildi
data_tumset.Age=data_tumset.Age.fillna(data_tumset.Age.mean())
data_tumset["Cabin"]=data_tumset.Cabin.fillna("U")

# To see the parameters list..
list([data_train.columns.values,"   ", data_test.columns.values])
# Pausing = input("1. Check the Columns of after Cabin NaN Check Train and Test Data than Press <ENTER> to continue")


#------Butunlesik Kategorileri Ayri Ayri ele almak-------
# pd.get_dummies metodu sayesinde "unique" tüm deerler bir 
# kategori olacak _ekilde ç1kart1lmaktad1r. Elde edilen Dataframe
# yeni bir dataframe'dir.

#embark--> liman isimlerini ayri ayri set
# parametre haline getirmek.
embarked = pd.get_dummies(data_train.Embarked , prefix='Embarked' )
print(embarked)
data_train=pd.concat([data_train,embarked],axis=1)
data_train=data_train.drop("Embarked",axis=1)

embarked = pd.get_dummies(data_test.Embarked , prefix='Embarked' )
data_test=pd.concat([data_test,embarked],axis=1)
data_test=data_test.drop("Embarked",axis=1)

embarked = pd.get_dummies(data_tumset.Embarked , prefix='Embarked' )
data_tumset=pd.concat([data_tumset,embarked],axis=1)
data_tumset=data_tumset.drop("Embarked",axis=1)

# To see the parameters list..
#list([data_train.columns.values,"   ", data_test.columns.values])
#Pausing = input("2. Check the Columns of after Embark Train and Test Data than Press <ENTER> to continue")


#pclass --> ticketlar ayri ayri
# parametre haline getiriliyor.
pclass = pd.get_dummies(data_train.Pclass , prefix='Pclass' )
print(pclass)
data_train=pd.concat([data_train,pclass],axis=1)
data_train=data_train.drop("Pclass",axis=1)

pclass = pd.get_dummies(data_test.Pclass , prefix='Pclass' )
data_test=pd.concat([data_test,pclass],axis=1)
data_test=data_test.drop("Pclass",axis=1)

pclass = pd.get_dummies(data_tumset.Pclass , prefix='Pclass' )
data_tumset=pd.concat([data_tumset,pclass],axis=1)
data_tumset=data_tumset.drop("Pclass",axis=1)

# To see the parameters list..
#list([data_train.columns.values,"   ", data_test.columns.values])
#Pausing = input("3. Check the Columns of after PClass Train and Test Data than Press <ENTER> to continue")


#Cabin --> Oda 0simlerinin ilk harflerine gore kategorize edilip
# parametre haline getiriliyor.
kabin = pd.DataFrame()
kabin.Cabin= data_train['Cabin'].map( lambda c: c[0] )
kabin = pd.get_dummies(kabin.Cabin, prefix='Cabin')
print(kabin)
data_train=pd.concat([data_train,kabin],axis=1)
data_train=data_train.drop("Cabin",axis=1)


kabin = pd.DataFrame()
kabin.Cabin= data_test['Cabin'].map( lambda c: c[0] )
kabin = pd.get_dummies(kabin.Cabin, prefix='Cabin')
data_test=pd.concat([data_test,kabin],axis=1)
data_test=data_test.drop("Cabin",axis=1)

kabin = pd.DataFrame()
kabin.Cabin= data_tumset['Cabin'].map( lambda c: c[0] )
kabin = pd.get_dummies(kabin.Cabin, prefix='Cabin')
data_tumset=pd.concat([data_tumset,kabin],axis=1)
data_tumset=data_tumset.drop("Cabin",axis=1)


# To see the parameters list..
#list([data_train.columns.values,"   ", data_test.columns.values])
#Pausing = input("4. Check the Columns of after Cabin Removal Check Train and Test Data than Press <ENTER> to continue")



#------ Conversion of Category names to Binary data -----
#Convert Sex category values male,female into binary 1,0
# a column of Dataframe is "Series"
# numpy's where function is used to find values "male" in Series.
data_train.Sex  = pd.Series(np.where(data_train.Sex=="male",1,0),name="Sex")
data_test.Sex   = pd.Series(np.where(data_test.Sex=="male",1,0),name="Sex")
data_tumset.Sex = pd.Series(np.where(data_tumset.Sex=="male",1,0),name="Sex")


#------ To see each element in Trainin data in classical way ------
# to see each Name in the training data
for i in range(860,len(data_train.Name)):
    print(i,".ci isim",data_train.Name[i])


#---- Passengers Prefixes of Names are categorized and added as new category-----
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
title = pd.DataFrame()
title["Title"]= data_train.Name.map(lambda name: name.split(',')[1].split('.')[0].strip())
title.Title = title.Title.map(Title_Dictionary)
title = pd.get_dummies(title.Title)
data_train = pd.concat([data_train,title],axis=1)

title = pd.DataFrame()
title["Title"]= data_test.Name.map(lambda name: name.split(',')[1].split('.')[0].strip())
title.Title = title.Title.map(Title_Dictionary)
title = pd.get_dummies(title.Title)
data_test = pd.concat([data_test,title],axis=1)


title = pd.DataFrame()
title["Title"]= data_tumset.Name.map(lambda name: name.split(',')[1].split('.')[0].strip())
title.Title = title.Title.map(Title_Dictionary)
title = pd.get_dummies(title.Title)
data_tumset = pd.concat([data_tumset,title],axis=1)

# To see the parameters list..
#list([data_train.columns.values,"   ", data_test.columns.values])
#Pausing = input("5. Check the Columns of after Title Mr/Mss Check Train and Test Data than Press <ENTER> to continue")



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
data_train=pd.concat([data_train,bilet],axis=1)
data_train=data_train.drop("Ticket",axis=1)


bilet = pd.DataFrame()
bilet["Ticket"] = data_test.Ticket.map(clean_ticket)
bilet = pd.get_dummies(bilet.Ticket,prefix="Ticket")
bilet.shape
print(bilet.head())
data_test=pd.concat([data_test,bilet],axis=1) 
data_test=data_test.drop("Ticket",axis=1)

bilet = pd.DataFrame()
bilet["Ticket"] = data_tumset.Ticket.map(clean_ticket)
bilet = pd.get_dummies(bilet.Ticket,prefix="Ticket")
bilet.shape
print(bilet.head())
data_tumset=pd.concat([data_tumset,bilet],axis=1) 
data_tumset=data_tumset.drop("Ticket",axis=1)

# To see the parameters list..
#list([data_train.columns.values,"   ", data_test.columns.values])
#Pausing = input("6. Check the Columns of after Ticket Check Train and Test Data than Press <ENTER> to continue")


#--- Obtain Train and Test again from tumset
data_train_tumset = data_tumset[:891]
data_test_tumset = data_tumset[891:]


#--- Data_Full = Data_Train + Data_Test
data_full = pd.DataFrame()
data_full = data_train.append(data_test,ignore_index = True)

#------Correlation Matrix -----
# Look which parameters are correlated with it
# We see Survival is correlated with 
# Fare(0.257) and Parch(0.081)

#------ To calculate & Visualise correlation of each parameters in Training data
# Look which parameters are correlated mostly with Survived
# First take correlation matrix of data_Train
corr= data_tumset.corr()
#print (corr) 

# to see which parameters -or indexes- 
# corrolated with Survived where corr > 0
corr_result=pd.Series(np.where(corr.Survived > 0,corr.Survived,0),corr.index)
print (corr_result, max(corr_result))
# Than plot the most corrolated parameters with Survived.
sn.barplot(y=corr_result.values,x=corr_result.index)


#corr= data_full.corr()
#print (corr)




#--- Machine Learnin Algorithms
# Model selection
model1 = RandomForestClassifier(n_estimators=100)
model2 = SVC()
model3 = KNeighborsClassifier(n_neighbors=3)
model4 = LogisticRegression()

# Data Training of the Model
# model1.fit(data_train_Input,data_train_Target)


