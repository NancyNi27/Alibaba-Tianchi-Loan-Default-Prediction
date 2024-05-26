# -*- coding: utf-8 -*-
"""
Student ID:476469252
Student Name: Wenyu Ni

"""
# d
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import warnings 
import os
import plotly.figure_factory as ff
import plotly.offline as pyo

warnings.filterwarnings('ignore')

##step 1: Explore the Data
## check current path
print("filepath:",os.getcwd())

#read the file
data_train = pd.read_csv('./train.csv')
data_testA = pd.read_csv('./testA.csv')

data_train_sample = pd.read_csv('./train.csv', nrows = 5)

'''
## set the chunksize to control the datasize of iteration
chunker = pd.read_csv('./train.csv', chunksize = 5)

for i in chunker:
    print(type(i))
    print(len(i))
'''

## data size and index
data_trainShape = data_train.shape
print(data_trainShape)

data_testAShape = data_testA.shape
print(data_testAShape)

data_train_sampleShape = data_train_sample.shape
print(data_train_sampleShape)

data_trainColumns = data_train.columns
print(data_trainColumns)

print(data_train.info())

description = data_train.describe()
print(description)
description_html1 = description.to_html()
with open('description.html','w') as f:
    f.write(description_html1)
    
## check the value and outliers
print(f'There are {data_train.isnull().any().sum()} columns in train dataset with missing values.')

have_null_fea_dict = ( data_train.isnull().sum() /len(data_train)).to_dict()
fea_null_moreThanHalf = {}
for key,value in have_null_fea_dict.items():
    if value > 0.5:
        fea_null_moreThanHalf[key] = value
        
fea_null_moreThanHalf

## NaN visulization
missing = data_train.isnull().sum()/len(data_train)
missing = missing[missing > 0]
missing.sort_values(inplace =True)
missing.plot.bar()


## check the columns have only value

one_value_fea = []
for col in data_train.columns:
    if data_train[col].nunique() <= 1:
        one_value_fea.append(col)

one_value_fea_test = []
for col in data_testA.columns:
    if data_testA[col].nunique() <= 1:
        one_value_fea_test.append(col)

one_value_fea
one_value_fea_test
    
print(f'There are {len(one_value_fea)} columns in train dataset with one unique value ')
print(f'There are {len(one_value_fea_test)} columns in testA dataset with one unique value ')

## check the numberical and object types of features
numeric_features = data_train.select_dtypes(include = ['int64','float64'])
print(f'the numerical types are:{numeric_features.columns}')

object_features = data_train.select_dtypes(include = ['object'])
print(f'the object types are:{object_features.columns}')

print(f'the grade data are:\n{data_train.grade}')

##divide the numercial data into continuous and discrete
def get_numerical_categories_fea(dataset, feas):
    numerical_continuous_fea = []
    numerical_discrete_fea = []
    for fea in feas:
        temp = dataset[fea].nunique()
        # if the nuique <= 10, recongise it as discrete
        if temp <= 10:
            numerical_discrete_fea.append(fea)
        else:
            numerical_continuous_fea.append(fea)
    return numerical_continuous_fea,numerical_discrete_fea

numerical_continuous_fea,numerical_discrete_fea = get_numerical_categories_fea(data_train, numeric_features)

print("the numerical_continuous are:")
numerical_continuous_fea

print("the numerical _discrete are:")
numerical_discrete_fea

## discrete value counts
data_train['term'].value_counts()
data_train['homeOwnership'].value_counts()
data_train['verificationStatus'].value_counts()
data_train['isDefault'].value_counts()
data_train['initialListStatus'].value_counts()
data_train['applicationType'].value_counts()
data_train['policyCode'].value_counts()
data_train['n11'].value_counts()
data_train['n12'].value_counts()

f = pd.melt(data_train, value_vars = numerical_continuous_fea)
g = sns.FacetGrid(f, col = "variable", col_wrap=2, sharex = False, sharey = False)
g = g.map(sns.displot, "value")

## ploting Transaction Amount Values Distribution
# setting up the figrue
plt.figure(figsize=(16,12))
plt.suptitle('Transaction Values Distribution',fontsize =22)

# First subplot: Raw loan amount distributio
plt.subplot(221)
sub_plot_1 = sns.histplot(data_train['loanAmnt'],kde =True)
sub_plot_1.set_title("Loan Amount Distribution", fontsize = 18)
sub_plot_1.set_xlabel("")
sub_plot_1.set_ylabel("Probability", fontsize = 15)

# Second subplot:Log-transformed loan amount distribution
plt.subplot(222)
# Transformation with np.log1p which is log(1+x) to handle zero values
sub_plot_2 = sns.histplot(np.log1p(data_train['loanAmnt']),kde = True)
sub_plot_2.set_title("Loan Amouny(Log) Distribution",fontsize = 18)
sub_plot_2.set_xlabel("")
sub_plot_2.set_ylabel("Probability", fontsize = 15)

plt.tight_layout()
plt.show()

## check the object values
object_features
data_train['grade'].value_counts()
data_train['subGrade'].value_counts()
data_train['employmentLength'].value_counts()
data_train['issueDate'].value_counts()
data_train['earliesCreditLine'].value_counts()
data_train['isDefault'].value_counts()

## Variable Distribution Visualization
# single one
plt.figure(figsize =(9,9))
sns.barplot(x = data_train["employmentLength"].value_counts(dropna = False)[:20].index, 
            y = data_train["employmentLength"].value_counts(dropna = False)[:20].values)

plt.show()

## check the different y value's distribution on X
train_loan_va = data_train.loc[data_train['isDefault'] == 1]
train_loan_nova = data_train.loc[data_train['isDefault'] == 0]

#setting subgraph layout
fig, axs = plt.subplots(2,2, figsize = (15,8))
ax1, ax2, ax3, ax4 =axs[0,0], axs[0,1],axs[1,0],axs[1,1]

# bar chart
train_loan_va.groupby('grade')['grade'].count().plot(kind = 'barh', ax = ax1,title = 'Count of grade fraud')
train_loan_nova.groupby('grade')['grade'].count().plot(kind = 'barh', ax = ax2, title ='Count of grade non-fraud')
train_loan_va.groupby('employmentLength')['employmentLength'].count().plot(kind = 'barh', ax = ax3, title = 'Count of employmentLength Fraud')
train_loan_nova.groupby('employmentLength')['employmentLength'].count().plot(kind = 'barh', ax = ax4, title = 'Count of employmentLength non-fraud')

## the Distribution of continuous variables on different y values
flg, axs = plt.subplots(1,2,figsize = (15,6))
ax1, ax2 = axs[0], axs[1]

data_train.loc[data_train['isDefault'] == 1]['loanAmnt'].apply(np.log)\
    .plot(kind = 'hist',bins = 100,title = 'Log Loan Amt - va', color = 'r', xlim = (-3,10), ax = ax1)

data_train.loc[data_train['isDefault'] == 0]['loanAmnt'].apply(np.log)\
    .plot(kind = 'hist',bins = 100,title = 'Log Loan Amt - nova', color = 'b', xlim = (-3,10), ax = ax2)
    
## implement visulization with matplot and sns
total = len(data_train)
total_amt = data_train.groupby(['isDefault'])['loanAmnt'].sum().sum()
plt.figure(figsize = (12,5))
## 1 represents row, 2 represents column, so thats mean there are 2 pictures in the total, 1 represents drawing the 1st picture at this time
plt.subplot(121)

plot_tr = sns.countplot(x = 'isDefault', data = data_train) ## count the amount of each features of data_train
plot_tr.set_title("Fraud Loan Distribution \n 1: good user | 1: bad user", fontsize = 14)
plot_tr.set_xlabel("Is fraud by count", fontsize = 16)
plot_tr.set_ylabel('Count', fontsize = 16)

for p in plot_tr.patches:
    height = p.get_height()
    plot_tr.text(p.get_x()+p.get_width()/2.,height + 3,'{:1.2f}%'.format(height/total*100), ha ="center",fontsize =15)
percent_amt = (data_train.groupby(['isDefault'])['loanAmnt'].sum())
percent_amt = percent_amt.reset_index()
plt.subplot(122)
plot_tr_2 = sns.barplot(x = 'isDefault', y = 'loanAmnt', dodge = True, data = percent_amt)
plot_tr_2.set_title("Total Ampunt in loanAmt \n 0: good user | 1: bad user", fontsize = 14)
plot_tr_2.set_xlabel("Is fraud by percent", fontsize = 16)
plot_tr_2.set_ylabel("Total Loan Amount Scalar", fontsize = 16)
for p in plot_tr_2.patches:
    height = p.get_height()
    plot_tr_2.text(p.get_x()+p.get_width()/2.,height+3,'{:1.2f}%'.format(height/total_amt * 100), ha = "center", fontsize =15)

#convert to time format issueDateDT feature represents the number of days from the data to date to the earliest date in the data set(2007-06-01)
data_train['issueDate'] = pd.to_datetime(data_train['issueDate'],format = '%Y-%m-%d')
startdate = datetime.datetime.strptime('2007-06-01', '%Y-%m-%d')
data_train['issueDateDT'] = data_train['issueDate'].apply(lambda x: x-startdate).dt.days
#convert to time format
data_testA['issueDate'] = pd.to_datetime(data_train['issueDate'],format = '%Y-%m-%d')
startdate =datetime.datetime.strptime('2007-06-01', '%Y-%m-%d')
data_testA['issueDateDT'] = data_testA['issueDate'].apply(lambda x: x-startdate).dt.days

#train and test issueDateDT dates ovrelap so it is unwise to use timebased splitting for validation
plt.hist(data_train['issueDateDT'],label = 'train');
plt.hist(data_testA['issueDateDT'], label = 'test');
plt.legend();
plt.title('Distribution of issueDateDT dates');


pivot = pd.pivot_table(data_train, index = ['grade'], columns =['issueDateDT'], values = ['loanAmnt'], aggfunc = np.sum)
pivot

import pandas_profiling
pfr = pandas_profiling.ProfileReport(data_train)
pfr.to_file("./example.html")
