import csv
import numpy as np
import random
import sklearn.linear_model
from  sklearn.linear_model import Lasso, Ridge, ElasticNet
from pandas import DataFrame

# Load the subset file.
train_subset = []
with open('train_subset.csv', 'r') as subset_fh:

    # Parse it as a CSV file.
    subset_csv = csv.reader(subset_fh, delimiter=',', quotechar='"')

    # Skip the header row.
    next(subset_csv, None)

    # Load the data.
    for row in subset_csv:
        smiles   = row[0]#first row, all chemical symbols
        numbers=[float(i) for i in row[1].split(']')[0].split()[1:]]#converts to floats
        features=DataFrame.transpose(DataFrame(numbers))#saves features
        gap=DataFrame(float(row[2]), index=[0], columns=[0])#saves gap, what we want to predict

        train_subset.append({ 'smiles':   smiles,
                            'features': features,
                            'gap':      gap })

# Shuffle train_subset
random.shuffle(train_subset)

# Split into train and test data
train_subset_train = train_subset[:501]#splits into train and test
train_subset_test = train_subset[501:]

#Lasso regression code here

for i in range(501):
    if i==0: 
        X=train_subset_train[0]['features']#makes first row of matrix    
        y=train_subset_train[0]['gap']
    else:
        X=X.append(train_subset_train[i]['features'], ignore_index=True)#appends next set
        y=y.append(train_subset_train[i]['gap'], ignore_index=True)
Xtrain=X.as_matrix()
ytrain=y.as_matrix()

for i in range(499):#ditto for test features
    if i==0:
        X=train_subset_test[0]['features']
        y=train_subset_test[0]['gap']
    else:
        X=X.append(train_subset_test[i]['features'], ignore_index=True)
        y=y.append(train_subset_test[i]['gap'], ignore_index=True)
Xtest=X.as_matrix()
ytest=y.as_matrix()

lasso=Lasso()

predictions=lasso.fit(Xtrain, ytrain).predict(Xtest)#trains and predicts
lasso.coef_#coefficients
predictions=DataFrame(predictions)#changes forms of predictions
ytest2=DataFrame(y.values, index=predictions.index, columns=['Correct Values'])
ytest2['Predictions']=predictions#dataframe with predictions and correct value
#print ytest2

ridge=Ridge(alpha=1.0)
ridpredictions=ridge.fit(Xtrain, ytrain).predict(Xtest)
ridpredictions=DataFrame(ridpredictions)#changes forms of predictions
ytestrid=DataFrame(y.values, index=ridpredictions.index, columns=['Correct Values'])
ytestrid['Predictions']=ridpredictions#dataframe with predictions and correct value
#print ytestrid
#print ridge.coef_

en=ElasticNet(alpha=1, l1_ratio= 0)#second term is l1/l2 ratio of penalty. set to 0 means all l2, 1 means l1
enpredictions=en.fit(Xtrain,ytrain).predict(Xtest)
enpredictions=DataFrame(enpredictions)#changes forms of predictions
ytesten=DataFrame(y.values, index=enpredictions.index, columns=['Correct Values'])
ytesten['Predictions']=enpredictions#dataframe with predictions and correct value
print en.coef_
print ytesten
