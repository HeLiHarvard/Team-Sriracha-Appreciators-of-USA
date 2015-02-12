import csv
import numpy as np
import random
import sklearn.linear_model
from  sklearn.linear_model import Lasso, Ridge, ElasticNet
from pandas import DataFrame
from sklearn.metrics import mean_squared_error
import math

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
train_subset_train = train_subset[:5001]#splits into train and test
train_subset_test = train_subset[5001:]

#Lasso regression code here

for i in range(5001):
    if i==0: 
        X=train_subset_train[0]['features']#makes first row of matrix    
        y=train_subset_train[0]['gap']
    else:
        X=X.append(train_subset_train[i]['features'], ignore_index=True)#appends next set
        y=y.append(train_subset_train[i]['gap'], ignore_index=True)
Xtrain=X.as_matrix()
ytrain=y.as_matrix()

for i in range(4999):#ditto for test features
    if i==0:
        X=train_subset_test[0]['features']
        y=train_subset_test[0]['gap']
    else:
        X=X.append(train_subset_test[i]['features'], ignore_index=True)
        y=y.append(train_subset_test[i]['gap'], ignore_index=True)
Xtest=X.as_matrix()
ytest=y.as_matrix()

alpha=10
lasso=Lasso(alpha=alpha)
print 'alpha: ' + str(alpha)

predictions=lasso.fit(Xtrain, ytrain).predict(Xtest)#trains and predicts
lasso.coef_#coefficients
predictions=DataFrame(predictions)#changes forms of predictions
ytest2=DataFrame(y.values, index=predictions.index, columns=['Correct Values'])
ytest2['Predictions']=predictions#dataframe with predictions and correct value
#print ytest2
lasso.coef_
print 'Lasso: ' + str(100*math.sqrt(mean_squared_error(ytest2['Correct Values'], ytest2['Predictions']))/ytest2['Correct Values'].mean(axis=1))

ridge=Ridge(alpha=alpha)
ridpredictions=ridge.fit(Xtrain, ytrain).predict(Xtest)
ridpredictions=DataFrame(ridpredictions)#changes forms of predictions
ytestrid=DataFrame(y.values, index=ridpredictions.index, columns=['Correct Values'])
ytestrid['Predictions']=ridpredictions#dataframe with predictions and correct value
#print ytestrid
ridge.coef_
print 'Ridge: '+str(100*math.sqrt(mean_squared_error(ytestrid['Correct Values'], ytestrid['Predictions']))/ytestrid['Correct Values'].mean(axis=1))

l1ratio=.1
en=ElasticNet(alpha=alpha, l1_ratio= l1ratio)#second term is l1/l2 ratio of penalty. set to 0 means all l2, 1 means l1
enpredictions=en.fit(Xtrain,ytrain).predict(Xtest)
enpredictions=DataFrame(enpredictions)#changes forms of predictions
ytesten=DataFrame(y.values, index=enpredictions.index, columns=['Correct Values'])
ytesten['Predictions']=enpredictions#dataframe with predictions and correct value
en.coef_
ytesten
print 'Elastic Net ' + '(L1 ratio = ' + str(l1ratio) + ') :'+ str(100*math.sqrt(mean_squared_error(ytesten['Correct Values'], ytesten['Predictions']))/ytest2['Correct Values'].mean(axis=1))

stupidprediction=DataFrame(np.mean(ytrain),index=enpredictions.index, columns=['Averages'])
print 'Stupid prediction: ' + str(100*math.sqrt(mean_squared_error(ytesten['Correct Values'], stupidprediction))/ytest2['Correct Values'].mean(axis=1))

allpredictions=DataFrame(0,index=enpredictions.index, columns=['Lasso', 'Ridge', 'Elastic Net', 'Average'])
allpredictions['Lasso'] = ytest2['Predictions']
allpredictions['Ridge']=ytestrid['Predictions']
allpredictions['Elastic Net']=ytesten['Predictions']
allpredictions['Average']=allpredictions[['Lasso', 'Ridge', 'Elastic Net']].mean(axis=1)
print 'Average Prediction: ' + str(100*math.sqrt(mean_squared_error(ytesten['Correct Values'], allpredictions['Average']))/ytest2['Correct Values'].mean(axis=1))

#make predictions stuff 

pred_filename  = 'predictionsExample.csv'

with open(pred_filename, 'w') as pred_fh:

    # Produce a CSV file.
    pred_csv = csv.writer(pred_fh, delimiter=',', quotechar='"',lineterminator='\n')

    # Write the header row.
    pred_csv.writerow(['Id', 'Prediction'])

    for i in range(4999):
        pred_csv.writerow([i, allpredictions['Average'][i]])