import csv
import numpy as np
import random
import gzip
import sklearn.linear_model
from  sklearn.linear_model import Lasso, Ridge, ElasticNet
from pandas import DataFrame
from sklearn.metrics import mean_squared_error
import math

train_filename = 'train.csv.gz'
test_filename  = 'test.csv.gz'
pred_filename  = 'example_mean.csv'
# Load the data file.
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
# Load the test file.
test_data = []
with gzip.open(test_filename, 'r') as test_fh:

    # Parse it as a CSV file.
    test_csv = csv.reader(test_fh, delimiter=',', quotechar='"')
    
    
    # Skip the header row.
    next(test_csv, None)

    # Load the data.
    for row in test_csv:
        id       = row[0]
        smiles   = row[1]
        numbers2 = np.array([float(x) for x in row[2:258]])
        features=DataFrame.transpose(DataFrame(numbers2))#saves features
        
        test_data.append({ 'id':       id,
                           'smiles':   smiles,
                           'features': features })

for i in range(10000):#ditto for test features
    if i==0:
        X=train_subset[0]['features']
        y=train_subset[0]['gap']
    else:
        X=X.append(train_subset[i]['features'], ignore_index=True)
        y=y.append(train_subset[i]['gap'], ignore_index=True)
Xtrain=X.as_matrix()
ytrain=y.as_matrix()


for i in range(824230):#ditto for test features
    if i==0:
        X=test_data[0]['features']
    else:
        X=X.append(test_data[i]['features'], ignore_index=True)
Xtest=X.as_matrix()

alpha=.01

ridge=Ridge(alpha=alpha)
ridpredictions=ridge.fit(Xtrain, ytrain).predict(Xtest)
ridpredictions=DataFrame(ridpredictions)#changes forms of predictions
#ytestrid=DataFrame(y.values, index=ridpredictions.index, columns=['Correct Values'])
#ytestrid['Predictions']=ridpredictions#dataframe with predictions and correct value
#print ytestrid
ridge.coef_

pred_filename  = 'predictionsExample.csv'

with open(pred_filename, 'w') as pred_fh:

    # Produce a CSV file.
    pred_csv = csv.writer(pred_fh, delimiter=',', quotechar='"',lineterminator='\n')

    # Write the header row.
    pred_csv.writerow(['Id', 'Prediction'])

    for i in range(824230):
        pred_csv.writerow([i, ridpredictions[i]])

