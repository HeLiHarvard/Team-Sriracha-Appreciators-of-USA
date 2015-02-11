import csv
import numpy as np
import random
import sklearn.linear_model
from  sklearn.linear_model import Lasso
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
        smiles   = row[0]
        blah=row[1]
        #testing2=[float(i) for i in row[1].split()[1:]]
        features=DataFrame.transpose(DataFrame(blah.split()[1:]))
        gap=DataFrame(float(row[2]), index=[0], columns=[0])

        train_subset.append({ 'smiles':   smiles,
                            'features': features,
                            'gap':      gap })

# Shuffle train_subset
random.shuffle(train_subset)

# Split into train and test data
train_subset_train = train_subset[:501]
train_subset_test = train_subset[501:]

#Lasso regression code here

for i in range(501):
    if i==0:
        X=train_subset_train[0]['features']
        y=train_subset_train[0]['gap']
    else:
        X=X.append(train_subset_train[i]['features'], ignore_index=True)
        y=y.append(train_subset_train[i]['gap'], ignore_index=True)
Xtest=X.as_matrix()
ytest=y.as_matrix()
print type(Xtest), type(ytest)
#testing=Lasso()
#testing.fit(Xtest, ytest)
print row[1].split()[1:]
for i in blah.split()[1:]:
    print i
    print float(i[:-1])
