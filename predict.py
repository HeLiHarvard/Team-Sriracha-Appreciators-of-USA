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
        text=row[1]
        features=DataFrame.transpose(DataFrame(text.split()[1:]))
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
        print type(X)     
        y=train_subset_train[0]['gap']
    else:
        X=X.append(train_subset_train[i]['features'], ignore_index=True)
        y=y.append(train_subset_train[i]['gap'], ignore_index=True)
print X
print y

#testing=Lasso()
#testing.fit(X, y)
#X = np.empty()