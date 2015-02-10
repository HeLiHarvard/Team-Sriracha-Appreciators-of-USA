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
        #features = np.array(row[1])
        gap      = float(row[2])
        
        train_subset.append({ 'smiles':   smiles,
                            'features': features,
                            'gap':      gap })

# Shuffle train_subset
random.shuffle(train_subset)

# Split into train and test data
train_subset_train = train_subset[:501]
train_subset_test = train_subset[501:]

#Lasso regression code here

for i in range(3):
    if i==0:
        X=train_subset_train[0]['features']
        print type(X)     
        #y=DataFrame(train_subset_train[0]['gap'])
        #print type(y)
    else:
        X=X.append(X, train_subset_train[i]['features'])
        #y=np.append(y, train_subset_train[i]['gap'])
        print 'okay'
print X.shape
#print y.shape
#first=DataFrame.transpose(DataFrame(testing1.split()[1:]))
#print first

#testing=Lasso()
#testing.fit(X, y)
#X = np.empty()