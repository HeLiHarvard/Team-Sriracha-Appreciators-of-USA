import csv
import numpy as np
import random

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
        features = np.array(row[1])
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
