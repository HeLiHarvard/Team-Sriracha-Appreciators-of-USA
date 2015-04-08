import numpy as np
import csv

profiles_file = 'profiles.csv'
train_file = 'train.csv'
test_file  = 'test.csv'
soln_file  = 'global_median.csv'

# Load the training data.
train_data = {}
with open(train_file, 'r') as train_fh:
    train_csv = csv.reader(train_fh, delimiter=',', quotechar='"')
    next(train_csv, None)
    for row in train_csv:
        user   = row[0]
        artist = row[1]
        plays  = int(row[2])

        if not user in train_data:
            train_data[user] = {'artists':{}}

        train_data[user]['artists'][artist] = plays

with open(profiles_file, 'r') as prof_fh:
    profiles_csv = csv.reader(prof_fh, delimiter=',', quotechar='"')
    next(profiles_csv, None)
    for row in profiles_csv:
        user  = row[0]
        if not user in train_data:
            train_data[user] = {'artists':{}}
        train_data[user]['sex'] = row[1]
        train_data[user]['age'] = row[2]
        train_data[user]['country'] = row[3]

print train_data

