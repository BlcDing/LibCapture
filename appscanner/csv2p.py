import pickle
import csv
import pandas

CSV_NAME = "dataset1.csv"
PICKLE_NAME = "dataset1.p"

dataset = pandas.read_csv(CSV_NAME)
dataset.drop("id", axis=1, inplace=True)

data = list()
for index, row in dataset.iterrows():
    line = row.tolist()
    label = line[-1]
    line = line[:-1]
    data.append((label, list(line)))

with open(PICKLE_NAME, "w") as f:
    pickle.dump(data, f)

# output test
# flowlist = pickle.load(open(PICKLE_NAME, 'rb'))
# print(flowlist[0])
