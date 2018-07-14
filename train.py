from sklearn.datasets import load_digits
from sklearn.svm import SVC
# from sklearn.model_selection import train_test_split
import numpy as np 
import _pickle as pickle

data = load_digits()

X = data.data 
y = data.target

trainer = SVC(gamma = 0.001)
trainer.fit(X, y)

filename = 'finalized_model.sav'
pickle.dump(trainer, open(filename, 'wb'))

print(X[0])