from sklearn.datasets import load_digits
from sklearn.svm import SVC
import _pickle as pickle
from scipy import misc

data = load_digits()

filename = 'finalized_model.sav'

X = data.data 
y = data.target

trained_model = pickle.load(open(filename,'rb'))

img = misc.imread('2_1.jpg', flatten = True)
img = misc.imresize(img, (8,8))
img = img.astype(X.dtype)
img = misc.bytescale(img, high=16, low=0)
# img = 16.00-img

X_test = []

for eachRow in img:
	for eachPixel in eachRow:
		X_test.append(eachPixel)

# print(X[0])
# print(y[0])

print(trained_model.predict([X_test]))
