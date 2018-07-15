from sklearn.datasets import load_digits
from sklearn.svm import SVC
import _pickle as pickle
from scipy import misc

data = load_digits()

filename = 'finalized_model.sav'

trained_model = pickle.load(open(filename,'rb'))


def isSparse(img):
	count=0
	for pixel in img:
		if pixel < 8:
			count+=1
	if count<len(img)/2:
		return False
	return True


img = misc.imread('0_1.jpg',flatten= True)
img = misc.imresize(img, (8,8))
img = img.astype(data.data.dtype)
img = misc.bytescale(img, high=16, low=0)


X_test = []

for eachRow in img:
	for eachPixel in eachRow:
		X_test.append(eachPixel)

if not (isSparse(X_test)):
	# print('isSparse')
	X_test = [16.0-x for x in X_test]

# print(X_test)
print(trained_model.predict([X_test]))
