import cv2
import numpy as np
from numpy.linalg import norm
import os

SZ = 20          #训练图片长宽
MODEL_SVM = 1
MODEL_CNN = 2

#来自opencv的sample，用于svm训练
def deskew(img):
	m = cv2.moments(img)
	if abs(m['mu02']) < 1e-2:
		return img.copy()
	skew = m['mu11']/m['mu02']
	M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
	img = cv2.warpAffine(img, M, (SZ, SZ), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
	return img

#来自opencv的sample，用于svm训练
def preprocess_hog(digits):
	samples = []
	for img in digits:
		gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
		gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
		mag, ang = cv2.cartToPolar(gx, gy)
		bin_n = 16
		bin = np.int32(bin_n*ang/(2*np.pi))
		bin_cells = bin[:10,:10], bin[10:,:10], bin[:10,10:], bin[10:,10:]
		mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
		hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
		hist = np.hstack(hists)
		
		# transform to Hellinger kernel
		eps = 1e-7
		hist /= hist.sum() + eps
		hist = np.sqrt(hist)
		hist /= norm(hist) + eps
		
		samples.append(hist)
	return np.float32(samples)

class StatModel(object):
	def load(self, fn):
		self.model = self.model.load(fn)  
	def save(self, fn):
		self.model.save(fn)

class SVM(StatModel):
	def __init__(self, C = 1, gamma = 0.5):
		self.model = cv2.ml.SVM_create()
		self.model.setGamma(gamma)
		self.model.setC(C)
		self.model.setKernel(cv2.ml.SVM_RBF)
		self.model.setType(cv2.ml.SVM_C_SVC)
#训练svm
	def train(self, samples, responses):
		self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)
#字符识别
	def predict(self, samples):
		r = self.model.predict(samples)
		return r[1].ravel()

class Trainer:
	def __del__(self):
		self.save_traindata()
	def train_svm(self):
		self.model = SVM(C=1, gamma=0.5)
		if os.path.exists("svm.dat"):
			print("loading svm.dat...")
			self.model.load("svm.dat")
			print(self.model)
		else:
			print("no svm.dat, waiting for training...")
			chars_train = []
			chars_label = []

			for root, dirs, files in os.walk("train\\chars"):
				if len(os.path.basename(root)) > 1:
					continue
				root_int = ord(os.path.basename(root))
				for filename in files:
					if not filename.startswith("gt_"):
						filepath = os.path.join(root,filename)
						digit_img = cv2.imread(filepath)
						digit_img = cv2.cvtColor(digit_img, cv2.COLOR_BGR2GRAY)
						chars_train.append(digit_img)
						chars_label.append(root_int)
			
			chars_train = list(map(deskew, chars_train))
			chars_train = preprocess_hog(chars_train)
			chars_label = np.array(chars_label)
			self.model.train(chars_train, chars_label)
		return self.model

	def test(self):
		chars_test = []
		chars_label = []

		for root, dirs, files in os.walk("train\\chars"):
			if len(os.path.basename(root)) > 1:
				continue
			root_int = ord(os.path.basename(root))
			for filename in files:
				if filename.startswith("gt_"):
					filepath = os.path.join(root,filename)
					digit_img = cv2.imread(filepath)
					digit_img = cv2.cvtColor(digit_img, cv2.COLOR_BGR2GRAY)
					chars_test.append(digit_img)
					chars_label.append(root_int)
		
		chars_test = list(map(deskew, chars_test))
		chars_test = preprocess_hog(chars_test)
		chars_label = np.array(chars_label)

		# print(chars_test)
		# print(chars_test.type())

		result = self.model.predict(chars_test)


		mask = (result==chars_label)
		correct = np.count_nonzero(mask)
		ratio = correct*100.0/result.size
		print(ratio)
		return ratio

	def save_traindata(self):
		if not os.path.exists("svm.dat"):
			self.model.save("svm.dat")

class CNNTrainer:
	def __init__(self):
		self.kNearest = cv2.ml.KNearest_create()
		self.kNearest.setDefaultK(1)
	def __del__(self):
		self.save_traindata()
	def loadKNNDataAndTrainKNN(self):
		npaFlattenedImages = []
		npaClassifications = []

		try:
			npaClassifications = np.loadtxt("classifications.txt", np.float32)
		except:
			print("error, unable to open classifications.txt, exiting program\n")
			os.system("pause")
			return False

		try:
			npaFlattenedImages = np.loadtxt("flattened_images.txt", np.float32)
		except:
			print("error, unable to open flattened_images.txt, exiting program\n")
			os.system("pause")
			return False
		npaClassifications = npaClassifications.reshape((npaClassifications.size, 1))
		self.kNearest.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)
		return True

	def save_traindata(self):
		if not os.path.exists("cnn.xml"):
			self.kNearest.save("cnn.xml")

if __name__ == '__main__':
	testFlag = False
	trainType = MODEL_CNN

	if trainType == MODEL_SVM:
		t = Trainer()
		t.train_svm()
		if testFlag:
			t.test()
	elif trainType == MODEL_CNN:
		t = CNNTrainer()
		t.loadKNNDataAndTrainKNN()
