# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from sklearn import datasets

# digits 데이터를 로드
digits = datasets.load_digits()

# 이미지를 2행 5열로 표시
for label, img in zip(digits.target[:10], digits.images[:10]):
	plt.subplot(2, 5, label+1)
	plt.axis('off')
	plt.imshow(img, cmap=plt.cm.gray_r, interpolation='nearest')
	plt.title('Digit:{0}'. format(label))

plt.show()
