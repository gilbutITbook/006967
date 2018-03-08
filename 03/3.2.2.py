# -*- coding: utf-8 -*-

import numpy as np
from sklearn import datasets

# 손으로 쓴 숫자 데이터 읽기
digits=datasets.load_digits()

# 3과 8의 데이터 위치를 구하기
flag_3_8=(digits.target==3)+(digits.target==8)

# 3과 8의 데이터를 구하기
images = digits.images[flag_3_8]
labels = digits.target[flag_3_8]

# 3과 8의 이미지 데이터를 1차원화
images = images.reshape(images.shape[0],-1)


########################################

from sklearn import tree

# 3과 8의 이미지 데이터를 1차원화
images = images.reshape(images.shape[0],-1)

# 분류기 생성
n_samples = len(flag_3_8[flag_3_8])
train_size = int(n_samples*3/5)
classifier = tree.DecisionTreeClassifier()
classifier.fit(images[:train_size], labels[:train_size])



########################################

from sklearn import metrics

expected=labels[train_size:]
predicted=classifier.predict(images[train_size:])

print('Accuracy:\n', metrics.accuracy_score(expected, predicted))
