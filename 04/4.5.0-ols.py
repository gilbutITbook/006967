# -*- coding: utf-8 -*-

import math

import numpy as np
import matplotlib.pyplot as plt


### 분산이 있는 사인파 데이터를 작성

x = np.random.rand(1000, 1)  # 0~1까지 난수를 1000개 만든다
x = x * 20 - 10              # 값의 범위를 -10~10으로 변경

y = np.array([math.sin(v) for v in x])  # 사인파 커브
y += np.random.randn(1000)  # 표준 정규 분포(평균0, 표준 편차1) 난수를 더한다


### 학습: 최소제곱법

from sklearn import linear_model


model = linear_model.LinearRegression()
model.fit(x, y)


### 계수, 절편, 결정 계수를 표시

print('계수', model.coef_)
print('절편', model.intercept_)

r2 = model.score(x, y)
print('결정계수', r2)


### 그래프 표시

plt.scatter(x, y, marker='+')
plt.scatter(x, model.predict(x), marker='o')
plt.show()
