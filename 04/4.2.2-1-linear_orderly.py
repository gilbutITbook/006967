# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np


### y = 3x -2 데이터 생성

x = np.random.rand(100, 1)  # 0~1까지 난수를 100개 만든다
x = x * 4 - 2               # 값의 범위를 -2~2로 변경

y = 3 * x - 2  # y = 3x - 2


### 학습

from sklearn import linear_model


model = linear_model.LinearRegression()
model.fit(x, y)


### 계수, 절편을 표시

print('계수', model.coef_)
print('절편', model.intercept_)


### 그래프 표시

plt.scatter(x, y, marker='+')
plt.show()
