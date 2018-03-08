# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np


### 분산 y = 3x_1 - 2x_2 + 1 데이터 생성

x1 = np.random.rand(100, 1)  # 0~1까지 난수를 100개 만든다
x1 = x1 * 4 - 2              # 값의 범위를 -2~2로 변경

x2 = np.random.rand(100, 1)  # x2에 대해서도 같게
x2 = x2 * 4 - 2

y = 3 * x1 - 2 * x2 + 1

y += np.random.randn(100, 1)  # 표준 정규 분포(평균 0, 표준 편차 1)의 난수를 추가함


### 학습

from sklearn import linear_model


x1_x2 = np.c_[x1, x2]  # [[x1_1, x2_1], [x1_2, x2_2], ..., [x1_100, x2_100]]
                       # 형태로 변환

model = linear_model.LinearRegression()
model.fit(x1_x2, y)


### 계수, 절편, 결정 계수를 표시

print('계수', model.coef_)
print('절편', model.intercept_)

print('결정계수', model.score(x1_x2, y))


### 그래프 표시

y_ = model.predict(x1_x2)  # 구한 회귀식으로 예측

plt.subplot(1, 2, 1)
plt.scatter(x1, y, marker='+')
plt.scatter(x1, y_, marker='o')
plt.xlabel('x1')
plt.ylabel('y')

plt.subplot(1, 2, 2)
plt.scatter(x2, y, marker='+')
plt.scatter(x2, y_, marker='o')
plt.xlabel('x2')
plt.ylabel('y')

plt.tight_layout()
plt.show()
