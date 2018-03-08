# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

x = np.random.rand(100, 1)      # 0~1까지 난수를 100개 만든다
x = x * 2 - 1                   # 값의 범위를 -2~2로 변경

y = 4 * x * 3 - 3 * x * 2 + 2 * x - 1

y += np.random.randn( 100, 1 )  # 표준 정규 분포(평균0, 표준 편차1)
                               #난수를 더한다
# 학습 데이터 30개
x_train = x[:30]
y_train = y[:30]

# 테스트 데이터 70개
x_test = x[30:]
y_test = y[30:]

plt.subplot( 2, 2, 1 )
plt.scatter( x, y, marker='+' )
plt.title( 'all' )
plt.subplot( 2, 2, 2 )
plt.scatter( x_train, y_train, marker='+' )
plt.title( 'train' )

plt.subplot( 2, 2, 3 )
plt.scatter( x_test, y_test, marker='+' )
plt.title( 'test' )

plt.tight_layout()
plt.show()
