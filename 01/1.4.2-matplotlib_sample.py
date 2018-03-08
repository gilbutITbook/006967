# 각 방정식을 설정하기 위한 NumPy를 임포트
import numpy as np
# matplotlib의 pyplot를 plt로 임포트
import matplotlib.pyplot as plt

# x축의 영역과 정밀도를 설정하고 x 값을 준비
x = np.arange( -3, 3, 0.1 )
# 각 방정식의 y 값을 준비
y_sin = np.sin( x )
x_rand = np.random.rand(100) * 6 - 3
y_rand = np.random.rand(100) * 6 - 3

# figure 객체를 생성
plt.figure()

# 1개의 그래프로 표시하는 설정
plt.subplot( 1, 1, 1 )

# 각 방정식의 선형과 마커, 라벨을 설정하고 플롯
## 선형도
plt.plot( x, y_sin, marker='o', markersize=5, label='line' )

## 산포도
plt.scatter( x_rand, y_rand, label='scatter' )

# 범례 표시를 설정
plt.legend()
# 그리드 라인을 표시
plt.grid( True )

# 그래프 표시
plt.show()
