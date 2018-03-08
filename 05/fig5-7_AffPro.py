# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from sklearn import cluster
from sklearn import datasets

# iris 데이터를 로드
iris = datasets.load_iris()
data = iris["data"]

# 학습 → 클러스터 생성
model = cluster.AffinityPropagation().fit(data)

# 학습 결과의 라벨 취득
labels = model.labels_

### 그래프 그리기

# 클러스터 수를 알기 때문에 마커는 배열로 가진다
markers = ["o", "^", "*","v", "+", "x", "d", "p", "s", "1", "2"]

# 데이터 정의
x_index = 2
y_index = 3

data_x=data[:,x_index]
data_y=data[:,y_index]

x_label = iris["feature_names"][x_index]
y_label = iris["feature_names"][y_index]

# 클러스터마다 산포도를 그린다
for idx in range(labels.max() + 1):
    plt.scatter(data_x[labels==idx], data_y[labels==idx],
                c='black' ,alpha=0.3,s=100, marker=markers[idx],
                label="cluster {0:d}".format(idx))

# 축 라벨과 타이틀 설정
plt.xlabel(x_label,fontsize='xx-large')
plt.ylabel(y_label,fontsize='xx-large')
plt.title("AffinityPropagation",fontsize='xx-large')

# 범례 표시
plt.legend( loc="upper left" )

plt.show()
