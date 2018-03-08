# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from sklearn import cluster
from sklearn import datasets

# iris 데이터를 로드
iris = datasets.load_iris()
data = iris["data"]

# 학습 → 클러스터 생성
model = cluster.KMeans(n_clusters=3)
model.fit(data)

# 학습 결과의 라벨 취득
labels = model.labels_

### 그래프 그리기
x_index = 2
y_index = 3

data_x=data[:,x_index]
data_y=data[:,y_index]

x_max = 7.5
x_min = 0
y_max = 3
y_min = 0
x_label = iris["feature_names"][x_index]
y_label = iris["feature_names"][y_index]


plt.scatter(data_x[labels==0], data_y[labels==0],c='black' ,alpha=0.3,s=100, marker="o",label="cluster 0")
plt.scatter(data_x[labels==1], data_y[labels==1],c='black' ,alpha=0.3,s=100, marker="o",label="cluster 1")
plt.scatter(data_x[labels==2], data_y[labels==2],c='black' ,alpha=0.3,s=100, marker="o",label="cluster 2")
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xlabel(x_label,fontsize='large')
plt.ylabel(y_label,fontsize='large')
plt.show()
