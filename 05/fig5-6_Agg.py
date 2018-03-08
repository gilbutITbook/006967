# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from sklearn import cluster
from sklearn import datasets

# iris データをロード
iris = datasets.load_iris()
data = iris["data"]

# 학습 → 클러스터 생성
model = cluster.AgglomerativeClustering(n_clusters=3, linkage="ward")
model.fit(data)

# 학습 결과의 라벨 취득
labels = model.labels_

### 그래프 그리기

# 데이터 정의
x_index = 2
y_index = 3

data_x=data[:,x_index]
data_y=data[:,y_index]

x_label = iris["feature_names"][x_index]
y_label = iris["feature_names"][y_index]

# 산포도 그리기
plt.scatter(data_x[labels==0], data_y[labels==0],c='black' ,alpha=0.3,s=100, marker="o")
plt.scatter(data_x[labels==1], data_y[labels==1],c='black' ,alpha=0.3,s=100, marker="^")
plt.scatter(data_x[labels==2], data_y[labels==2],c='black' ,alpha=0.3,s=100, marker="*")

# 축 라벨과 타이틀 설정
plt.xlabel(x_label,fontsize='xx-large')
plt.ylabel(y_label,fontsize='xx-large')
plt.title("AgglomerativeClustering(ward)",fontsize='xx-large')

plt.show()
