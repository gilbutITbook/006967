import matplotlib.pyplot as plt
from sklearn import cluster
from sklearn import datasets


# iris 데이터를 로드
iris = datasets.load_iris()
data = iris['data']

# 학습 → 클러스터 생성
model = cluster.KMeans(n_clusters=3)
model.fit(data)

# 学習結果のラベル取得
labels = model.labels_


### グラフの描画
MARKERS = ["o", "^" , "*" , "v", "+", "x", "d", "p", "s", "1", "2"]

# 指定されたインデックスの feature 値で散布図を作成する関数
def scatter_by_features(feat_idx1, feat_idx2):
    for lbl in range(labels.max() + 1):
        clustered = data[labels == lbl]
        plt.scatter(clustered[:, feat_idx1], clustered[:, feat_idx2],
                    c='black' ,alpha=0.3,s=100,
                    marker=MARKERS[lbl], label='label {}'.format(lbl))

    plt.xlabel(iris["feature_names"][feat_idx1],fontsize='xx-large')
    plt.ylabel(iris["feature_names"][feat_idx2],fontsize='xx-large')


plt.figure(figsize=(16, 16))

# feature "sepal length" と "sepal width"
plt.subplot(3, 2, 1)
scatter_by_features(0, 1)

# feature "sepal length" と "petal length"
plt.subplot(3, 2, 2)
scatter_by_features(0, 2)

# feature "sepal length" と "petal width"
plt.subplot(3, 2, 3)
scatter_by_features(0, 3)

# feature "sepal width" と "petal length"
plt.subplot(3, 2, 4)
scatter_by_features(1, 2)

# feature "sepal width" と "petal width"
plt.subplot(3, 2, 5)
scatter_by_features(1, 3)

# feature "petal length" と "petal width"
plt.subplot(3, 2, 6)
scatter_by_features(2, 3)

plt.tight_layout()
plt.show()
