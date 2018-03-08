# -*- coding: utf-8 -*-

from sklearn import cluster
from sklearn import datasets

iris = datasets.load_iris()
data = iris['data']

model = cluster.KMeans( n_clusters=3 )
model.fit( data )

print(model.labels_)
