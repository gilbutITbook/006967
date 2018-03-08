# -*- coding: utf-8 -*-
import os
import sys
import glob
import numpy as np
from skimage import io
from sklearn import datasets

IMAGE_SIZE = 40
COLOR_BYTE = 3
CATEGORY_NUM = 6

## 라벨명(0～)을 붙인 디렉터리로 분류된 이미지 파일을 읽어 들인다
## 입력 경로는 라벨명의 상위 디렉터리
def load_handimage(path):

    # 파일 목록을 취득
    files = glob.glob(os.path.join(path, '*/*.png'))

    # 이미지와 라벨 영역을 확보
    images = np.ndarray((len(files), IMAGE_SIZE, IMAGE_SIZE,
                            COLOR_BYTE), dtype = np.uint8)
    labels = np.ndarray(len(files), dtype=np.int)

    # 이미지와 라벨을 읽어 들인다
    for idx, file in enumerate(files):
       # 이미지 읽어 들인다
       image = io.imread(file)
       images[idx] = image

       # 디렉터리명으로부터 라벨을 취득
       label = os.path.split(os.path.dirname(file))[-1]
       labels[idx] = int(label)

    # scikit-learn의 다른 데이터 세트의 형식에 합한다
    flat_data = images.reshape((-1, IMAGE_SIZE * IMAGE_SIZE * COLOR_BYTE))
    images = flat_data.view()
    return datasets.base.Bunch(data=flat_data,
                 target=labels.astype(np.int),
                 target_names=np.arange(CATEGORY_NUM),
                 images=images,
                 DESCR=None)

#####################################
from sklearn import svm, metrics

## 학습 데이터의 디렉터리, 테스트 데이터의 디렉터리를 지정한다
if __name__ == '__main__':
    argvs  = sys.argv
    train_path = argvs[1]
    test_path = argvs[2]

    # 학습 데이터의 읽어 들이기
    train = load_handimage(train_path)

    # 수법:선형 SVM
    classifier = svm.LinearSVC()

    # 학습
    classifier.fit(train.data, train.target)

    # 테스트 데이터의 읽어 들이기
    test = load_handimage(test_path)

    # 테스트
    predicted = classifier.predict(test.data)

    # 결과표시
    print("Accuracy:\n%s" % metrics.accuracy_score(test.target, predicted))
