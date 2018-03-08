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

    # scikit-learn 의 다른 데이터 세트의 형식에 합한다
    flat_data = images.reshape((-1, IMAGE_SIZE * IMAGE_SIZE * COLOR_BYTE))
    images = flat_data.view()
    return datasets.base.Bunch(data=flat_data,
                 target=labels.astype(np.int),
                 target_names=np.arange(CATEGORY_NUM),
                 images=images,
                 DESCR=None)

#####################################
from sklearn import svm, metrics

## usage:
##    python classify_handsign_1.py <n> <dir_1> <dir_2> ... <dir_m>
##      n          테스트용 데이터 디렉터리 수
##      dir_1      데이터 디렉터리 1
##      dir_m      데이터 디렉터리 m

if __name__ == '__main__':
    argvs = sys.argv

    # 평가용 디렉터리 수의 취득
    paths_for_test = argvs[2:2+int(argvs[1])]
    paths_for_train = argvs[2+int(argvs[1]):]

    print('test ', paths_for_test)
    print('train', paths_for_train)

    # 학습 데이터 읽어 들이기
    data = []
    label = []
    for i in range(len(paths_for_train)):
        path = paths_for_train[i]
        d = load_handimage(path)
        data.append(d.data)
        label.append(d.target)
    train_data = np.concatenate(data)
    train_label = np.concatenate(label)

    # 수법:선형 SVM
    classifier = svm.LinearSVC()

    # 학습
    classifier.fit(train_data, train_label)

    for path in paths_for_test:
        # 테스트 데이터 읽어 들이기
        d = load_handimage(path)

        # 테스트
        predicted = classifier.predict(d.data)

        # 결과표시
        print("### %s ###" % path)
        print("Accuracy:\n%s"
            % metrics.accuracy_score(d.target, predicted))
        print("Classification report:\n%s\n"
            % metrics.classification_report(d.target, predicted))
