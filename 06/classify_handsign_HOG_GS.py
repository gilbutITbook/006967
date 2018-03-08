# -*- coding: utf-8 -*-
import os
import sys
import glob
import numpy as np
from skimage import io
from sklearn import datasets
from skimage.feature import hog

CATEGORY_NUM = 6

## 라벨명(0～)을 붙인 디렉터리로 분류된 이미지 파일을 읽어 들인다
## 입력 경로는 라벨명의 상위 디렉터리
def load_handimage(path):

    # 파일 목록을 취득
    files = glob.glob(os.path.join(path, '*/*.png'))

    # 이미지와 라벨 영역을 확보
    hogs = np.ndarray((len(files), 3600), dtype = np.float)
    labels = np.ndarray(len(files), dtype=np.int)

    # 이미지와 라벨을 읽어 들인다
    for idx, file in enumerate(files):
        # 이미지 읽어 들인다
        image = io.imread(file, as_grey=True)
        h = hog(image, orientations=9, pixels_per_cell=(5, 5),
            cells_per_block=(5, 5))
        hogs[idx] = h

        # 디렉터리명으로부터 라벨을 취득
        label = os.path.split(os.path.dirname(file))[-1]
        labels[idx] = int(label)

    return datasets.base.Bunch(data=hogs,
                 target=labels.astype(np.int),
                 target_names=np.arange(CATEGORY_NUM),
                 DESCR=None)

#####################################
from sklearn import svm, metrics
from sklearn import grid_search

param_grid = {
   'C': [1, 10, 100],
   'loss': ['hinge', 'squared_hinge']
   }

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
    classifier = grid_search.GridSearchCV(svm.LinearSVC(), param_grid)

    # 학습
    classifier.fit(train_data, train_label)

    # Grid Search 결과표시
    print("Best Estimator:\n%s\n",classifier.best_estimator_)
    for params, mean_score, all_scores in classifier.grid_scores_:
        print("{:.3f} (+/- {:.3f}) for {}".format(mean_score,
            all_scores.std() / 2, params))

    for path in paths_for_test:
        # 테스트 데이터의 읽어 들이기
        d = load_handimage(path)

        # 테스트
        predicted = classifier.predict(d.data)

        # 결과표시
        print("### %s ###" % path)
        print("Accuracy:\n%s"
            % metrics.accuracy_score(d.target, predicted))
        print("Classification report:\n%s\n"
            % metrics.classification_report(d.target, predicted))
