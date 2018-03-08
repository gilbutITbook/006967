# coding: utf-8
import pandas as pd

# 시코쿠 전력의 전력 소비량 데이터를 읽어 들이기
ed = [pd.read_csv(
    'shikoku_electricity_%d.csv' % year,
    skiprows=3,
    names=['DATE', 'TIME', 'consumption'],
    parse_dates={'date_hour': ['DATE', 'TIME']},
    index_col='date_hour')
    for year in [2012, 2013, 2014, 2015, 2016]
]

elec_data = pd.concat(ed)

# 기상 데이터를 읽어 들이기
tmp = pd.read_csv(
    u'47891_city.csv',
    parse_dates={'date_hour': ["일시"]},
    index_col="date_hour",
    low_memory=False,
    na_values="×"
)

del tmp["시"]  #  [시] 열은 사용하지 않으므로 삭제

# 열 이름에 한국어가 들어가 있으면 좋지 않으므로 지금부터 사용하는 열의 이름만 영어로 변경
columns = {
    "강수량(mm)": "rain",
    "기온(℃)": "temperature",
    "일조시간(h)": "sunhour",
    "습도(％)": "humid",
}
tmp.rename(columns=columns, inplace=True)

# 기상 데이터와 전력 소비량 데이터를 일단 통합해 시간축을 맞추고 다시 분할
takamatsu = elec_data.join(tmp["temperature"]).dropna().as_matrix()

takamatsu_elec = takamatsu[:, 0:1]
takamatsu_wthr = takamatsu[:, 1:]

import sklearn.cross_validation
import sklearn.svm

data_count = len(takamatsu_elec)


# 교차검정의 준비(데이터 생성)
x_train, x_test, y_train, y_test = sklearn.cross_validation.train_test_split(
    takamatsu_wthr, takamatsu_elec, test_size=0.2)

# -- SVR --
model = sklearn.svm.SVR()
x_train = x_train
y_train = y_train.flatten()
x_test = x_test
y_test = y_test.flatten()

model.fit(x_train, y_train)

# -- 가시화 --
import matplotlib.pyplot as plt
import numpy as np

plt.show()
px = np.arange(takamatsu_wthr.min(), takamatsu_wthr.max(), 0.01)[:, np.newaxis]
py = model.predict(px)

# 이하 그래프 생성
plt.xlabel('electricity consumption(measured *10000 kW)')
plt.ylabel('electricity consumption(predicted *10000 kW)')

predicted = model.predict(takamatsu_wthr)

# 이하 컬러 환경의 설정
#plt.scatter(takamatsu_elec, predicted, s=0.5)
# 이하 모노크롬(흑백) 환경 설정
plt.scatter(takamatsu_elec, predicted, s=0.5, color="black")

plt.savefig('7-5-2-3-graph.png')
