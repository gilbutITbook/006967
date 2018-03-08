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
    na_values=["×", "--"]
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
tmp.fillna(-1,inplace=True)

# 월, 일, 시 구하기
tmp["month"] = tmp.index.month
tmp['day'] = tmp.index.day
tmp['dayofyear'] = tmp.index.dayofyear
tmp['hour'] = tmp.index.hour

# 기상 데이터와 전력 소비량 데이터를 일단 통합해 시간축을 맞추고 다시 분할
takamatsu = elec_data.join(tmp[["temperature","sunhour","month","hour"]]).dropna().as_matrix()

takamatsu_elec = takamatsu[:, 0:1]
takamatsu_wthr = takamatsu[:, 1:]

# 학습과 성능의 평가
import sklearn.cross_validation
import sklearn.svm
model = sklearn.svm.SVR()



x_train, x_test, y_train, y_test = sklearn.cross_validation.train_test_split(
    takamatsu_wthr, takamatsu_elec, test_size=0.2)

y_train = y_train.flatten()
y_test = y_test.flatten()

model.fit(x_train, y_train)
date_name = ["기온", "일조시간","월","시간"]

output = "사용항목 = %s, 훈련스코어 = %f,검증스코어 = %f" % \
         (", ".join(date_name),
          model.score(x_train, y_train),
          model.score(x_test, y_test)
          )
#    print (output.decode('utf-8')) # Python2의 경우 이쪽 행을 사용해주세요
print (output)  # Python3용



# -- 가시화 --
import matplotlib.pyplot as plt

# 이미지 크기를 설정한다
plt.figure(figsize=(10, 6))

predicted = model.predict(x_test)

plt.xlabel('electricity consumption(measured *10000 kW)')
plt.ylabel('electricity consumption(predicted *10000 kW)')
plt.scatter(y_test, predicted, s=0.5, color="black")

plt.savefig("7-5-5-4-graph.png")
