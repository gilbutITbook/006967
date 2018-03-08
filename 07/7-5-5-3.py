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

tmp_org = tmp

for h_count in range(2,6):
    print("\n h_count:"+str(h_count))

    tmp = tmp_org[["temperature","sunhour"]]
    ld = tmp

    for i in range(1,h_count):
        ld = ld.join(tmp.shift(i),rsuffix="_"+str(i)).dropna()

    tmp = ld
    tmp["month"] = tmp.index.month
    tmp['hour'] = tmp.index.hour

    ## 데이터의 결합
    takamatsu = elec_data.join(tmp).dropna().as_matrix()

    takamatsu_elec = takamatsu[:, 0:1]
    takamatsu_wthr = takamatsu[:, 1:]

    # 학습과 성능의 평가
    import sklearn.cross_validation
    import sklearn.svm

    data_count = len(takamatsu_elec)

    # 교차검정의 준비
    kf = sklearn.cross_validation.KFold(data_count, n_folds=5)

    # 교차검정 실시(모든 패턴을 실시)
    for train, test in kf:
        x_train = takamatsu_wthr[train]
        x_test = takamatsu_wthr[test]
        y_train = takamatsu_elec[train]
        y_test = takamatsu_elec[test]

        # -- SVR --
        model = sklearn.svm.SVR()
        y_train = y_train.flatten()
        y_test = y_test.flatten()

        model.fit(x_train, y_train)
        print ("SVR: Training Score = %f, Testing(Validate) Score = %f" %
               (model.score(x_train, y_train), model.score(x_test, y_test)))
