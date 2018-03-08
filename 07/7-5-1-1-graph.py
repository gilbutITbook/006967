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
    na_values="×"
)

del tmp["시"]  # [시] 열은 사용하지 않으므로 삭제

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
takamatsu_whhr = takamatsu[:, 1:]

# -- 가시화 --
import matplotlib.pyplot as plt

# 이하 그래프 생성
plt.xlabel('Temperature(C degree)')
plt.ylabel('electricity consumption(*10000 kW)')

# 이하 모노크롬(흑백) 환경의 설정
plt.scatter(takamatsu_whhr, takamatsu_elec, s=0.5,
            color="gray", label='electricity consumption(measured)')

plt.savefig('7-5-1-1-graph.png')
