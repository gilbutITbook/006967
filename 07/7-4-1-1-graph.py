# coding: utf-8
import pandas as pd

# 시코쿠 전력의 전력 소비량 데이터를 읽어 들이기
ed = [pd.read_csv(
    'shikoku_electricity_%d.csv' % year,
    skiprows=3,
    names=['DATE', 'TIME', 'consumption'],
    parse_dates={'date_hour': ['DATE', 'TIME']},
    index_col = "date_hour")
    for year in [2012, 2013, 2014, 2015, 2016]
]

elec_data = pd.concat(ed)

# -- 가시화 --
import matplotlib.pyplot as plt

# 이미지 사이즈를 설정한다
plt.figure(figsize=(10, 6))

# 시계열 그래프 생성
delta = elec_data.index - pd.to_datetime('2012/07/01 00:00:00')
elec_data['time'] = delta.days + delta.seconds / 3600.0 / 24.0

plt.scatter(elec_data['time'], elec_data['consumption'], s=0.1)
plt.xlabel('days from 2012/7/1')
plt.ylabel('electricity consumption(*10000 kWh)')

# 그래프 저장
plt.savefig('7-4-1-1-graph.png')
