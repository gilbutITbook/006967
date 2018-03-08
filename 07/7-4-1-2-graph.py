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

# -- 가시화 --
import matplotlib.pyplot as plt

# 이미지 사이즈를 설정한다
plt.figure(figsize=(10, 6))

# 히스토그램 생성
plt.hist(elec_data['consumption'], bins=50, color="gray")
plt.xlabel('electricity consumption(*10000 kW)')
plt.ylabel(u'count')

# 그래프 저장
plt.savefig('7-4-1-2-graph.png')
