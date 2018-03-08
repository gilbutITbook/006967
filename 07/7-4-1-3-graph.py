# coding: utf-8
import pandas as pd

# 기상 데이터를 읽어 들이기
tmp = pd.read_csv(
    u'47891_city.csv',
    parse_dates={'date_hour': ["일시"]},
    index_col = "date_hour",
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

# -- 가시화 --
import matplotlib.pyplot as plt

# 이미지 사이즈를 설정한다
plt.figure(figsize=(10, 6))

# 히스토그램 생성
delta = tmp.index - pd.to_datetime('2012/07/01 00:00:00')
tmp['time'] = delta.days + delta.seconds / 3600.0 / 24.0

plt.scatter(tmp['time'], tmp['temperature'], s=0.1)
plt.xlabel('days from 2012/7/1')
plt.ylabel('Temperature(C degree)')

# 그래프 저장
plt.savefig('7-4-1-3-graph.png')
