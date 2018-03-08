『Python을 이용한 기계학습 입문 7장 센서 데이터를 이용한 회귀문제』 샘플코드에 대해서.
2016-11-28 ISP

[사용법]
・데이터 파일（CSV파일）은 스크립트와 같은 폴더에 넣어주세요.
・코드 실행은 다음과 같이 합니다.

  #명령어 라인의 경우
  python 스크립트명

  # IPython을 포함하는 python 콘솔의 경우는 다음과 같이 실행합니다.
  run 스크립트명

・7-5-2이후의 스크립트 실행에는 시간이 걸립니다.


[수정장소]
・P183
  2번째 줄  dt.를 삭제
  틀림：tmp['day'] = tmp.index.dt.day
  ↓
  수정：tmp['day'] = tmp.index.day

・P180
  21번째 줄 fillna에서 inplace=True지정을 추가

  틀림：tmp["sunhour"].fillna(-1)
  ↓
  수정：tmp["sunhour"].fillna(-1,inplace=True)

・P184
  마지막으로, 그림7.8을 출력하는 코드는 7-5-3-2-graph.py이 아닌 7-5-5-4-graph.py입니다. 또한, 플롯 대상에 학습 데이터를 포함하고 있어 그림7-8도 약간 다릅니다.



이상
