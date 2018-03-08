『Python에 의한 머신 러닝 입문 6장 이미지를 사용한 손 모양 분류에 대해
2016-11-28 ISP

[사용방법]
・data.zip은 스크립트 폴더에 같이 넣어주세요.

・스크립트는 명령 프롬프트에서 실행하세요.
 （P140의 HOG 가시화만 명령 프롬프트 사용）
  인수 지정이 필요하니 주의하세요.

・다음은 이 책에서 실행할 스크립트입니다.
P122
  run trial_handsign_SVM.py ./data/my_learn8/ ./data/my_test2/
  run trial_handsign_SVM.py ./data/my_learn10/ ./data/other_test2/

P126
  run classify_handsign_1.py 1 ./data/m01 ./data/m02 ./data/m03 ./data/m04
  run classify_handsign_1.py 1 ./data/other_test2 ./data/m02 ./data/m03 ./data/m04

P128
  run classify_handsign_1.py 4 ./data/m01 ./data/m05 ./data/m06 ./data/m07 ./data/m02 ./data/m03 ./data/m04 ./data/m08 ./data/m09 ./data/m10 ./data/m11 ./data/m12 ./data/m13 ./data/m14 ./data/m15 ./data/m16

P130
  run classify_handsign_2.py 4 ./data/m01 ./data/m05 ./data/m06 ./data/m07 ./data/m02 ./data/m03 ./data/m04 ./data/m08 ./data/m09 ./data/m10 ./data/m11 ./data/m12 ./data/m13 ./data/m14 ./data/m15 ./data/m16

P136
  run classify_handsign_2.py 4 ./data/m01 ./data/m05 ./data/m06 ./data/m07 ./data/m02 ./data/m03 ./data/m04c ./data/m08 ./data/m09 ./data/m10 ./data/m11 ./data/m12 ./data/m13 ./data/m14 ./data/m15 ./data/m16

P140
  # 이것은 명령 프롬프트에서 실행한 것입니다.
  python viewHOG40.py ./data/m01/2/01_2_001.png

  # IPython을 포함해 python 명령 프롬프트인 경우는 다음과 같이 실행합니다.
  run viewHOG40.py ./data/m01/2/01_2_001.png

P142
  run classify_handsign_HOG_2.py 4 ./data/m01 ./data/m05 ./data/m06 ./data/m07 ./data/m02 ./data/m03 ./data/m04c ./data/m08 ./data/m09 ./data/m10 ./data/m11 ./data/m12 ./data/m13 ./data/m14 ./data/m15 ./data/m16

P148
  run classify_handsign_HOG_2.py 4 ./data/m01 ./data/m05 ./data/m06 ./data/m07 ./data/m02 ./data/m03 ./data/m04c ./data/m08 ./data/m09 ./data/m10 ./data/m11 ./data/m12 ./data/m13 ./data/m14 ./data/m15 ./data/m16

[수정장소]
・P124 리스트 6-2
  60행, 61행째 print 문에 괄호 추가

・P129 리스트 6-3
  60행, 61행째 print 문에 괄호 추가


이상