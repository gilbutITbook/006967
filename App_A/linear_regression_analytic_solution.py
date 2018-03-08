
# -*- coding: utf-8 -*-

import time
import numpy as np

# 데이터 구하기 및 결과 가시화를 행하는 메서드는
# 외부 모듈 get_data.py에 정의합니다.
from get_data import get_all, visualize_result


def main():
    # 실험용 파라미터 설정
    # dimension, nonlinear, num_of_samples를 바꿔서 결과를 비교해주세요
    # NOTE: 이것들은 실험을 위한 데이터 세트 자체를 변경하는 것이므로
    #       알고리즘 동작을 규정하는 파라미터와는 다릅니다

    # 특징 벡터의 차원
    dimension = 100
    # 비선형 플래그
    # True  -> 초평면
    # False -> 초곡면
    # 선형회귀는 초평면 모델이므로 당연히 False쪽이 좋은 측정결과를 줍니다
    nonlinear = False
    # 전체 데이터의 수
    num_of_samples = 1000
    # 노이즈의 진폭
    noise_amplitude = 0.01

    # 전체 데이터 구하기
    # NOTE: 테스트 데이터에는 표식 '_test'를 붙이고 있지만,
    #       학습용 데이터에 대해서는 계산을 실행하는 코드 안에서 식을 쉽게 찾을 수 있도록
    #       '_train'과 같은 표식은 붙이지 않습니다
    (A, Y), (A_test, Y_test) = get_all(
        dimension, nonlinear, num_of_samples, noise_amplitude
    )

    # 역행렬에 의한 측정값
    start = time.time()
    # (A^tA)^(-1) A^t Y 를 직접 계산
    D_hat_inv = (np.linalg.inv(A.T.dot(A)).dot(A.T)).dot(Y)
    print("D_hat_inv: {0:.16f}[s]".format(time.time() - start))

    # 연립방정식의 해답에 의한 측정값
    start = time.time()
    # A.tA * D = A.t Y 를  D 에 대해서 푼다
    D_hat_slv = np.linalg.solve(A.T.dot(A), A.T.dot(Y))
    print("D_hat_slv: {0:.16f}[s]".format(time.time() - start))

    # 2해의 차
    dD = np.linalg.norm(D_hat_inv - D_hat_slv)
    print("difference of two solutions: {0:.4e}".format(dD))

    # NOTE: 2개의 해에 그다지 차이가 없다는 것을 확인할 수 있으므로
    #       이하 플롯에서는 D_hat_slv만을 이용합니다
    # 테스트 데이터로의 피팅
    Y_hat = A_test.dot(D_hat_slv)
    mse = np.linalg.norm(Y_test-Y_hat) / dimension
    print("test error: {:.4e}".format(mse))

    # 실험기록용
    parameters = {
        "linearity": "nonlinear" if nonlinear else "linear",
        "dimension": dimension,
        "num_of_samples": num_of_samples,
    }
    # 결과 표시
    # NOTE: 표시용에 2차원만 건네고 있습니다
    visualize_result(
        "linear_regression_analytic_solution",
        A_test[:, :2], Y_test, Y_hat, parameters
    )

if __name__ == "__main__":
    main()
