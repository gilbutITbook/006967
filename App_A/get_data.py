
# -*- coding: utf-8 -*-


import os
import json
from datetime import datetime

import numpy as np
from matplotlib import pyplot as plt


def visualize_result(
    experiment_name,
    X_test, Y_test, Y_hat, parameters,
    losses=None, save_dir="results"
):
    """
    결과 과시화
    """
    # 저장장소 디렉터리가 없는 경우는 작성
    now = datetime.now().strftime("%Y%m%d%H%M%S")
    save_dir += "_" + experiment_name + os.sep + now
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 테스트 데이터의 적합 (최초 2축만)
    # 표시 영역의 작성
    plt.figure()
    # 추정값과 참값을 동시에 표시하기 위해  hold="on"로 설정한다
    plt.hold("on")
    # x_0 vs y 표시
    plt.subplot(211)
    plt.plot(X_test[:, 0], Y_test, "+", label="True")
    plt.plot(X_test[:, 0], Y_hat, "x", label="Estimate")
    plt.xlabel("x_0")
    plt.ylabel("y")
    plt.legend()
    # x_1 vs y 표시
    plt.subplot(212)
    plt.plot(X_test[:, 1], Y_test, "+")
    plt.plot(X_test[:, 1], Y_hat, "x")
    plt.xlabel("x_1")
    plt.ylabel("y")

    # 파라미터를 파일에 저장
    # NOTE: json형식은 설정 파일 등의 데이터 기술에 편리한 형식입니다
    #       그 실체는 구조화 텍스트 파일입니다
    #       열람할 때도 적당한 텍스트 에디터를 이용해주세요
    #       python의 경우, json을 다루는 모듈이 표준으로 준비되어 있습니다
    #       （그 이름도 json 모듈）
    #       그 외의 데이터 기술 형식으로는 yaml, xml등이 있습니다
    fn_param = "parameters.json"
    with open(save_dir + os.sep + fn_param, "w") as fp:
        json_str = json.dumps(parameters, indent=4)
        fp.write(json_str)

    # 이미지를 파일에 저장
    fn_fit = "fitting.png"  #각종 조건
    plt.savefig(save_dir + os.sep + fn_fit)

    # 손실의 추이가 주어지고 있는 경우는 표시
    if losses is not None:
        train_losses, test_losses = losses
        # NOTE: 손실의 추이는 통상지수적이므로
        #       로그 스케일로 표시하는 경우가 많습니다
        x_train = range(len(train_losses))
        x_test = range(len(test_losses))
        plt.figure()
        plt.plot(
            x_train, np.log(train_losses),
            x_test, np.log(test_losses)
        )
        plt.xlabel("steps")
        plt.ylabel("ln(loss)")
        plt.legend(["training loss", "test loss"])

        fn_loss = "loss.png"
        plt.savefig(save_dir + os.sep + fn_loss)


def flat_nd(xs):
    """
    하나의 numpy.array에 성형해서 반환합니다
    """
    return np.c_[tuple([x.flatten() for x in xs])]


def genearate_original_data(
    dimension=2, nonlinear=False, num_of_samples=10000, noise_amplitude=0.1
):
    """
    그 외의 메서드로 반환하는 변수의 기본이 되는 데이터를 생성합니다
    """
    # 차원은 최저라도 2로 합니다
    if dimension < 2:
        raise ValueError("'dimension' must be larger than 2")

    # NOTE: 입력값 x の범위는 미리 정해두어 [0, 1]로 합니다.
    #       단지, 샘플링을 하는 것은 같은 난수로 결정합니다.
    x_sample = np.random.rand(num_of_samples, dimension)
    # NOTE: 표시용으로 균일하고 노이즈가 없는 데이터도 반환합니다
    #       다차원 데이터는 표시해도 알 수 없으므로
    #       편의상 처음 2차원만 움직이고
    #       그 외의 차원은 모두 상수로 고정합니다
    grid_1d = np.arange(0.0, 1.0, 0.01)
    fixed_coeff = 0.0
    x_grid = flat_nd(np.meshgrid(grid_1d, grid_1d))

    # NOTE: ”정답” 관게식은
    #         f(x) = -1.0 + x_1 + 0.5 * x_2 + Σ_{i>=3} 1/i * x_i
    #                + sin(2πx_1) + cos(2πx_2)
    #                  + Σ_{i>=3, odd} sin(2πx_i)
    #                  + Σ_{i>=4, even} cos(2πx_i)
    #       입니다.
    #       특별히 의미가 있는 식은 아닙니다.
    def f(x):
        # 3차이상의 항은 없는 경우가 있습니다.
        higher_terms = x[:, 2:] / np.arange(2, x.shape[1])
        if len(higher_terms) == 0:
            ht_sum = 0.0
        else:
            ht_sum = np.sum(higher_terms, axis=1)

        # 우선 선형인 항을 넣습니다
        y = -1.0 + 1.0 * x[:, 0] + 0.5 * x[:, 1] + ht_sum

        # 비선형 플래그가 서 있으면 비선형 항을 더합니다
        if nonlinear:
            if len(higher_terms) == 0:
                ht_sum = 0.0
            else:
                PI2 = np.pi*2
                sin = np.sin(PI2*x[:, 2::2])
                cos = np.cos(PI2*x[:, 3::2])
                ht_sum = np.sum(sin) + np.sum(cos)
            y += np.sin(PI2*x[:, 0]) + np.cos(PI2*x[:, 1]) + ht_sum

        return y

    # 출력값을 계산합니다.
    # NOTE: 샘플된 데이터에는 정규 노이즈를 부가합니다.
    noise = noise_amplitude * np.random.randn(x_sample.shape[0])
    y_sample = f(x_sample) + noise

    y_grid = f(x_grid).reshape(x_grid.shape[0])
    # 고정값 추가
    fixed_columns = fixed_coeff * np.ones((x_grid.shape[0], dimension-2))
    x_grid = np.concatenate((x_grid, fixed_columns), axis=1)
    return (
        (x_sample, y_sample),
        (x_grid, y_grid),
    )


def coeff(x):
    """
    생 데이터 x 를 계수행렬로 성형해서 반환합니다
    """
    return np.c_[x, np.ones(x.shape[0])]


def get_all(
    dimension, nonlinear, num_of_samples, noise_amplitude,
    return_coefficient_matrix = True
):
    """
    입력값 x 를 선형회귀의 계수행렬,
    출력값 y를 벡터로
    전체 데이터를 일괄 반환합니다
    """

    # 원래 데이터 구하기
    # NOTE: 격자점 상의 값은 불필요하므로格子点上の値は不要なので
    #       관용적으로 보이지 않는 것을 의미하는 변수명 _로 받아서 무시합니다
    #       어디까지나 관용적인 의미를 붙인 것으로
    #       실제로는 접근할 수 있는 보통 변수이므로 주의해주세요
    data_sample, _ = genearate_original_data(
        dimension, nonlinear, num_of_samples, noise_amplitude
    )
    X, Y = data_sample

    # 학습/테스트 데이터를 정하기 위하여 난수로 인덱스를 선택
    N = X.shape[0]
	
    print(N)
	
    perm_indices = np.random.permutation(range(N))
    train = perm_indices[:int(N/2)]  # 정수연산이므로 내림
    test = perm_indices[int(N/2):]
	
    # 계수행렬로 반환할지
    if return_coefficient_matrix:
        X = coeff(X)

    return(X[train], Y[train]), (X[test], Y[test])


def get_batch(data, batch_size):
    """
    입력값 x, 출력값 y의 튜플 data를
    패치 단위로 잘라 반환합니다
    """
    X, Y = data
    N = len(X)

    # 오름차순으로 정렬한 자연수를 permutation 메서드에서 셔플합니다
    indices = np.random.permutation(np.arange(N))
    # 셔플한 정수열을 패치 크기로 잘라
    # X와 Y 인덱스로 사용합니다
    data_batch = [
        (X[indices[i: i+batch_size]], Y[indices[i: i+batch_size]])
        for i in range(0, N, batch_size)
    ]

    return data_batch


def main():
    """
    이 파일을 단독으로 사용하는 경우
    전체 취득한 데이터 가시화합니다
    """

    # 데이터의 수
    num_of_samples = 1000

    # 비선형 플래그
    # True - > 평면
    # False -> 곡면
    nonlinear = False

    # 데이터 생성
    data_sample, data_grid = genearate_original_data(
        nonlinear=nonlinear, num_of_samples=num_of_samples
    )
    x_sample, y_sample = data_sample
    x_grid, y_grid = data_grid

    # 표시용 성형
    num_of_grid_points = int(np.sqrt(len(y_grid)))
    x_grid_0 = x_grid[:, 0].reshape((num_of_grid_points,)*2)
    x_grid_1 = x_grid[:, 1].reshape((num_of_grid_points,)*2)
    y_grid = y_grid.reshape((num_of_grid_points,)*2)

    # NOTE: 그림을 보면 등고선의 완만한 곳도 경사진 곳도
    #       비슷한 밀도로 점을 취하고 있는 것을 알 수 있습니다
    #       （보기 어려운 경우는 점수를 줄여서 실행해주세요）.
    #       실제로는 함수형을 파악하는데는 경사진 곳의 정보가 중요하므로
    #       균일하게 점을 취해야 좋다는 것은 아니라는 것을 알 수 있습니다.
    plt.figure()
    plt.contour(x_grid_0, x_grid_1, y_grid, levels=np.arange(-2.0, 2.0, 0.1))
    plt.xlabel("x_0")
    plt.ylabel("x_1")
    plt.title("countour of f(x)")
    plt.hold("on")
    plt.scatter(x_sample[:, 0], x_sample[:, 1], color="k", marker="+")

    plt.savefig("original_data.png")

if __name__ == "__main__":
    main()
