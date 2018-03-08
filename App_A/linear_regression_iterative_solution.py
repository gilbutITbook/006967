
# -*- coding: utf-8 -*-

import numpy as np
from get_data import get_all, get_batch, visualize_result, coeff


def raw_gradient(A, E):
    """
    기울기를 계산합니다
    """
    # NOTE: 선형회귀 최소제곱법은 데이터 덩어리에 대해서 하나의 경사를 계산하기 때문에
    #       미니배치법의 설명에서 말한
    #       샘플마다의 경사합을 취하는 처리는 하지 않습니다
    return A.T.dot(E)


def momentum_method(
    A, E, current_parameter,
    learning_rate, momentum, regularize_coeff, prev_difference,
):
    """
    모멘트법을 적용합니다
    """
    # Δw := -α * ∇L + β Δw + γ w
    return (
        - learning_rate * raw_gradient(A, E) +  # 경사
        momentum * prev_difference -  # 모멘트
        regularize_coeff * current_parameter  # 벌칙항
    )


def train_epoch(data, D_hat, learning_rate, momentum, regularize_coeff):
    """
    1에포크 분의 학습을 실행합니다
    """
    difference = 0.0
    losses = []
    for step, (X, Y) in enumerate(data):
        # 계수행렬로 변형
        A = coeff(X)

        # 손실 게산
        E = A.dot(D_hat) - Y
        loss = E.T.dot(E)
        losses.append(loss)

        # 경사과 갱신량의 계산
        difference = momentum_method(
            A, E, D_hat,  # 데이터
            learning_rate, momentum, regularize_coeff,  # 하이퍼 파라미터
            difference,  # 앞의 갱신량
        )

        # 파라미터를 갱신
        D_hat += difference

        # 정기적으로 도중경과를 표시
        if step % 100 == 0:
            print("step {0:8}: loss = {1:.4e}".format(step, loss))

    # 손실의 평균과 이 에포크에서의 최종 측정값을 반환한다
    return np.mean(losses), D_hat


def main():
    # 선형회귀와 같은 파라미터
    # 특징 벡터의 차원 설정
    dimension = 10
    # 비선형 플래그
    nonlinear = False
    # 전체 데이터의 수
    num_of_samples = 1000
    # 노이즈의 진폭
    noise_amplitude = 0.01

    # 하이퍼 파라미터의 설정
    batch_size = 10
    max_epoch = 10000
    learning_rate = 1e-3
    momentum = 0.9  # 이 값을  하면 모멘트법이 됩니다
    regularize_coeff = 0.0  # 이 값을 하면 L2 노름에 의한 법칙이 걸립니다

    # 전체 데이터 구하기
    # NOTE: 여기에서는 미니배치 동작만을 보기 위해서
    #       일단 전체 데이터를 취득하고 나서 배치 단위로 잘라서 반환합니다
    #       그러나, 미니배치법이 필요하게 되는 상황에서는
    #       전체 데이터를 한번에 읽어 들이지 않는 경우가 보통이므로
    #       수 배치분을 읽어 들여 버퍼하고 나서 1배치만 반환해야 합니다
    #       이러한 경우, python의 기능을 살려
    #       순서를 따라 읽어 들이는 제너레이터를 작성하는 것이 유효합니다
    data_train, (X_test, Y_test) = get_all(
        dimension, nonlinear, num_of_samples, noise_amplitude,
        return_coefficient_matrix=False
    )
    A_test = coeff(X_test)

    # 손실의 이력
    train_losses = []
    test_losses = []
    # 파라미터의 초기값
    D_hat = np.zeros(dimension+1)
    # 에포크에 대해서의 루프
    for epoch in range(max_epoch):
        print("epoch: {0} / {1}".format(epoch, max_epoch))
        # 배치 단위로 분할
        data_train_batch = get_batch(data_train, batch_size)
        # 1에포크 분 학습
        loss, D_hat = train_epoch(
            data_train_batch, D_hat,
            learning_rate, momentum, regularize_coeff
        )

        # 손실을 이력에 저장
        train_losses.append(loss)

        # 전형적인 코드에서는 어떤 에포크에 한 번 테스트를 실시하고
        # 도중 경과가 어느 정도의 범화성능을 나타내는지 확인하지만
        # 여기에서는 매회 테스트를 실시하고 있습니다
        Y_hat = A_test.dot(D_hat)
        E = Y_hat - Y_test
        test_loss = E.T.dot(E)
        test_losses.append(test_loss)

    # 실험기록용
    parameters = {
        "linearity": "nonlinear" if nonlinear else "linear",
        "dimension": dimension,
        "num_of_samples": num_of_samples,
        "batch_size": batch_size,
        "max_epoch": max_epoch,
        "learning_rate": learning_rate,
        "momentum": momentum,
        "regularize_coeff": regularize_coeff,
    }
    # 결과 표시
    visualize_result(
        "linear_regression_iterative_solution",
        A_test[:, 0:2], Y_test, Y_hat, parameters,
        losses=(train_losses, test_losses)
    )

if __name__ == "__main__":
    main()
