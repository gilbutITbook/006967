
# -*- coding: utf-8 -*-

import numpy as np
from get_data import get_all, get_batch, visualize_result


# 시그모이드 계산 시에 허용하는 최소값
FLOOR = np.log(np.spacing(1))


def backward(
    Z, E, W, learning_rate, momentum, regularize_coeff, prev_difference
):
    """
    오차 역전파의 계산을 실행합니다
    """
    # 하이퍼 파라미터를 넣는다
    # 로컬인 함수 객체를 작성
    def dW(e, z, w, pd):
        """
        단순한 경사에 모멘트법을 적용해
        배치 내의 합을 취해 반환합니다
        """
        # NOTE: 제0성분이 샘플을 나타내기 때문에
        #       백터끼리의 곱을 제대로 작성할 수 없습니다
        #       그 대신, 브로드 캐스트를 사용해 전체 곱을 취하고 있습니다
        g = e[:, :, np.newaxis] * z[:, np.newaxis, :]
        dW_batch = momentum_method(
            g, w, learning_rate, momentum, regularize_coeff, pd
        )
        return np.sum(dW_batch, axis=0)

    # NOTE: 최종 층에 시그모이드를 걸지 않았기 때문에
    #       본문 중의 식으로 말하는 부분의 최종 층의 갱신율 식은 이용할 수 없습니다
    #       시그모이드를 곱한 경우, 여기에
    #         E = grad_sigmoid(Z[-1]) * E
    #       를 삽입합니다

    # 갱신량
    d = [dW(E, Z[-2], W[-1], prev_difference[-1])]
    # 출력과 가중치를 역방향으로 거슬러 갑니다
    # NOTE: _Zp 는 f(u^(k)) , _Zn 는 z_k에  해당합니다
    for _Wp, _Wn, _Zp, _Zn, pd in zip(
        W[-1::-1], W[-2::-1], Z[-2::-1], Z[-3::-1], prev_difference[-2::-1]
    ):
        # 하나 앞 층의 오차로부터 단순한 경사법의 갱신량을 계산
        E = (_Zp*(1-_Zp)) * E.dot(_Wp)
        # 모멘트 법을 적용한 값을 저장
        d.insert(0, dW(E, _Zn, _Wn, pd))

    # NOTE: 선형회귀의 코드에 합해서 갱신량을 반환하는 메서드로 하고 있으나
    #       이 메서드 안에서 갱신을 시행해도 OK입니다
    return d


def forward(X, W):
    """
    순방향 계산을 실행합니다
    """
    Z = [X]
    for _W in W[:-1]:
        # NOTE: 각 배치를 제0성분에 저장하고 있기 때문에
        #       식과는 달리 다른 위치를 취한 표현으로 되어 있습니다
        Z.append(sigmoid(Z[-1].dot(_W.T)))
    # 회귀문제를 푸는 사정 상, 최종층에서는 시그모이드를 걸지 않습니다
    # 시그모이드 영역은 [0, 1]이고 임의의 실수를 출력할 수 없기 때문입니다
    Z.append(Z[-1].dot(W[-1].T))
    return Z


def sigmoid(X):
    """
    요소 마다 시그모이드 계산을 실행합니다
    """
    # 그대로 X를 사용하면 큰 값으로 오버 플로우가 일어납니다
    # 이것을 피하려면 모든 요소를 0으로 초기화해 두고
    # 충분히 큰 X만 실제 계산에 이용합니다
    out = np.zeros(X.shape)
    stable = (X > FLOOR)  # 안정적인 영역
    out[stable] = 1/(1+np.exp(-X[stable]))
    return out


def momentum_method(
    raw_gradient, current_parameter,
    learning_rate, momentum, regularize_coeff, prev_difference,
):
    """
    모멘트 법을 적용합니다
    """
    # Δw := -α * ∇L + β Δw - γ w
    return (
        - learning_rate * raw_gradient +  # 경사
        momentum * prev_difference -   # 모멘트
        regularize_coeff * current_parameter  # 법칙항
    )


def train_epoch(data, W_hat, learning_rate, momentum, regularize_coeff):
    """
    1에포크 분의 학습을 실행합니다
    """
    difference = [0.0]*len(W_hat)
    losses = []
    for step, (X, Y) in enumerate(data):
        # 손실 계산
        # 순방향 계산
        Z = forward(X, W_hat)

        # 최종층의 오차
        # NOTE: Z[-1]의 차원(m, 1)로 채우기 위해
        #       Y에도 차원을 더합니다
        E = Z[-1] - Y[:, np.newaxis]
        loss = E[:, 0].T.dot(E[:, 0])
        losses.append(loss)

        # 경사와 갱신량의 계산
        difference = backward(
            Z, E,  # 데이터 및 중간층의 출력과 오차
            W_hat,  # 파라미터
            learning_rate, momentum, regularize_coeff,  # 하이퍼 파라미터
            difference,  # 전 회의 갱신량
        )

        # 파라미터를 갱신
        for _W_hat, _difference in zip(W_hat, difference):
            _W_hat += _difference

        # 정기적으로 도중 경과를 표시
        if step % 100 == 0:
            print("step {0:8}: loss = {1:.4e}".format(step, loss))

    # 손실의 평균과 이 에포크에서의 최종 측정값을 반환한다
    return np.mean(losses), W_hat


def init_weights(num_units, prev_num_unit):
    W = []
    # NOTE: 최종층은 1차원이므로 num_units에 [1]을 더합니다
    for num_unit in num_units+[1]:
        # 가중치 크기는（현재 층의 유닛수, 직전 층의 유닛수)입니다
        # NOTE: 오차는 가중치를 붙여 전파하기 때문에
        #       초기값이 0이면 갱신이 시행되지 않습니다
        #       여기에서는 정규난수에 따라 초기화합니다
        #       이 때, 표준오차를
        #         √(2.0/prev_num_unit)
        #       로써 주면 수습하기 쉬운 것이 알려져 있습니다
        #       python2계열에서 실행하는 경우, 2로 하면 정수의 나눗셈이 되어 버리므로 주의합니다
        random_weigts = np.random.randn(num_unit, prev_num_unit)
        normalized_weights = np.sqrt(2.0/prev_num_unit) * random_weigts
        W.append(normalized_weights)
        prev_num_unit = num_unit
    return W


def main():
    # 선형회귀와 같은 파라미터
    # 특징 벡터의 차원의 설정
    dimension = 10
    # 비선형 플래그
    nonlinear = True
    # 전체 데이터 수
    num_of_samples = 1000
    # 노이즈 진폭
    noise_amplitude = 0.01

    # 선형회귀와 공통의 하이퍼 파라미터
    batch_size = 10
    max_epoch = 10000
    learning_rate = 1e-3
    momentum = 0.0
    regularize_coeff = 0.0

    # 뉴럴 네트워크 특유의 하이퍼 파라미터
    # 중간층의 유닛수(채널 수)
    # NOTE: 여기에서는 1층만으로 하지만
    #       리스트에 값을 붙여 나감으로써 층을 추가할 수 있습니다
    #       단지, 메모리 사용량에는 주의하세요!
    num_units = [
        50,
        100,
    ]

    # 전체 데이터 구하기
    data_train, (X_test, Y_test) = get_all(
        dimension, nonlinear, num_of_samples, noise_amplitude,
        return_coefficient_matrix=False
    )

    # 손실의 이력
    train_losses = []
    test_losses = []
    # 파라미터의 초기값
    W_hat = init_weights(num_units, dimension)
    for epoch in range(max_epoch):
        print("epoch: {0}/{1}".format(epoch, max_epoch))
        # 배치 단위로 분할
        data_train_batch = get_batch(data_train, batch_size)
        # 1에포크 분 학습
        train_loss, W_hat = train_epoch(
            data_train_batch,  W_hat,
            learning_rate, momentum, regularize_coeff
        )

        # 결과를 이력에 저장
        train_losses.append(train_loss)

        # 테스트 데이터로의 피팅
        Y_hat = forward(X_test, W_hat)[-1][:, 0]
        E = Y_hat - Y_test
        test_loss = E.T.dot(E)
        test_losses.append(test_loss)

    # 실험 기록용
    parameters = {
        "linearity": "nonlinear" if nonlinear else "linear",
        "dimension": dimension,
        "num_of_samples": num_of_samples,
        "batch_size": batch_size,
        "max_epoch": max_epoch,
        "learning_rate": learning_rate,
        "momentum": momentum,
        "regularize_coeff": regularize_coeff,
        "num_units": num_units,
    }
    # 결과 표시
    visualize_result(
        "neural_network",
        X_test[:, 0:2], Y_test, Y_hat, parameters,
        losses=(train_losses, test_losses)
    )


if __name__ == "__main__":
    main()
