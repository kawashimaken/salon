# -*- coding: utf-8 -*-


def calc_pi(n: int) -> float:
    '''
    メソッド：パイを計算する
    ライプニッツ公式
    π=4/1-4/3+4/5-4/7+4/9-4/11...
    '''
    numerator: float = 4.0  # 分子（全ての項）
    denominator: float = 1.0  # 分母（2ずつ増えていく）
    operation: float = 1.0  # 符号（1か-1かで、マイナスとプラスを実現）
    pi: float = 0.0  # 初期値ゼロにセット
    for _ in range(n):
        # n　は無限級数の項の数、大きいければ、円周率πが正確になる
        pi = pi + operation * (numerator / denominator)  # 計算する
        denominator = denominator + 2.0  # 分母を2ずつ増やしていく
        operation = operation * -1.0  # 符号を反転させる（1なら-1に、-1なら1に）
    return pi


if __name__ == "__main__":
    print(calc_pi(10000))  # 計算して出力する
