# -*- coding: utf-8 -*-

#
#  サロン内の質問 2020/02/07
#


def first_question():
    '''
    階層1の質問処理
    '''
    first_input = input('階層1の質問：何をお探しですか、ほしいパーツを入力してください。>')
    response = ''
    if 'cpu' in first_input:
        second_response = second_question()
        return second_response

    if 'motherboard' in first_input:
        response = 'マザーボードですね'
        return response

    if 'gpu' in first_input:
        response = 'GPUですね'
        return response
    else:
        response = '何か入力してください'
        return response


def second_question():
    '''
    階層2の質問処理
    '''
    while True:
        second_input = input('どんなcpuがお探しですか？、aかbを入力してね')
        response = ''
        if 'a' in second_input:
            response = 'Aタイプですね'
            return response
        elif 'b' in second_input:
            response = 'Bタイプですね'
            return response
        else:
            print('aかbを入力してください')


if __name__ == '__main__':
    while True:
        response = first_question()
        # 質問の結果を表示する
        print(response)
