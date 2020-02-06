# -*- coding: utf-8 -*-

while True:
    command = input('pybot...何をお探しですか、ほしいパーツを入力してください。>')
    resoponse = ''
    if 'cpu' in command:
        resoponse = 'cpuはこちらです。'
        print(resoponse)
        command = input('pybot...こちらはどうですか？yesかnoでお答えください。 >')
        if 'yes' in command:
            resoponse = 'yesを選んでくれてありがとう'
            print(resoponse)
        if 'no' in command:
            resoponse = 'noを選んでしまったんだね'
            print(resoponse)
    if 'さようなら' in command:
        break
