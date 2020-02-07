# -*- coding: utf-8 -*-

# PyTorchでVGG-16の学習済みモデルを使って、犬の分類
# -----------------------------------------------------------------------------
# まず、必要なファイルをダウンロードしてください
# cd data ない場合は、mkdir data && cd data
# curl -O https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json

# VGG-16はILSVRC　2014年で第2位
# http: // www.image - net.org / challenges / LSVRC /
# ILSVRC:ImageNet Large Scale Visual Recognition Challengs

import numpy as np
import json
from PIL import Image

from torchvision import models, transforms
# -----------------------------------------------------------------------------
# MacOSで、下記のエラーに遭遇する人は、下の二行が必要
# OMP: Error #15: Initializing libiomp5.dylib, but found libiomp5.dylib already initialized.
# OMP: Hint This means that multiple copies of the OpenMP runtime have been linked into the program. That is dangerous, since it can degrade performance or cause incorrect results. The best thing to do is to ensure that only a single OpenMP runtime is linked into the process, e.g. by avoiding static linking of the OpenMP runtime in any library. As an unsafe, unsupported, undocumented workaround you can set the environment variable KMP_DUPLICATE_LIB_OK=TRUE to allow the program to continue to execute, but that may cause crashes or silently produce incorrect results. For more information, please see http://www.intel.com/software/products/support/.
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class vgg16_demo():
    def __init__(self):
        '''
        コンストラクタ
        ここで初期化を行う
        '''
        # 推論してもらう画像
        self.input_image = None
        # 学習済みモデル
        self.model = None
        # 1000クラスのラベル
        self.class_index = None
        #
        self.transformed_image = None
        # モデルに渡すためのフォーマット変換済み画像データ
        self.ready_for_model_image = None
        # 分類器
        self.predictor = None
        # 変換器
        self.transformer = None
        # なぜ224にリサイズするかというと
        # VGG-16 では 224x224 の画像を入力として受け付けするので, ここで、画像を224にリサイズする必要がある
        # リサイズ先の画像の大きさ
        self.resize_to = 224
        # 各色チャネルの平均値
        self.mean = (0.5, 0.5, 0.5)
        # 各色チャネルの標準偏差
        self.standard_deviations = (0.3, 0.3, 0.3)

    def _step1_load_model(self):
        # 学習済みのVGG-16モデルをロードする
        # １回目の実行では、学習済みパラメータをダウンロードするため、実行に時間がかかる
        # VGG-16モデルのモデルを生成する
        # 学習済みのパラメータを使用すると宣言する
        use_pretrained = True
        self.model = models.vgg16(pretrained=use_pretrained)
        # 推論モードに設定する
        self.model.eval()

        # モデルのネットワーク構成を出力する
        print('VGG-16モデル', self.model)

    def _step2_prepare_inpu_image(self):
        '''
        画像像読み込み
        テスト用の画像は、適宜見つけて、dataフォルダに格納しておいてください
        このファイル名は、実際のファイル名と一致させてください。
        '''
        # image_file_path = './data/dog_vizsla.jpg'
        image_file_path = './data/dog_weimaraner.jpg'
        self.input_image = Image.open(image_file_path)

    def _step3_transform(self):
        '''
        学習済みに渡せるために、画像データを変換する処理です
        '''

        # 変換器を使って画像データを変換する
        # torch.Size([3, 224, 224])
        self.transformed_image = self._transform(self.input_image,
                                                 self.resize_to, self.mean,
                                                 self.standard_deviations)

        # torch.Size([1, 3, 224, 224])
        self.ready_for_model_image = self.transformed_image.unsqueeze_(0)

    def _step4_predict(self):
        '''
        推論する
        '''
        # ラベル情報をロードする
        self.class_index = json.load(
            open('./data/imagenet_class_index.json', 'r'))

        # モデルに入力し、モデル出力をラベルに変換する
        # torch.Size([1, 1000])
        result = self.model(self.ready_for_model_image)
        anwser = self._predict_max(result)

        # 予測結果を出力する
        print("予測結果：", anwser)

    def _predict_max(self, out_put):
        '''
        予測正解率の一番高いラベルを返す
        '''
        max_id = np.argmax(out_put.detach().numpy())
        result_class_name = self.class_index[str(max_id)][1]

        return result_class_name

    def _transform(self, input_image, resize, mean, standard_deviations):
        '''
        入力画像のサイズをリサイズし、色を標準化する。
        '''
        transformer = transforms.Compose([
            # 短い辺の長さがresizeの大きさになる
            transforms.Resize(resize),
            # 画像中央を基準にしてresize × resizeで切り取りする
            transforms.CenterCrop(resize),
            # データの型をTensorに変換する
            transforms.ToTensor(),
            # 色情報を標準化する
            transforms.Normalize(mean, standard_deviations)
        ])
        return transformer(input_image)


if __name__ == '__main__':
    # デモのインスタンスを作る
    demo = vgg16_demo()
    # メソッドを順次呼び出して実行していく
    demo._step1_load_model()
    demo._step2_prepare_inpu_image()
    demo._step3_transform()
    demo._step4_predict()
