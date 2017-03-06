# NMT2016

ニューラル機械翻訳(Bahdanau et al., 2015)の実装  
RNNの各ユニットはLSTMで構成  

## 1. 事前準備
- Python 3.4.x
- chainer (>ver 1.16)
- hdf5

## 2. モデルのトレーニング

translate.shの以下の変数でパスを指定。()内は実行時オプション。

- MODE (-a)
- SOURCE_FILE (-s)
- TARGET_FILE （-o）
- MODEL_DIR (-m）

MODE: train  
SOURCE_FILE: 原言語コーパス  
TARGET_FILE: 目的言語コーパス  
MODEL_DIR: モデルを保存するためのディレクトリ   

SOURCE_FILEとTARGET_FILEに含まれる文は①単語分割済み、②1行1文、を満たしていること。  
```  
私 は 日本人 です 。
I am Japanese .
```
モデルは各エポックごとにMODEL_DIRに'ファイル名.エポック数'で書き込まれる（デフォルトのファイル名は'epoch'）。    
GPUを用いない場合（USE_GPU=0）、学習にかなりの時間を要します。  
上記の変数を指定したのち、./translate.shを実行で学習開始。  

## 3. テスト

translate.shの以下の変数でパスを指定

- MODE (-a)
- SOURCE_FILE (-s)
- OUTPUT_FILE (-o)
- MODEL_DIR (-m)
- MODEL_NUM (-n)

MODE: test  
OUTPUT_FILE: 出力を書き込むファイル  
MODEL_DIR: モデルを読み込むディレクトリ  
MODEL_NUM: エポック数  
上記の変数を指定したのち、./translate.shを実行でテスト開始。

