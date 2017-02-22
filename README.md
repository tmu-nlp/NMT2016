# NMT2016

ニューラル機械翻訳(Bahdanau et al., 2015)の実装  
RNNの各ユニットはLSTMで構成  

## 1. 事前準備
- Python 3.4.x
- chainer (>ver 1.16)
- hdf5

## 2. モデルのトレーニング

translate.shの以下の変数でパスを指定 　

- TRAIN_SOURCE_FILE （原言語コーパス)
- TRAIN_TARGET_FILE （目的言語コーパス）
- MODEL_DIR （モデルを保存するディレクトリ）

TRAIN_SOURCE_FILEとTRAIN_TARGET_FILEを用いてモデルを学習する。  
モデルは各エポックごとにMODEL_DIRに'ファイル名.エポック数'で書き込まれる（デフォルトのファイル名は'epoch'）。  
translate.sh実行時、$1にtrainを引数として与えることで学習開始。　
※GPUを用いない場合（USE_GPU=0）、学習にかなりの時間を要します。
```
./translate.sh train
```

## 3. テスト

translate.shの以下の変数でパスを指定

- TEST_SOURCE_FILE
- TEST_TARGET_FILE 

以下を実行（minibatchは1で固定）
```
./translate.sh test
```
