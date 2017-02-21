# NMT2016

## 1. 事前準備
- Python 3.4.x
- chainer (>ver 1.16)
- hdf5

## 2. モデルのトレーニング

translate.shの以下の変数にパスを与える。

- TRAIN_SOURCE_FILE （原言語コーパス)
- TRAIN_TARGET_FILE （目的言語コーパス）
- MODEL_DIR （モデルを保存するディレクトリ）

TRAIN_SOURCE_FILEとTRAIN_TARGET_FILEを用いてモデルを学習する。  
モデルは各エポックごとにMODEL_DIRに'ファイル名.エポック数'で書き込まれる（デフォルトのファイル名は'epoch'）。  
translate.sh実行時、$1にtrainを引数として与えることで学習開始。
```
./translate.sh train
```



