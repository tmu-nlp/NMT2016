#!/bin/sh

TRAIN_SOURCE_FILE=''
TRAIN_TARGET_FILE=''
DEV_SOURCE_FILE=''
DEV_TARGET_FILE=''
TEST_SOURCE_FILE=''
TEST_TARGET_FILE=''

MODEL_DIR=''
VOCAB_SIZE=30000
EMBED_SIZE=512
HIDDEN_SIZE=512
MAXOUT_SIZE=512
EPOCH_NUM=30
MINIBATCH=10
POOLING=100
GENERATION_LIMIT=60
#WORD2VEC_SOURCE=''
#WORD2VEC_TARGET=''
OPTIMIZER='Adagrad'
LEARNING_RATE=0.01
GPU_DEVICE=3

MODEL_NUM='001'

if [ $1 = 'train' ]; then  
    mkdir $MODEL_DIR
    python attentionNMT.py --mode train --source $TRAIN_SOURCE_FILE --target $TRAIN_TARGET_FILE \
    --model $MODEL_DIR/epoch --vocab $VOCAB_SIZE --embed $EMBED_SIZE --hidden $HIDDEN_SIZE \
    --maxout $MAXOUT_SIZE --epoch $EPOCH_NUM --minibatch $MINIBATCH --pooling $POOLING  \
    --gpu-device $GPU_DEVICE --optimizer $OPTIMIZER --learning_rate $LEARNING_RATE \
    #--word2vec_source $WORD2VEC_SOURCE --word2vec_target $WORD2VEC_TARGET
elif [ $1 = 'dev' ]; then
    python attentionNMT.py --mode test --source $DEV_SOURCE_FILE --target $DEV_TARGET_FILE \
    --model $MODEL_DIR/epoch.$MODEL_NUM --minibatch 1 --generation_limit $GENERATION_LIMIT --gpu-device $GPU_DEVICE
elif [ $1 = 'test' ]; then
    python attentionNMT.py --mode test --source $TEST_SOURCE_FILE --target $TEST_TARGET_FILE \
    --model $MODEL_DIR/epoch.$MODEL_NUM --minibatch 1 --generation_limit $GENERATION_LIMIT --gpu-device $GPU_DEVICE
else
    echo 'set the first argument $1 ./translate.sh {} with train, dev, or test' 
fi

