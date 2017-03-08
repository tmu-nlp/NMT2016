#!/bin/sh

MODE=''
SOURCE_FILE=''
TARGET_FILE=''
OUTPUT_FILE=''
MODEL_DIR=''
USE_GPU=1
GPU_DEVICE=0
VOCAB_SIZE=30000
EMBED_SIZE=512
HIDDEN_SIZE=512
MAXOUT_SIZE=512
EPOCH_NUM=30
BATCH_SIZE=10
POOLING=100
OPTIMIZER='Adagrad'
LEARNING_RATE=0.01
GENERATION_LIMIT=60
WORD2VEC_SOURCE=''
WORD2VEC_TARGET=''
MODEL_NUM='001'

while getopts a:s:t:o:m:n:u:g:v:e:h:x:c:b:p:z:l:i: OPT
do
    case $OPT in
        a ) MODE=$OPTARG ;; 
        s ) SOURCE_FILE=$OPTARG ;; 
        t ) TARGET_FILE=$OPTARG ;; 
        o ) OUTPUT_FILE=$OPTARG ;;
        m ) MODEL_DIR=$OPTARG ;;
        n ) MODEL_NUM=$OPTARG ;;
        u ) USE_GPU=$OPTARG ;;
        g ) GPU_DEVICE=$OPTARG ;;
        v ) VOCAB_SIZE=$OPTARG ;;
        e ) EMBED_SIZE=$OPTARG ;;
        h ) HIDDEN_SIZE=$OPTARG ;;
        x ) MAXOUT_SIZE=$OPTARG ;;
        c ) EPOCH_NUM=$OPTARG ;;
        b ) BATCH_SIZE=$OPTARG ;;
        p ) POOLING=$OPTARG ;;
        z ) OPTIMIZER=$OPTARG ;;
        l ) LEARNING_RATE=$OPTARG ;;
        i ) GENERATION_LIMIT=$OPTARG ;;
    esac
done

if [ "$MODE" = "train" ]; then
    mkdir $MODEL_DIR
    MODEL=$MODEL_DIR/epoch
elif [ "$MODE" = "test" ]; then
    MINIBATCH=1
    MODEL=$MODEL_DIR/epoch.$MODEL_NUM
    TARGET_FILE=$OUTPUT_FILE
fi

python attentionNMT.py --mode "$MODE" --source "$SOURCE_FILE" --target "$TARGET_FILE" \
       --model "$MODEL" --use_gpu "$USE_GPU" --gpu_device "$GPU_DEVICE" --vocab "$VOCAB_SIZE" \
       --embed "$EMBED_SIZE" --hidden "$HIDDEN_SIZE" --maxout "$MAXOUT_SIZE" --epoch "$EPOCH_NUM" \
       --minibatch "$BATCH_SIZE" --pooling "$POOLING" --optimizer "$OPTIMIZER" --learning_rate "$LEARNING_RATE" \
       --word2vec_source "$WORD2VEC_SOURCE" --word2vec_target "$WORD2VEC_TARGET"

