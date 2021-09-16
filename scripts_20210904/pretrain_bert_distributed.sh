#!/bin/bash

GPUS_PER_NODE=3
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

## 预处理数据路径
DATA_PATH=experiments/outputs/pretraining/data_0904/unlabeled_data_text_sentence 
## 模型加载和保存路径
CHECKPOINT_PATH=experiments/outputs/pretraining/bert_tmp_0905_1
## 词表路径
VOCAB_FILE=resources/daguan_bert_base_v3/steps_120k/vocab.txt

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

## global-batch-size 384 需要整除你的GPU数量
## micro-batch-size 128 batch 的数量
## checkpoint-activations 梯度优化，显存不足的时候使用，分批更新
python -m torch.distributed.launch $DISTRIBUTED_ARGS \
       pretrain_bert.py \
       --num-layers 12 \
       --hidden-size 768 \
       --num-attention-heads 12 \
       --micro-batch-size 128 \
       --global-batch-size 384 \
       --seq-length 512 \
       --max-position-embeddings 512 \
       --train-iters 1000000 \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATA_PATH \
       --vocab-file $VOCAB_FILE \
       --data-impl mmap \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.0001 \
       --lr-decay-style linear \
       --min-lr 1.0e-5 \
       --lr-decay-iters 990000 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --lr-warmup-fraction .01 \
       --log-interval 100 \
       --save-interval 10000 \
       --eval-interval 1000 \
       --eval-iters 10 \
       --fp16 \
       --checkpoint-activations \
       --activations-checkpoint-method uniform


# gradient_checkpointing:
