#!/bin/bash

DATASET="cifar10"
ARCH="resnet18-bn"
NUM_MAIN_RUNS=1
NUM_BASELINE_REPEATS=1
EPOCHS_MAIN=1
EPOCHS_BASELINE=1

for i in $(seq 1 $NUM_MAIN_RUNS)
do
    RUN_NAME=$(printf "run_%02d" $i)
    echo "================================================="
    echo "STEP 1: RUNNING MAIN DISTILLATION ($RUN_NAME)"
    echo "================================================="
    
    poetry run python -m src.main \
        --dataset $DATASET \
        --run-name $RUN_NAME \
        --run-distill \
        --epochs $EPOCHS_MAIN
        #--seed $i
        
    echo "-------------------------------------------------"
    echo "STEP 2: RUNNING 10 BASELINE REPEATS FOR $RUN_NAME"
    echo "-------------------------------------------------"
    
    for j in $(seq 1 $NUM_BASELINE_REPEATS)
    do
        REPEAT_DIR="runs/${RUN_NAME}/baselines_eval/repeat_${j}"
        
        poetry run python -m src.train_baselines \
            --dataset $DATASET \
            --arch $ARCH \
            --subset-folder "runs/${RUN_NAME}/subsets/" \
            --out "$REPEAT_DIR" \
            --epochs $EPOCHS_BASELINE 
            #--seed $j 
    done
done
