#!/bin/bash
MAX_JOBS=4
GPUS=(0 1 2 3 4 5 6 7)
TOTAL_GPUS=${#GPUS[@]}

get_gpu_allocation(){
    local job_number=$1
    # Calculate which GPU to allocate based on the job number
    local gpu_id=${GPUS[$((job_number % TOTAL_GPUS))]}
    echo $gpu_id
}

check_jobs(){
    while true; do
        jobs_count=$(jobs -p | wc -l)
        if [ "$jobs_count" -lt "$MAX_JOBS" ]; then
            break
        fi
        sleep 1
    done
}

job_number=0

DATA_ROOT=/home/home_new/panlc/workspace/TSF-PCA/dataset
EXP_NAME=finetune
seed=2023
des='iTransformer'

model_name=iTransformer
datasets=(ETTh1)


# hyper-parameters
dst=ETTh1

train_epochs=30
patience=5
test_batch_size=1
use_revin=1
model_type=linear
dropout=0.5
cycle=24
first_order=1
auxi_loss=MSE
overlap_ratio=0.0
reg_lambda=0.0
lambda=1.0
auxi_batch_size=64
lradj=type1
max_norm=5.0
rerun=1

pl_list=(96 192 336 720)
# NOTE: ETTh1 settings


for pl in ${pl_list[@]}; do
    if ! [[ " ${datasets[@]} " =~ " ${dst} " ]]; then
        continue
    fi

    case $pl in
        96) lr=0.005 lr_inner=0.005 lr_meta=0.2 meta_steps=300 num_tasks=3 meta_inner_steps=3 batch_size=32;;
        192) lr=0.005 lr_inner=0.005 lr_meta=0.02 meta_steps=300 num_tasks=1 meta_inner_steps=3 batch_size=32;;
        336) lr=0.002 lr_inner=0.002 lr_meta=0.01 meta_steps=300 num_tasks=3 meta_inner_steps=4 batch_size=32;;
        720) lr=0.005 lr_inner=0.005 lr_meta=0.01 meta_steps=300 num_tasks=5 meta_inner_steps=2 batch_size=32;;
    esac

    rl=$lambda
    ax=$(echo "1 - $lambda" | bc)
    decimal_places=$(echo "$lambda" | awk -F. '{print length($2)}')
    ax=$(printf "%.${decimal_places}f" $ax)

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lr_inner}_${lr_meta}_${lradj}_${train_epochs}_${patience}_${batch_size}_${auxi_batch_size}_${reg_lambda}_${max_norm}_${num_tasks}_${overlap_ratio}_${meta_inner_steps}_${auxi_loss}_${cycle}_${dropout}_${model_type}_${use_revin}_${first_order}_${meta_steps}
    OUTPUT_DIR="./results/${EXP_NAME}/${JOB_NAME}"

    CHECKPOINTS=$OUTPUT_DIR/checkpoints/
    RESULTS=$OUTPUT_DIR/results/
    TEST_RESULTS=$OUTPUT_DIR/test_results/
    LOG_PATH=$OUTPUT_DIR/result_long_term_forecast.txt

    mkdir -p "${OUTPUT_DIR}/"
    # if rerun, remove the previous stdout
    if [ $rerun -eq 1 ]; then
        rm -rf "${OUTPUT_DIR}/stdout.log"
    else
        subdirs=("$RESULTS"/*)
        if [ ${#subdirs[@]} -eq 1 ] && [ -f "${subdirs[0]}/metrics.npy" ]; then
            echo ">>>>>>> Job: $JOB_NAME already run, skip <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
            continue
        fi
    fi


    check_jobs
    # Get GPU allocation for this job
    gpu_allocation=$(get_gpu_allocation $job_number)
    # Increment job number for the next iteration
    ((job_number++))

    echo "Running command for $JOB_NAME"
    {
        # Set CUDA_VISIBLE_DEVICES for this script and run it in the background
        CUDA_VISIBLE_DEVICES=$gpu_allocation python -u run.py \
            --task_name long_term_forecast_meta_ml3 \
            --is_training 1 \
            --root_path $DATA_ROOT/ETT-small/ \
            --data_path ETTh1.csv \
            --model_id "${dst}_96_${pl}" \
            --model ${model_name} \
            --data_id $dst \
            --data ETTh1 \
            --features M \
            --seq_len 96 \
            --label_len 48 \
            --pred_len ${pl} \
            --enc_in 7 \
            --dec_in 7 \
            --c_out 7 \
            --factor 3 \
            --des ${des} \
            --learning_rate ${lr} \
            --lradj ${lradj} \
            --train_epochs ${train_epochs} \
            --patience ${patience} \
            --batch_size ${batch_size} \
            --test_batch_size ${test_batch_size} \
            --itr 1 \
            --rec_lambda ${rl} \
            --auxi_lambda ${ax} \
            --reg_lambda ${reg_lambda} \
            --auxi_batch_size ${auxi_batch_size} \
            --fix_seed ${seed} \
            --checkpoints $CHECKPOINTS \
            --results $RESULTS \
            --test_results $TEST_RESULTS \
            --log_path $LOG_PATH \
            --rerun $rerun \
            --inner_lr $lr_inner \
            --meta_lr $lr_meta \
            --meta_inner_steps $meta_inner_steps \
            --overlap_ratio $overlap_ratio \
            --num_tasks $num_tasks \
            --max_norm $max_norm \
            --auxi_loss ${auxi_loss} \
            --model_type $model_type \
            --cycle $cycle \
            --use_revin $use_revin \
            --dropout $dropout \
            --first_order $first_order \
            --warmup_steps $meta_steps

        sleep 5
    } 2>&1 | tee -a "${OUTPUT_DIR}/stdout.log" &
done







# hyper-parameters
dst=ETTh2

train_epochs=30
patience=5
test_batch_size=1
use_revin=1
model_type=linear
dropout=0.5
cycle=24
first_order=1
auxi_loss=MSE
overlap_ratio=0.0
reg_lambda=0.0
lambda=1.0
auxi_batch_size=64
lradj=type1
max_norm=5.0
rerun=0

pl_list=(96 192 336 720)
# NOTE: ETTh2 settings



for pl in ${pl_list[@]}; do
    if ! [[ " ${datasets[@]} " =~ " ${dst} " ]]; then
        continue
    fi

    case $pl in
        96) lr=0.0005 lr_inner=0.0005 lr_meta=0.2 meta_steps=300 num_tasks=1 meta_inner_steps=2 batch_size=32;;
        192) lr=0.0005 lr_inner=0.0005 lr_meta=0.01 meta_steps=300 num_tasks=4 meta_inner_steps=1 batch_size=32;;
        336) lr=0.0005 lr_inner=0.0005 lr_meta=0.2 meta_steps=300 num_tasks=1 meta_inner_steps=2 batch_size=32;;
        720) lr=0.0005 lr_inner=0.0005 lr_meta=0.2 meta_steps=300 num_tasks=2 meta_inner_steps=4 batch_size=32;;
    esac

    rl=$lambda
    ax=$(echo "1 - $lambda" | bc)
    decimal_places=$(echo "$lambda" | awk -F. '{print length($2)}')
    ax=$(printf "%.${decimal_places}f" $ax)

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lr_inner}_${lr_meta}_${lradj}_${train_epochs}_${patience}_${batch_size}_${auxi_batch_size}_${reg_lambda}_${max_norm}_${num_tasks}_${overlap_ratio}_${meta_inner_steps}_${auxi_loss}_${cycle}_${dropout}_${model_type}_${use_revin}_${first_order}_${meta_steps}
    OUTPUT_DIR="./results/${EXP_NAME}/${JOB_NAME}"

    CHECKPOINTS=$OUTPUT_DIR/checkpoints/
    RESULTS=$OUTPUT_DIR/results/
    TEST_RESULTS=$OUTPUT_DIR/test_results/
    LOG_PATH=$OUTPUT_DIR/result_long_term_forecast.txt

    mkdir -p "${OUTPUT_DIR}/"
    # if rerun, remove the previous stdout
    if [ $rerun -eq 1 ]; then
        rm -rf "${OUTPUT_DIR}/stdout.log"
    else
        subdirs=("$RESULTS"/*)
        if [ ${#subdirs[@]} -eq 1 ] && [ -f "${subdirs[0]}/metrics.npy" ]; then
            echo ">>>>>>> Job: $JOB_NAME already run, skip <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
            continue
        fi
    fi


    check_jobs
    # Get GPU allocation for this job
    gpu_allocation=$(get_gpu_allocation $job_number)
    # Increment job number for the next iteration
    ((job_number++))

    echo "Running command for $JOB_NAME"
    {
        # Set CUDA_VISIBLE_DEVICES for this script and run it in the background
        CUDA_VISIBLE_DEVICES=$gpu_allocation python -u run.py \
            --task_name long_term_forecast_meta_ml3 \
            --is_training 1 \
            --root_path $DATA_ROOT/ETT-small/ \
            --data_path ETTh2.csv \
            --model_id "${dst}_96_${pl}" \
            --model ${model_name} \
            --data_id $dst \
            --data ETTh2 \
            --features M \
            --seq_len 96 \
            --label_len 48 \
            --pred_len ${pl} \
            --enc_in 7 \
            --dec_in 7 \
            --c_out 7 \
            --factor 3 \
            --des ${des} \
            --learning_rate ${lr} \
            --lradj ${lradj} \
            --train_epochs ${train_epochs} \
            --patience ${patience} \
            --batch_size ${batch_size} \
            --test_batch_size ${test_batch_size} \
            --itr 1 \
            --rec_lambda ${rl} \
            --auxi_lambda ${ax} \
            --reg_lambda ${reg_lambda} \
            --auxi_batch_size ${auxi_batch_size} \
            --fix_seed ${seed} \
            --checkpoints $CHECKPOINTS \
            --results $RESULTS \
            --test_results $TEST_RESULTS \
            --log_path $LOG_PATH \
            --rerun $rerun \
            --inner_lr $lr_inner \
            --meta_lr $lr_meta \
            --meta_inner_steps $meta_inner_steps \
            --overlap_ratio $overlap_ratio \
            --num_tasks $num_tasks \
            --max_norm $max_norm \
            --auxi_loss ${auxi_loss} \
            --model_type $model_type \
            --cycle $cycle \
            --use_revin $use_revin \
            --dropout $dropout \
            --first_order $first_order \
            --warmup_steps $meta_steps

        sleep 5
    } 2>&1 | tee -a "${OUTPUT_DIR}/stdout.log" &
done






# hyper-parameters
dst=ETTm1

train_epochs=30
patience=5
test_batch_size=1
use_revin=1
model_type=linear
dropout=0.5
cycle=96
first_order=1
auxi_loss=MSE
overlap_ratio=0.0
reg_lambda=0.0
lambda=1.0
auxi_batch_size=64
lradj=type1
max_norm=5.0
rerun=0


pl_list=(96 192 336 720)
# NOTE: ETTm1 settings



for pl in ${pl_list[@]}; do
    if ! [[ " ${datasets[@]} " =~ " ${dst} " ]]; then
        continue
    fi

    case $pl in
        96) lr=0.001 lr_inner=0.001 lr_meta=0.1 meta_steps=500 num_tasks=2 meta_inner_steps=1 batch_size=32;;
        192) lr=0.001 lr_inner=0.001 lr_meta=0.01 meta_steps=500 num_tasks=2 meta_inner_steps=3 batch_size=32;;
        336) lr=0.001 lr_inner=0.001 lr_meta=0.2 meta_steps=500 num_tasks=4 meta_inner_steps=1 batch_size=32;;
        720) lr=0.001 lr_inner=0.001 lr_meta=0.2 meta_steps=500 num_tasks=2 meta_inner_steps=5 batch_size=32;;
    esac

    rl=$lambda
    ax=$(echo "1 - $lambda" | bc)
    decimal_places=$(echo "$lambda" | awk -F. '{print length($2)}')
    ax=$(printf "%.${decimal_places}f" $ax)

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lr_inner}_${lr_meta}_${lradj}_${train_epochs}_${patience}_${batch_size}_${auxi_batch_size}_${reg_lambda}_${max_norm}_${num_tasks}_${overlap_ratio}_${meta_inner_steps}_${auxi_loss}_${cycle}_${dropout}_${model_type}_${use_revin}_${first_order}_${meta_steps}
    OUTPUT_DIR="./results/${EXP_NAME}/${JOB_NAME}"

    CHECKPOINTS=$OUTPUT_DIR/checkpoints/
    RESULTS=$OUTPUT_DIR/results/
    TEST_RESULTS=$OUTPUT_DIR/test_results/
    LOG_PATH=$OUTPUT_DIR/result_long_term_forecast.txt

    mkdir -p "${OUTPUT_DIR}/"
    # if rerun, remove the previous stdout
    if [ $rerun -eq 1 ]; then
        rm -rf "${OUTPUT_DIR}/stdout.log"
    else
        subdirs=("$RESULTS"/*)
        if [ ${#subdirs[@]} -eq 1 ] && [ -f "${subdirs[0]}/metrics.npy" ]; then
            echo ">>>>>>> Job: $JOB_NAME already run, skip <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
            continue
        fi
    fi


    check_jobs
    # Get GPU allocation for this job
    gpu_allocation=$(get_gpu_allocation $job_number)
    # Increment job number for the next iteration
    ((job_number++))

    echo "Running command for $JOB_NAME"
    {
        # Set CUDA_VISIBLE_DEVICES for this script and run it in the background
        CUDA_VISIBLE_DEVICES=$gpu_allocation python -u run.py \
            --task_name long_term_forecast_meta_ml3 \
            --is_training 1 \
            --root_path $DATA_ROOT/ETT-small/ \
            --data_path ETTm1.csv \
            --model_id "${dst}_96_${pl}" \
            --model ${model_name} \
            --data_id $dst \
            --data ETTm1 \
            --features M \
            --seq_len 96 \
            --label_len 48 \
            --pred_len ${pl} \
            --enc_in 7 \
            --dec_in 7 \
            --c_out 7 \
            --factor 3 \
            --des ${des} \
            --learning_rate ${lr} \
            --lradj ${lradj} \
            --train_epochs ${train_epochs} \
            --patience ${patience} \
            --batch_size ${batch_size} \
            --test_batch_size ${test_batch_size} \
            --itr 1 \
            --rec_lambda ${rl} \
            --auxi_lambda ${ax} \
            --reg_lambda ${reg_lambda} \
            --auxi_batch_size ${auxi_batch_size} \
            --fix_seed ${seed} \
            --checkpoints $CHECKPOINTS \
            --results $RESULTS \
            --test_results $TEST_RESULTS \
            --log_path $LOG_PATH \
            --rerun $rerun \
            --inner_lr $lr_inner \
            --meta_lr $lr_meta \
            --meta_inner_steps $meta_inner_steps \
            --overlap_ratio $overlap_ratio \
            --num_tasks $num_tasks \
            --max_norm $max_norm \
            --auxi_loss ${auxi_loss} \
            --model_type $model_type \
            --cycle $cycle \
            --use_revin $use_revin \
            --dropout $dropout \
            --first_order $first_order \
            --warmup_steps $meta_steps

        sleep 5
    } 2>&1 | tee -a "${OUTPUT_DIR}/stdout.log" &
done








# hyper-parameters
dst=ETTm2

train_epochs=30
patience=5
test_batch_size=1
use_revin=1
model_type=linear
dropout=0.5
cycle=96
first_order=1
auxi_loss=MSE
overlap_ratio=0.0
reg_lambda=0.0
lambda=1.0
auxi_batch_size=64
lradj=type1
max_norm=5.0
rerun=0


pl_list=(96 192 336 720)
# NOTE: ETTm2 settings



for pl in ${pl_list[@]}; do
    if ! [[ " ${datasets[@]} " =~ " ${dst} " ]]; then
        continue
    fi

    case $pl in
        96) lr=0.0002 lr_inner=0.0002 lr_meta=0.2 meta_steps=500 num_tasks=1 meta_inner_steps=5 batch_size=32;;
        192) lr=0.0002 lr_inner=0.0002 lr_meta=0.2 meta_steps=500 num_tasks=1 meta_inner_steps=2 batch_size=32;;
        336) lr=0.0002 lr_inner=0.0002 lr_meta=0.1 meta_steps=500 num_tasks=3 meta_inner_steps=4 batch_size=32;;
        720) lr=0.0002 lr_inner=0.0002 lr_meta=0.2 meta_steps=500 num_tasks=5 meta_inner_steps=5 batch_size=32;;
    esac

    rl=$lambda
    ax=$(echo "1 - $lambda" | bc)
    decimal_places=$(echo "$lambda" | awk -F. '{print length($2)}')
    ax=$(printf "%.${decimal_places}f" $ax)

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lr_inner}_${lr_meta}_${lradj}_${train_epochs}_${patience}_${batch_size}_${auxi_batch_size}_${reg_lambda}_${max_norm}_${num_tasks}_${overlap_ratio}_${meta_inner_steps}_${auxi_loss}_${cycle}_${dropout}_${model_type}_${use_revin}_${first_order}_${meta_steps}
    OUTPUT_DIR="./results/${EXP_NAME}/${JOB_NAME}"

    CHECKPOINTS=$OUTPUT_DIR/checkpoints/
    RESULTS=$OUTPUT_DIR/results/
    TEST_RESULTS=$OUTPUT_DIR/test_results/
    LOG_PATH=$OUTPUT_DIR/result_long_term_forecast.txt

    mkdir -p "${OUTPUT_DIR}/"
    # if rerun, remove the previous stdout
    if [ $rerun -eq 1 ]; then
        rm -rf "${OUTPUT_DIR}/stdout.log"
    else
        subdirs=("$RESULTS"/*)
        if [ ${#subdirs[@]} -eq 1 ] && [ -f "${subdirs[0]}/metrics.npy" ]; then
            echo ">>>>>>> Job: $JOB_NAME already run, skip <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
            continue
        fi
    fi


    check_jobs
    # Get GPU allocation for this job
    gpu_allocation=$(get_gpu_allocation $job_number)
    # Increment job number for the next iteration
    ((job_number++))

    echo "Running command for $JOB_NAME"
    {
        # Set CUDA_VISIBLE_DEVICES for this script and run it in the background
        CUDA_VISIBLE_DEVICES=$gpu_allocation python -u run.py \
            --task_name long_term_forecast_meta_ml3 \
            --is_training 1 \
            --root_path $DATA_ROOT/ETT-small/ \
            --data_path ETTm2.csv \
            --model_id "${dst}_96_${pl}" \
            --model ${model_name} \
            --data_id $dst \
            --data ETTm2 \
            --features M \
            --seq_len 96 \
            --label_len 48 \
            --pred_len ${pl} \
            --enc_in 7 \
            --dec_in 7 \
            --c_out 7 \
            --factor 3 \
            --des ${des} \
            --learning_rate ${lr} \
            --lradj ${lradj} \
            --train_epochs ${train_epochs} \
            --patience ${patience} \
            --batch_size ${batch_size} \
            --test_batch_size ${test_batch_size} \
            --itr 1 \
            --rec_lambda ${rl} \
            --auxi_lambda ${ax} \
            --reg_lambda ${reg_lambda} \
            --auxi_batch_size ${auxi_batch_size} \
            --fix_seed ${seed} \
            --checkpoints $CHECKPOINTS \
            --results $RESULTS \
            --test_results $TEST_RESULTS \
            --log_path $LOG_PATH \
            --rerun $rerun \
            --inner_lr $lr_inner \
            --meta_lr $lr_meta \
            --meta_inner_steps $meta_inner_steps \
            --overlap_ratio $overlap_ratio \
            --num_tasks $num_tasks \
            --max_norm $max_norm \
            --auxi_loss ${auxi_loss} \
            --model_type $model_type \
            --cycle $cycle \
            --use_revin $use_revin \
            --dropout $dropout \
            --first_order $first_order \
            --warmup_steps $meta_steps

        sleep 5
    } 2>&1 | tee -a "${OUTPUT_DIR}/stdout.log" &
done






# hyper-parameters
dst=ECL

train_epochs=30
patience=5
test_batch_size=1
use_revin=1
model_type=linear
dropout=0.0
cycle=168
first_order=1
auxi_loss=MSE
overlap_ratio=0.0
reg_lambda=0.0
lambda=1.0
auxi_batch_size=64
lradj=type1
max_norm=5.0
rerun=0


pl_list=(96 192 336 720)
# NOTE: ECL settings




for pl in ${pl_list[@]}; do
    if ! [[ " ${datasets[@]} " =~ " ${dst} " ]]; then
        continue
    fi

    case $pl in
        96) lr=0.005 lr_inner=0.005 lr_meta=0.1 meta_steps=300 num_tasks=3 meta_inner_steps=2 batch_size=16;;
        192) lr=0.005 lr_inner=0.005 lr_meta=0.1 meta_steps=300 num_tasks=3 meta_inner_steps=1 batch_size=16;;
        336) lr=0.005 lr_inner=0.005 lr_meta=0.02 meta_steps=300 num_tasks=3 meta_inner_steps=2 batch_size=16;;
        720) lr=0.01 lr_inner=0.01 lr_meta=0.05 meta_steps=300 num_tasks=1 meta_inner_steps=1 batch_size=16;;
    esac

    rl=$lambda
    ax=$(echo "1 - $lambda" | bc)
    decimal_places=$(echo "$lambda" | awk -F. '{print length($2)}')
    ax=$(printf "%.${decimal_places}f" $ax)

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lr_inner}_${lr_meta}_${lradj}_${train_epochs}_${patience}_${batch_size}_${auxi_batch_size}_${reg_lambda}_${max_norm}_${num_tasks}_${overlap_ratio}_${meta_inner_steps}_${auxi_loss}_${cycle}_${dropout}_${model_type}_${use_revin}_${first_order}_${meta_steps}
    OUTPUT_DIR="./results/${EXP_NAME}/${JOB_NAME}"

    CHECKPOINTS=$OUTPUT_DIR/checkpoints/
    RESULTS=$OUTPUT_DIR/results/
    TEST_RESULTS=$OUTPUT_DIR/test_results/
    LOG_PATH=$OUTPUT_DIR/result_long_term_forecast.txt

    mkdir -p "${OUTPUT_DIR}/"
    # if rerun, remove the previous stdout
    if [ $rerun -eq 1 ]; then
        rm -rf "${OUTPUT_DIR}/stdout.log"
    else
        subdirs=("$RESULTS"/*)
        if [ ${#subdirs[@]} -eq 1 ] && [ -f "${subdirs[0]}/metrics.npy" ]; then
            echo ">>>>>>> Job: $JOB_NAME already run, skip <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
            continue
        fi
    fi


    check_jobs
    # Get GPU allocation for this job
    gpu_allocation=$(get_gpu_allocation $job_number)
    # Increment job number for the next iteration
    ((job_number++))

    echo "Running command for $JOB_NAME"
    {
        # Set CUDA_VISIBLE_DEVICES for this script and run it in the background
        CUDA_VISIBLE_DEVICES=$gpu_allocation python -u run.py \
            --task_name long_term_forecast_meta_ml3 \
            --is_training 1 \
            --root_path $DATA_ROOT/electricity/ \
            --data_path electricity.csv \
            --model_id "${dst}_96_${pl}" \
            --model ${model_name} \
            --data_id $dst \
            --data custom \
            --features M \
            --seq_len 96 \
            --label_len 48 \
            --pred_len ${pl} \
            --enc_in 321 \
            --dec_in 321 \
            --c_out 321 \
            --factor 3 \
            --des ${des} \
            --learning_rate ${lr} \
            --lradj ${lradj} \
            --train_epochs ${train_epochs} \
            --patience ${patience} \
            --batch_size ${batch_size} \
            --test_batch_size ${test_batch_size} \
            --itr 1 \
            --rec_lambda ${rl} \
            --auxi_lambda ${ax} \
            --reg_lambda ${reg_lambda} \
            --auxi_batch_size ${auxi_batch_size} \
            --fix_seed ${seed} \
            --checkpoints $CHECKPOINTS \
            --results $RESULTS \
            --test_results $TEST_RESULTS \
            --log_path $LOG_PATH \
            --rerun $rerun \
            --inner_lr $lr_inner \
            --meta_lr $lr_meta \
            --meta_inner_steps $meta_inner_steps \
            --overlap_ratio $overlap_ratio \
            --num_tasks $num_tasks \
            --max_norm $max_norm \
            --auxi_loss ${auxi_loss} \
            --model_type $model_type \
            --cycle $cycle \
            --use_revin $use_revin \
            --dropout $dropout \
            --first_order $first_order \
            --warmup_steps $meta_steps

        sleep 5
    } 2>&1 | tee -a "${OUTPUT_DIR}/stdout.log" &
done





# hyper-parameters
dst=Weather

train_epochs=30
patience=5
test_batch_size=1
use_revin=1
model_type=linear
dropout=0.5
cycle=144
first_order=1
auxi_loss=MSE
overlap_ratio=0.0
reg_lambda=0.0
lambda=1.0
auxi_batch_size=64
lradj=type1
max_norm=5.0
rerun=0


pl_list=(96 192 336 720)
# NOTE: Weather settings




for pl in ${pl_list[@]}; do
    if ! [[ " ${datasets[@]} " =~ " ${dst} " ]]; then
        continue
    fi

    case $pl in
        96) lr=0.002 lr_inner=0.002 lr_meta=0.1 meta_steps=700 num_tasks=5 meta_inner_steps=4 batch_size=32;;
        192) lr=0.002 lr_inner=0.002 lr_meta=0.02 meta_steps=700 num_tasks=3 meta_inner_steps=1 batch_size=32;;
        336) lr=0.002 lr_inner=0.002 lr_meta=0.05 meta_steps=700 num_tasks=4 meta_inner_steps=5 batch_size=32;;
        720) lr=0.002 lr_inner=0.002 lr_meta=0.2 meta_steps=700 num_tasks=3 meta_inner_steps=4 batch_size=32;;
    esac

    rl=$lambda
    ax=$(echo "1 - $lambda" | bc)
    decimal_places=$(echo "$lambda" | awk -F. '{print length($2)}')
    ax=$(printf "%.${decimal_places}f" $ax)

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lr_inner}_${lr_meta}_${lradj}_${train_epochs}_${patience}_${batch_size}_${auxi_batch_size}_${reg_lambda}_${max_norm}_${num_tasks}_${overlap_ratio}_${meta_inner_steps}_${auxi_loss}_${cycle}_${dropout}_${model_type}_${use_revin}_${first_order}_${meta_steps}
    OUTPUT_DIR="./results/${EXP_NAME}/${JOB_NAME}"

    CHECKPOINTS=$OUTPUT_DIR/checkpoints/
    RESULTS=$OUTPUT_DIR/results/
    TEST_RESULTS=$OUTPUT_DIR/test_results/
    LOG_PATH=$OUTPUT_DIR/result_long_term_forecast.txt

    mkdir -p "${OUTPUT_DIR}/"
    # if rerun, remove the previous stdout
    if [ $rerun -eq 1 ]; then
        rm -rf "${OUTPUT_DIR}/stdout.log"
    else
        subdirs=("$RESULTS"/*)
        if [ ${#subdirs[@]} -eq 1 ] && [ -f "${subdirs[0]}/metrics.npy" ]; then
            echo ">>>>>>> Job: $JOB_NAME already run, skip <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
            continue
        fi
    fi


    check_jobs
    # Get GPU allocation for this job
    gpu_allocation=$(get_gpu_allocation $job_number)
    # Increment job number for the next iteration
    ((job_number++))

    echo "Running command for $JOB_NAME"
    {
        # Set CUDA_VISIBLE_DEVICES for this script and run it in the background
        CUDA_VISIBLE_DEVICES=$gpu_allocation python -u run.py \
            --task_name long_term_forecast_meta_ml3 \
            --is_training 1 \
            --root_path $DATA_ROOT/weather/ \
            --data_path weather.csv \
            --model_id "${dst}_96_${pl}" \
            --model ${model_name} \
            --data_id $dst \
            --data custom \
            --features M \
            --seq_len 96 \
            --label_len 48 \
            --pred_len ${pl} \
            --enc_in 21 \
            --dec_in 21 \
            --c_out 21 \
            --factor 3 \
            --des ${des} \
            --learning_rate ${lr} \
            --lradj ${lradj} \
            --train_epochs ${train_epochs} \
            --patience ${patience} \
            --batch_size ${batch_size} \
            --test_batch_size ${test_batch_size} \
            --itr 1 \
            --rec_lambda ${rl} \
            --auxi_lambda ${ax} \
            --reg_lambda ${reg_lambda} \
            --auxi_batch_size ${auxi_batch_size} \
            --fix_seed ${seed} \
            --checkpoints $CHECKPOINTS \
            --results $RESULTS \
            --test_results $TEST_RESULTS \
            --log_path $LOG_PATH \
            --rerun $rerun \
            --inner_lr $lr_inner \
            --meta_lr $lr_meta \
            --meta_inner_steps $meta_inner_steps \
            --overlap_ratio $overlap_ratio \
            --num_tasks $num_tasks \
            --max_norm $max_norm \
            --auxi_loss ${auxi_loss} \
            --model_type $model_type \
            --cycle $cycle \
            --use_revin $use_revin \
            --dropout $dropout \
            --first_order $first_order \
            --warmup_steps $meta_steps

        sleep 5
    } 2>&1 | tee -a "${OUTPUT_DIR}/stdout.log" &
done





# hyper-parameters
dst=PEMS03

train_epochs=30
patience=5
test_batch_size=1
use_revin=0
model_type=linear
dropout=0.0
cycle=288
first_order=1
auxi_loss=MSE
overlap_ratio=0.0
reg_lambda=0.0
lambda=1.0
auxi_batch_size=64
lradj=type1
max_norm=5.0
rerun=0

pl_list=(12 24 36 48)
# NOTE: PEMS03 settings



for pl in ${pl_list[@]}; do
    if ! [[ " ${datasets[@]} " =~ " ${dst} " ]]; then
        continue
    fi

    case $pl in
        12) lr=0.002 lr_inner=0.002 lr_meta=0.015 meta_steps=200 num_tasks=3 meta_inner_steps=2 batch_size=32;;
        24) lr=0.002 lr_inner=0.002 lr_meta=0.015 meta_steps=200 num_tasks=3 meta_inner_steps=2 batch_size=32;;
        36) lr=0.002 lr_inner=0.002 lr_meta=0.05 meta_steps=200 num_tasks=3 meta_inner_steps=1 batch_size=32;;
        48) lr=0.002 lr_inner=0.002 lr_meta=0.05 meta_steps=200 num_tasks=3 meta_inner_steps=1 batch_size=32;;
    esac

    rl=$lambda
    ax=$(echo "1 - $lambda" | bc)
    decimal_places=$(echo "$lambda" | awk -F. '{print length($2)}')
    ax=$(printf "%.${decimal_places}f" $ax)

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lr_inner}_${lr_meta}_${lradj}_${train_epochs}_${patience}_${batch_size}_${auxi_batch_size}_${reg_lambda}_${max_norm}_${num_tasks}_${overlap_ratio}_${meta_inner_steps}_${auxi_loss}_${cycle}_${dropout}_${model_type}_${use_revin}_${first_order}_${meta_steps}
    OUTPUT_DIR="./results/${EXP_NAME}/${JOB_NAME}"

    CHECKPOINTS=$OUTPUT_DIR/checkpoints/
    RESULTS=$OUTPUT_DIR/results/
    TEST_RESULTS=$OUTPUT_DIR/test_results/
    LOG_PATH=$OUTPUT_DIR/result_long_term_forecast.txt

    mkdir -p "${OUTPUT_DIR}/"
    # if rerun, remove the previous stdout
    if [ $rerun -eq 1 ]; then
        rm -rf "${OUTPUT_DIR}/stdout.log"
    else
        subdirs=("$RESULTS"/*)
        if [ ${#subdirs[@]} -eq 1 ] && [ -f "${subdirs[0]}/metrics.npy" ]; then
            echo ">>>>>>> Job: $JOB_NAME already run, skip <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
            continue
        fi
    fi


    check_jobs
    # Get GPU allocation for this job
    gpu_allocation=$(get_gpu_allocation $job_number)
    # Increment job number for the next iteration
    ((job_number++))

    echo "Running command for $JOB_NAME"
    {
        # Set CUDA_VISIBLE_DEVICES for this script and run it in the background
        CUDA_VISIBLE_DEVICES=$gpu_allocation python -u run.py \
            --task_name long_term_forecast_meta_ml3 \
            --is_training 1 \
            --root_path $DATA_ROOT/PEMS/ \
            --data_path PEMS03.npz \
            --model_id "${dst}_96_${pl}" \
            --model ${model_name} \
            --data_id $dst \
            --data PEMS \
            --features M \
            --seq_len 96 \
            --label_len 48 \
            --pred_len ${pl} \
            --enc_in 358 \
            --dec_in 358 \
            --c_out 358 \
            --factor 3 \
            --des ${des} \
            --learning_rate ${lr} \
            --lradj ${lradj} \
            --train_epochs ${train_epochs} \
            --patience ${patience} \
            --batch_size ${batch_size} \
            --test_batch_size ${test_batch_size} \
            --itr 1 \
            --rec_lambda ${rl} \
            --auxi_lambda ${ax} \
            --reg_lambda ${reg_lambda} \
            --auxi_batch_size ${auxi_batch_size} \
            --fix_seed ${seed} \
            --checkpoints $CHECKPOINTS \
            --results $RESULTS \
            --test_results $TEST_RESULTS \
            --log_path $LOG_PATH \
            --rerun $rerun \
            --inner_lr $lr_inner \
            --meta_lr $lr_meta \
            --meta_inner_steps $meta_inner_steps \
            --overlap_ratio $overlap_ratio \
            --num_tasks $num_tasks \
            --max_norm $max_norm \
            --auxi_loss ${auxi_loss} \
            --model_type $model_type \
            --cycle $cycle \
            --use_revin $use_revin \
            --dropout $dropout \
            --first_order $first_order \
            --warmup_steps $meta_steps

        sleep 5
    } 2>&1 | tee -a "${OUTPUT_DIR}/stdout.log" &
done







# hyper-parameters
dst=PEMS08

train_epochs=30
patience=5
test_batch_size=1
use_revin=1
model_type=linear
dropout=0.0
cycle=288
first_order=1
auxi_loss=MSE
overlap_ratio=0.0
reg_lambda=0.0
lambda=1.0
auxi_batch_size=64
lradj=type1
max_norm=5.0
rerun=0

pl_list=(12 24 36 48)
# NOTE: PEMS08 settings





for pl in ${pl_list[@]}; do
    if ! [[ " ${datasets[@]} " =~ " ${dst} " ]]; then
        continue
    fi

    case $pl in
        12) lr=0.003 lr_inner=0.003 lr_meta=0.01 meta_steps=700 num_tasks=3 meta_inner_steps=2 batch_size=32;;
        24) lr=0.003 lr_inner=0.003 lr_meta=0.02 meta_steps=700 num_tasks=3 meta_inner_steps=3 batch_size=32;;
        36) lr=0.003 lr_inner=0.003 lr_meta=0.02 meta_steps=700 num_tasks=3 meta_inner_steps=2 batch_size=32;;
        48) lr=0.003 lr_inner=0.003 lr_meta=0.03 meta_steps=300 num_tasks=3 meta_inner_steps=1 batch_size=32;;
    esac

    rl=$lambda
    ax=$(echo "1 - $lambda" | bc)
    decimal_places=$(echo "$lambda" | awk -F. '{print length($2)}')
    ax=$(printf "%.${decimal_places}f" $ax)

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lr_inner}_${lr_meta}_${lradj}_${train_epochs}_${patience}_${batch_size}_${auxi_batch_size}_${reg_lambda}_${max_norm}_${num_tasks}_${overlap_ratio}_${meta_inner_steps}_${auxi_loss}_${cycle}_${dropout}_${model_type}_${use_revin}_${first_order}_${meta_steps}
    OUTPUT_DIR="./results/${EXP_NAME}/${JOB_NAME}"

    CHECKPOINTS=$OUTPUT_DIR/checkpoints/
    RESULTS=$OUTPUT_DIR/results/
    TEST_RESULTS=$OUTPUT_DIR/test_results/
    LOG_PATH=$OUTPUT_DIR/result_long_term_forecast.txt

    mkdir -p "${OUTPUT_DIR}/"
    # if rerun, remove the previous stdout
    if [ $rerun -eq 1 ]; then
        rm -rf "${OUTPUT_DIR}/stdout.log"
    else
        subdirs=("$RESULTS"/*)
        if [ ${#subdirs[@]} -eq 1 ] && [ -f "${subdirs[0]}/metrics.npy" ]; then
            echo ">>>>>>> Job: $JOB_NAME already run, skip <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
            continue
        fi
    fi


    check_jobs
    # Get GPU allocation for this job
    gpu_allocation=$(get_gpu_allocation $job_number)
    # Increment job number for the next iteration
    ((job_number++))

    echo "Running command for $JOB_NAME"
    {
        # Set CUDA_VISIBLE_DEVICES for this script and run it in the background
        CUDA_VISIBLE_DEVICES=$gpu_allocation python -u run.py \
            --task_name long_term_forecast_meta_ml3 \
            --is_training 1 \
            --root_path $DATA_ROOT/PEMS/ \
            --data_path PEMS08.npz \
            --model_id "${dst}_96_${pl}" \
            --model ${model_name} \
            --data_id $dst \
            --data PEMS \
            --features M \
            --seq_len 96 \
            --label_len 48 \
            --pred_len ${pl} \
            --enc_in 170 \
            --dec_in 170 \
            --c_out 170 \
            --factor 3 \
            --des ${des} \
            --learning_rate ${lr} \
            --lradj ${lradj} \
            --train_epochs ${train_epochs} \
            --patience ${patience} \
            --batch_size ${batch_size} \
            --test_batch_size ${test_batch_size} \
            --itr 1 \
            --rec_lambda ${rl} \
            --auxi_lambda ${ax} \
            --reg_lambda ${reg_lambda} \
            --auxi_batch_size ${auxi_batch_size} \
            --fix_seed ${seed} \
            --checkpoints $CHECKPOINTS \
            --results $RESULTS \
            --test_results $TEST_RESULTS \
            --log_path $LOG_PATH \
            --rerun $rerun \
            --inner_lr $lr_inner \
            --meta_lr $lr_meta \
            --meta_inner_steps $meta_inner_steps \
            --overlap_ratio $overlap_ratio \
            --num_tasks $num_tasks \
            --max_norm $max_norm \
            --auxi_loss ${auxi_loss} \
            --model_type $model_type \
            --cycle $cycle \
            --use_revin $use_revin \
            --dropout $dropout \
            --first_order $first_order \
            --warmup_steps $meta_steps

        sleep 5
    } 2>&1 | tee -a "${OUTPUT_DIR}/stdout.log" &
done




wait