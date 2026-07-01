#!/bin/bash
MAX_JOBS=32
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

DATA_ROOT=$USRDIR/dataset
OUT_ROOT=/mnt/tidalfs-bdsz01/dataset/llm_ckpt/plc_data/FreDF
EXP_NAME=long_term_ext
seed=2023
des='iTransformer'

model_name=iTransformer

auxi_mode=rfft
auxi_type=complex
module_first=1
first_order=1
overlap_ratio=0.0
test_batch_size=1

datasets=(ECL)
# datasets=(ETTh1 ETTh2 ETTm1 ETTm2)



# hyper-parameters
dst=ETTh1

# pl_list=(96 192 336 720)
# lbd_list=(0.2 0.1 0.0)
# lr_list=(0.001 0.003 0.002)
# lradj_list=(type1)
# bs_list=(32)
# meta_lr_list=(0.01 0.05 0.02 0.1)
# meta_inner_steps_list=(3 4 5)
# inner_lr_list=(same)
# num_tasks_list=(3 4 5)
# auxi_bs_list=(64)
# max_norm_list=(5.0)
# meta_step_list=(300 500)
# auxi_loss_list=(MAE)


pl_list=(96 192 336 720)
lbd_list=(0.0)
lr_list=(0.005 0.002)
lradj_list=(type1 type3)
bs_list=(32)
meta_lr_list=(0.01 0.05 0.1)
meta_inner_steps_list=(4 5)
inner_lr_list=(same)
num_tasks_list=(3 5)
auxi_bs_list=(64)
max_norm_list=(5.0)
meta_step_list=(300 500)
auxi_loss_list=(MAE)

train_epochs=30
patience=5
batch_size=32

rerun=0

for lr in ${lr_list[@]}; do
for inner_lr in ${inner_lr_list[@]}; do
if [[ $inner_lr == "double" ]]; then
    lr_inner=$(echo "scale=10; $lr * 2" | bc)
elif [[ $inner_lr == "half" ]]; then
    lr_inner=$(echo "scale=10; $lr / 2" | bc)
elif [[ $inner_lr == "same" ]]; then
    lr_inner=$lr
else
    lr_inner=$inner_lr
fi
[[ "$lr_inner" == .* ]] && lr_inner="0$lr_inner"
lr_inner=$(echo "$lr_inner" | sed 's/0*$//; s/\.$//')

for meta_lr in ${meta_lr_list[@]}; do
if [[ $meta_lr == "double" ]]; then
    lr_meta=$(echo "scale=10; $lr * 2" | bc)
elif [[ $meta_lr == "half" ]]; then
    lr_meta=$(echo "scale=10; $lr / 2" | bc)
elif [[ $meta_lr == "same" ]]; then
    lr_meta=$lr
else
    lr_meta=$meta_lr
fi
[[ "$lr_meta" == .* ]] && lr_meta="0$lr_meta"
lr_meta=$(echo "$lr_meta" | sed 's/0*$//; s/\.$//')

for meta_steps in ${meta_step_list[@]}; do
for max_norm in ${max_norm_list[@]}; do
for num_tasks in ${num_tasks_list[@]}; do
for meta_inner_steps in ${meta_inner_steps_list[@]}; do
for batch_size in ${bs_list[@]}; do
for auxi_batch_size in ${auxi_bs_list[@]}; do
for lradj in ${lradj_list[@]}; do
for lambda in ${lbd_list[@]}; do
for auxi_loss in ${auxi_loss_list[@]}; do
for pl in ${pl_list[@]}; do
    if ! [[ " ${datasets[@]} " =~ " ${dst} " ]]; then
        continue
    fi

    rl=$lambda
    ax=$(echo "1 - $lambda" | bc)
    decimal_places=$(echo "$lambda" | awk -F. '{print length($2)}')
    ax=$(printf "%.${decimal_places}f" $ax)

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lradj}_${train_epochs}_${patience}_${batch_size}_${auxi_mode}_${auxi_type}_${auxi_loss}_${module_first}_${first_order}_${overlap_ratio}_${auxi_batch_size}_${max_norm}_${num_tasks}_${meta_inner_steps}_${meta_steps}_${lr_inner}_${lr_meta}
    OUTPUT_DIR="${OUT_ROOT}/results/${EXP_NAME}/${JOB_NAME}"

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
        if [ ${#subdirs[@]} -eq 1 ] && [ -f "${subdirs[0]}/metrics.yaml" ]; then
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
            --e_layers 2 \
            --d_layers 1 \
            --factor 3 \
            --d_model 128 \
            --d_ff 128 \
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
            --fix_seed ${seed} \
            --checkpoints $CHECKPOINTS \
            --results $RESULTS \
            --test_results $TEST_RESULTS \
            --log_path $LOG_PATH \
            --rerun $rerun \
            --auxi_mode $auxi_mode \
            --auxi_type $auxi_type \
            --auxi_loss $auxi_loss \
            --module_first $module_first \
            --inner_lr $lr_inner \
            --meta_lr $lr_meta \
            --meta_inner_steps $meta_inner_steps \
            --overlap_ratio $overlap_ratio \
            --num_tasks $num_tasks \
            --max_norm $max_norm \
            --first_order $first_order \
            --warmup_steps $meta_steps \
            --auxi_batch_size ${auxi_batch_size}

        sleep 5
    # } 2>&1 | tee -a "${OUTPUT_DIR}/stdout.log" &
    } &
done
done
done
done
done
done
done
done
done
done
done
done
done





# hyper-parameters
dst=ETTh2

# pl_list=(96 192 336 720)
# lbd_list=(0.3 0.2 0.1 0.8)
# lr_list=(0.001 0.0005 0.002)
# lradj_list=(type1 type3)
# bs_list=(32)
# meta_lr_list=(0.01)
# meta_inner_steps_list=(1 3)
# inner_lr_list=(same)
# num_tasks_list=(4)
# auxi_bs_list=(64)
# max_norm_list=(5.0)
# meta_step_list=(300 500)
# auxi_loss_list=(MAE)


pl_list=(96 192 336 720)
lbd_list=(0.0 0.1)
lr_list=(0.005 0.002)
lradj_list=(type1 type3)
bs_list=(32)
meta_lr_list=(0.01 0.1)
meta_inner_steps_list=(1 3)
inner_lr_list=(same)
num_tasks_list=(4 5)
auxi_bs_list=(64)
max_norm_list=(5.0)
meta_step_list=(300 500)
auxi_loss_list=(MAE)


train_epochs=30
patience=5
batch_size=32

rerun=0

for lr in ${lr_list[@]}; do
for inner_lr in ${inner_lr_list[@]}; do
if [[ $inner_lr == "double" ]]; then
    lr_inner=$(echo "scale=10; $lr * 2" | bc)
elif [[ $inner_lr == "half" ]]; then
    lr_inner=$(echo "scale=10; $lr / 2" | bc)
elif [[ $inner_lr == "same" ]]; then
    lr_inner=$lr
else
    lr_inner=$inner_lr
fi
[[ "$lr_inner" == .* ]] && lr_inner="0$lr_inner"
lr_inner=$(echo "$lr_inner" | sed 's/0*$//; s/\.$//')

for meta_lr in ${meta_lr_list[@]}; do
if [[ $meta_lr == "double" ]]; then
    lr_meta=$(echo "scale=10; $lr * 2" | bc)
elif [[ $meta_lr == "half" ]]; then
    lr_meta=$(echo "scale=10; $lr / 2" | bc)
elif [[ $meta_lr == "same" ]]; then
    lr_meta=$lr
else
    lr_meta=$meta_lr
fi
[[ "$lr_meta" == .* ]] && lr_meta="0$lr_meta"
lr_meta=$(echo "$lr_meta" | sed 's/0*$//; s/\.$//')

for meta_steps in ${meta_step_list[@]}; do
for max_norm in ${max_norm_list[@]}; do
for num_tasks in ${num_tasks_list[@]}; do
for meta_inner_steps in ${meta_inner_steps_list[@]}; do
for batch_size in ${bs_list[@]}; do
for auxi_batch_size in ${auxi_bs_list[@]}; do
for lradj in ${lradj_list[@]}; do
for lambda in ${lbd_list[@]}; do
for auxi_loss in ${auxi_loss_list[@]}; do
for pl in ${pl_list[@]}; do
    if ! [[ " ${datasets[@]} " =~ " ${dst} " ]]; then
        continue
    fi

    rl=$lambda
    ax=$(echo "1 - $lambda" | bc)
    decimal_places=$(echo "$lambda" | awk -F. '{print length($2)}')
    ax=$(printf "%.${decimal_places}f" $ax)

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lradj}_${train_epochs}_${patience}_${batch_size}_${auxi_mode}_${auxi_type}_${auxi_loss}_${module_first}_${first_order}_${overlap_ratio}_${auxi_batch_size}_${max_norm}_${num_tasks}_${meta_inner_steps}_${meta_steps}_${lr_inner}_${lr_meta}
    OUTPUT_DIR="${OUT_ROOT}/results/${EXP_NAME}/${JOB_NAME}"

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
        if [ ${#subdirs[@]} -eq 1 ] && [ -f "${subdirs[0]}/metrics.yaml" ]; then
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
            --e_layers 2 \
            --d_layers 1 \
            --factor 3 \
            --d_model 128 \
            --d_ff 128 \
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
            --fix_seed ${seed} \
            --checkpoints $CHECKPOINTS \
            --results $RESULTS \
            --test_results $TEST_RESULTS \
            --log_path $LOG_PATH \
            --rerun $rerun \
            --auxi_mode $auxi_mode \
            --auxi_type $auxi_type \
            --auxi_loss $auxi_loss \
            --module_first $module_first \
            --inner_lr $lr_inner \
            --meta_lr $lr_meta \
            --meta_inner_steps $meta_inner_steps \
            --overlap_ratio $overlap_ratio \
            --num_tasks $num_tasks \
            --max_norm $max_norm \
            --first_order $first_order \
            --warmup_steps $meta_steps \
            --auxi_batch_size ${auxi_batch_size}

        sleep 5
    # } 2>&1 | tee -a "${OUTPUT_DIR}/stdout.log" &
    } &
done
done
done
done
done
done
done
done
done
done
done
done
done







# hyper-parameters
dst=ETTm1
# pl_list=(96 192 336 720)
# lbd_list=(0.2 0.5 0.6)
# lr_list=(0.005 0.002)
# lradj_list=(type1)
# bs_list=(32)
# meta_lr_list=(0.1 0.2 0.01 0.05)
# meta_inner_steps_list=(1 3 5)
# inner_lr_list=(same)
# num_tasks_list=(3 4 5)
# auxi_bs_list=(64)
# max_norm_list=(5.0)
# meta_step_list=(500)
# auxi_loss_list=(MAE)

pl_list=(192 336 720)
lbd_list=(0.2 0.6 0.8)
lr_list=(0.005 0.002 0.01 0.003 0.004)
lradj_list=(type1)
bs_list=(32)
meta_lr_list=(0.1 0.01 0.05 0.005)
meta_inner_steps_list=(1 3 4 5)
inner_lr_list=(same)
num_tasks_list=(3 4)
auxi_bs_list=(64)
max_norm_list=(5.0)
meta_step_list=(300 500)
auxi_loss_list=(MAE)


train_epochs=30
patience=5
batch_size=32

rerun=0

for lr in ${lr_list[@]}; do
for inner_lr in ${inner_lr_list[@]}; do
if [[ $inner_lr == "double" ]]; then
    lr_inner=$(echo "scale=10; $lr * 2" | bc)
elif [[ $inner_lr == "half" ]]; then
    lr_inner=$(echo "scale=10; $lr / 2" | bc)
elif [[ $inner_lr == "same" ]]; then
    lr_inner=$lr
else
    lr_inner=$inner_lr
fi
[[ "$lr_inner" == .* ]] && lr_inner="0$lr_inner"
lr_inner=$(echo "$lr_inner" | sed 's/0*$//; s/\.$//')

for meta_lr in ${meta_lr_list[@]}; do
if [[ $meta_lr == "double" ]]; then
    lr_meta=$(echo "scale=10; $lr * 2" | bc)
elif [[ $meta_lr == "half" ]]; then
    lr_meta=$(echo "scale=10; $lr / 2" | bc)
elif [[ $meta_lr == "same" ]]; then
    lr_meta=$lr
else
    lr_meta=$meta_lr
fi
[[ "$lr_meta" == .* ]] && lr_meta="0$lr_meta"
lr_meta=$(echo "$lr_meta" | sed 's/0*$//; s/\.$//')

for meta_steps in ${meta_step_list[@]}; do
for max_norm in ${max_norm_list[@]}; do
for num_tasks in ${num_tasks_list[@]}; do
for meta_inner_steps in ${meta_inner_steps_list[@]}; do
for batch_size in ${bs_list[@]}; do
for auxi_batch_size in ${auxi_bs_list[@]}; do
for lradj in ${lradj_list[@]}; do
for lambda in ${lbd_list[@]}; do
for auxi_loss in ${auxi_loss_list[@]}; do
for pl in ${pl_list[@]}; do
    if ! [[ " ${datasets[@]} " =~ " ${dst} " ]]; then
        continue
    fi

    rl=$lambda
    ax=$(echo "1 - $lambda" | bc)
    decimal_places=$(echo "$lambda" | awk -F. '{print length($2)}')
    ax=$(printf "%.${decimal_places}f" $ax)

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lradj}_${train_epochs}_${patience}_${batch_size}_${auxi_mode}_${auxi_type}_${auxi_loss}_${module_first}_${first_order}_${overlap_ratio}_${auxi_batch_size}_${max_norm}_${num_tasks}_${meta_inner_steps}_${meta_steps}_${lr_inner}_${lr_meta}
    OUTPUT_DIR="${OUT_ROOT}/results/${EXP_NAME}/${JOB_NAME}"

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
        if [ ${#subdirs[@]} -eq 1 ] && [ -f "${subdirs[0]}/metrics.yaml" ]; then
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
            --e_layers 2 \
            --d_layers 1 \
            --factor 3 \
            --d_model 128 \
            --d_ff 128 \
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
            --fix_seed ${seed} \
            --checkpoints $CHECKPOINTS \
            --results $RESULTS \
            --test_results $TEST_RESULTS \
            --log_path $LOG_PATH \
            --rerun $rerun \
            --auxi_mode $auxi_mode \
            --auxi_type $auxi_type \
            --auxi_loss $auxi_loss \
            --module_first $module_first \
            --inner_lr $lr_inner \
            --meta_lr $lr_meta \
            --meta_inner_steps $meta_inner_steps \
            --overlap_ratio $overlap_ratio \
            --num_tasks $num_tasks \
            --max_norm $max_norm \
            --first_order $first_order \
            --warmup_steps $meta_steps \
            --auxi_batch_size ${auxi_batch_size}

        sleep 5
    # } 2>&1 | tee -a "${OUTPUT_DIR}/stdout.log" &
    } &
done
done
done
done
done
done
done
done
done
done
done
done
done







# hyper-parameters
dst=ETTm2
# pl_list=(96 192 336 720)
# lbd_list=(0.1 0.2 0.3 0.4)
# lr_list=(0.002 0.001 0.005)
# lradj_list=(type1)
# bs_list=(32)
# meta_lr_list=(0.1 0.15)
# meta_inner_steps_list=(4 5 6)
# inner_lr_list=(same)
# num_tasks_list=(2 3 4)
# auxi_bs_list=(64)
# max_norm_list=(5.0)
# meta_step_list=(300 500)
# auxi_loss_list=(MAE)

# pl_list=(96 336 720)
# lbd_list=(0.1 0.2 0.3 0.4 0.6 0.8)
# lr_list=(0.002 0.001 0.0005 0.0002)
# lradj_list=(type1)
# bs_list=(32)
# meta_lr_list=(0.1 0.05 0.02 0.01)
# meta_inner_steps_list=(3 4 5)
# inner_lr_list=(same)
# num_tasks_list=(3 4 5)
# auxi_bs_list=(64)
# max_norm_list=(5.0)
# meta_step_list=(700 500)
# auxi_loss_list=(MAE)

pl_list=(96 336 720)
lbd_list=(0.1 0.2 0.3 0.05 0.15 0.25)
lr_list=(0.002 0.001 0.0015 0.0012 0.0008)
lradj_list=(type1 type3)
bs_list=(32)
meta_lr_list=(0.1 0.12 0.08)
meta_inner_steps_list=(4 5 6)
inner_lr_list=(same)
num_tasks_list=(2 3 4)
auxi_bs_list=(64)
max_norm_list=(5.0)
meta_step_list=(100 300 500)
auxi_loss_list=(MAE)

train_epochs=30
patience=5
batch_size=32

rerun=0

for lr in ${lr_list[@]}; do
for inner_lr in ${inner_lr_list[@]}; do
if [[ $inner_lr == "double" ]]; then
    lr_inner=$(echo "scale=10; $lr * 2" | bc)
elif [[ $inner_lr == "half" ]]; then
    lr_inner=$(echo "scale=10; $lr / 2" | bc)
elif [[ $inner_lr == "same" ]]; then
    lr_inner=$lr
else
    lr_inner=$inner_lr
fi
[[ "$lr_inner" == .* ]] && lr_inner="0$lr_inner"
lr_inner=$(echo "$lr_inner" | sed 's/0*$//; s/\.$//')

for meta_lr in ${meta_lr_list[@]}; do
if [[ $meta_lr == "double" ]]; then
    lr_meta=$(echo "scale=10; $lr * 2" | bc)
elif [[ $meta_lr == "half" ]]; then
    lr_meta=$(echo "scale=10; $lr / 2" | bc)
elif [[ $meta_lr == "same" ]]; then
    lr_meta=$lr
else
    lr_meta=$meta_lr
fi
[[ "$lr_meta" == .* ]] && lr_meta="0$lr_meta"
lr_meta=$(echo "$lr_meta" | sed 's/0*$//; s/\.$//')

for meta_steps in ${meta_step_list[@]}; do
for max_norm in ${max_norm_list[@]}; do
for num_tasks in ${num_tasks_list[@]}; do
for meta_inner_steps in ${meta_inner_steps_list[@]}; do
for batch_size in ${bs_list[@]}; do
for auxi_batch_size in ${auxi_bs_list[@]}; do
for lradj in ${lradj_list[@]}; do
for lambda in ${lbd_list[@]}; do
for auxi_loss in ${auxi_loss_list[@]}; do
for pl in ${pl_list[@]}; do
    if ! [[ " ${datasets[@]} " =~ " ${dst} " ]]; then
        continue
    fi

    rl=$lambda
    ax=$(echo "1 - $lambda" | bc)
    decimal_places=$(echo "$lambda" | awk -F. '{print length($2)}')
    ax=$(printf "%.${decimal_places}f" $ax)

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lradj}_${train_epochs}_${patience}_${batch_size}_${auxi_mode}_${auxi_type}_${auxi_loss}_${module_first}_${first_order}_${overlap_ratio}_${auxi_batch_size}_${max_norm}_${num_tasks}_${meta_inner_steps}_${meta_steps}_${lr_inner}_${lr_meta}
    OUTPUT_DIR="${OUT_ROOT}/results/${EXP_NAME}/${JOB_NAME}"

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
        if [ ${#subdirs[@]} -eq 1 ] && [ -f "${subdirs[0]}/metrics.yaml" ]; then
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
            --e_layers 2 \
            --d_layers 1 \
            --factor 3 \
            --d_model 128 \
            --d_ff 128 \
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
            --fix_seed ${seed} \
            --checkpoints $CHECKPOINTS \
            --results $RESULTS \
            --test_results $TEST_RESULTS \
            --log_path $LOG_PATH \
            --rerun $rerun \
            --auxi_mode $auxi_mode \
            --auxi_type $auxi_type \
            --auxi_loss $auxi_loss \
            --module_first $module_first \
            --inner_lr $lr_inner \
            --meta_lr $lr_meta \
            --meta_inner_steps $meta_inner_steps \
            --overlap_ratio $overlap_ratio \
            --num_tasks $num_tasks \
            --max_norm $max_norm \
            --first_order $first_order \
            --warmup_steps $meta_steps \
            --auxi_batch_size ${auxi_batch_size}

        sleep 5
    # } 2>&1 | tee -a "${OUTPUT_DIR}/stdout.log" &
    } &
done
done
done
done
done
done
done
done
done
done
done
done
done






# hyper-parameters
dst=ECL

# pl_list=(96 192 336 720)
# lbd_list=(0.5 0.6)
# lr_list=(0.001 0.005 0.002)
# lradj_list=(type1)
# bs_list=(16 32)
# meta_lr_list=(0.01)
# meta_inner_steps_list=(1 2)
# inner_lr_list=(same)
# num_tasks_list=(1 2)
# auxi_bs_list=(64)
# max_norm_list=(5.0)
# meta_step_list=(300 100)
# auxi_loss_list=(MAE)


pl_list=(96 192 336 720)
lbd_list=(0.5 0.6 0.8 0.9)
lr_list=(0.001 0.0005 0.002)
lradj_list=(type1 type3)
bs_list=(16)
meta_lr_list=(0.01 0.1)
meta_inner_steps_list=(1)
inner_lr_list=(same)
num_tasks_list=(1 2)
auxi_bs_list=(64)
max_norm_list=(5.0)
meta_step_list=(300 100)
auxi_loss_list=(MAE)

train_epochs=30
patience=5
batch_size=16

rerun=0

for lr in ${lr_list[@]}; do
for inner_lr in ${inner_lr_list[@]}; do
if [[ $inner_lr == "double" ]]; then
    lr_inner=$(echo "scale=10; $lr * 2" | bc)
elif [[ $inner_lr == "half" ]]; then
    lr_inner=$(echo "scale=10; $lr / 2" | bc)
elif [[ $inner_lr == "same" ]]; then
    lr_inner=$lr
else
    lr_inner=$inner_lr
fi
[[ "$lr_inner" == .* ]] && lr_inner="0$lr_inner"
lr_inner=$(echo "$lr_inner" | sed 's/0*$//; s/\.$//')

for meta_lr in ${meta_lr_list[@]}; do
if [[ $meta_lr == "double" ]]; then
    lr_meta=$(echo "scale=10; $lr * 2" | bc)
elif [[ $meta_lr == "half" ]]; then
    lr_meta=$(echo "scale=10; $lr / 2" | bc)
elif [[ $meta_lr == "same" ]]; then
    lr_meta=$lr
else
    lr_meta=$meta_lr
fi
[[ "$lr_meta" == .* ]] && lr_meta="0$lr_meta"
lr_meta=$(echo "$lr_meta" | sed 's/0*$//; s/\.$//')

for meta_steps in ${meta_step_list[@]}; do
for max_norm in ${max_norm_list[@]}; do
for num_tasks in ${num_tasks_list[@]}; do
for meta_inner_steps in ${meta_inner_steps_list[@]}; do
for batch_size in ${bs_list[@]}; do
for auxi_batch_size in ${auxi_bs_list[@]}; do
for lradj in ${lradj_list[@]}; do
for lambda in ${lbd_list[@]}; do
for auxi_loss in ${auxi_loss_list[@]}; do
for pl in ${pl_list[@]}; do
    if ! [[ " ${datasets[@]} " =~ " ${dst} " ]]; then
        continue
    fi

    rl=$lambda
    ax=$(echo "1 - $lambda" | bc)
    decimal_places=$(echo "$lambda" | awk -F. '{print length($2)}')
    ax=$(printf "%.${decimal_places}f" $ax)

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lradj}_${train_epochs}_${patience}_${batch_size}_${auxi_mode}_${auxi_type}_${auxi_loss}_${module_first}_${first_order}_${overlap_ratio}_${auxi_batch_size}_${max_norm}_${num_tasks}_${meta_inner_steps}_${meta_steps}_${lr_inner}_${lr_meta}
    OUTPUT_DIR="${OUT_ROOT}/results/${EXP_NAME}/${JOB_NAME}"

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
        if [ ${#subdirs[@]} -eq 1 ] && [ -f "${subdirs[0]}/metrics.yaml" ]; then
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
            --e_layers 3 \
            --d_layers 1 \
            --factor 3 \
            --d_model 512 \
            --d_ff 512 \
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
            --fix_seed ${seed} \
            --checkpoints $CHECKPOINTS \
            --results $RESULTS \
            --test_results $TEST_RESULTS \
            --log_path $LOG_PATH \
            --rerun $rerun \
            --auxi_mode $auxi_mode \
            --auxi_type $auxi_type \
            --auxi_loss $auxi_loss \
            --module_first $module_first \
            --inner_lr $lr_inner \
            --meta_lr $lr_meta \
            --meta_inner_steps $meta_inner_steps \
            --overlap_ratio $overlap_ratio \
            --num_tasks $num_tasks \
            --max_norm $max_norm \
            --first_order $first_order \
            --warmup_steps $meta_steps \
            --auxi_batch_size ${auxi_batch_size}

        sleep 5
    # } 2>&1 | tee -a "${OUTPUT_DIR}/stdout.log" &
    } &
done
done
done
done
done
done
done
done
done
done
done
done
done





# hyper-parameters
dst=Traffic
# pl_list=(96 192 336 720)
# lbd_list=(0.9 0.6 0.2 0.1)
# lr_list=(0.01 0.005 0.02 0.002 0.001)
# lradj_list=(type1)
# bs_list=(16)
# meta_lr_list=(0.1 0.02 0.05)
# meta_inner_steps_list=(1 2 3)
# inner_lr_list=(same)
# num_tasks_list=(1 2 3)
# auxi_bs_list=(64)
# max_norm_list=(5.0)
# meta_step_list=(300 100)
# auxi_loss_list=(MAE)

pl_list=(96 192 336 720)
lbd_list=(0.9 0.92 0.94 0.96 0.98)
lr_list=(0.0005 0.0008 0.001 0.0002)
lradj_list=(type1 type3)
bs_list=(16)
meta_lr_list=(0.01 0.02 0.05)
meta_inner_steps_list=(1)
inner_lr_list=(same)
num_tasks_list=(1)
auxi_bs_list=(64)
max_norm_list=(5.0)
meta_step_list=(200 100 50)
auxi_loss_list=(MAE)

train_epochs=30
patience=5
batch_size=16

rerun=0

for lr in ${lr_list[@]}; do
for inner_lr in ${inner_lr_list[@]}; do
if [[ $inner_lr == "double" ]]; then
    lr_inner=$(echo "scale=10; $lr * 2" | bc)
elif [[ $inner_lr == "half" ]]; then
    lr_inner=$(echo "scale=10; $lr / 2" | bc)
elif [[ $inner_lr == "same" ]]; then
    lr_inner=$lr
else
    lr_inner=$inner_lr
fi
[[ "$lr_inner" == .* ]] && lr_inner="0$lr_inner"
lr_inner=$(echo "$lr_inner" | sed 's/0*$//; s/\.$//')

for meta_lr in ${meta_lr_list[@]}; do
if [[ $meta_lr == "double" ]]; then
    lr_meta=$(echo "scale=10; $lr * 2" | bc)
elif [[ $meta_lr == "half" ]]; then
    lr_meta=$(echo "scale=10; $lr / 2" | bc)
elif [[ $meta_lr == "same" ]]; then
    lr_meta=$lr
else
    lr_meta=$meta_lr
fi
[[ "$lr_meta" == .* ]] && lr_meta="0$lr_meta"
lr_meta=$(echo "$lr_meta" | sed 's/0*$//; s/\.$//')

for meta_steps in ${meta_step_list[@]}; do
for max_norm in ${max_norm_list[@]}; do
for num_tasks in ${num_tasks_list[@]}; do
for meta_inner_steps in ${meta_inner_steps_list[@]}; do
for batch_size in ${bs_list[@]}; do
for auxi_batch_size in ${auxi_bs_list[@]}; do
for lradj in ${lradj_list[@]}; do
for lambda in ${lbd_list[@]}; do
for auxi_loss in ${auxi_loss_list[@]}; do
for pl in ${pl_list[@]}; do
    if ! [[ " ${datasets[@]} " =~ " ${dst} " ]]; then
        continue
    fi

    rl=$lambda
    ax=$(echo "1 - $lambda" | bc)
    decimal_places=$(echo "$lambda" | awk -F. '{print length($2)}')
    ax=$(printf "%.${decimal_places}f" $ax)

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lradj}_${train_epochs}_${patience}_${batch_size}_${auxi_mode}_${auxi_type}_${auxi_loss}_${module_first}_${first_order}_${overlap_ratio}_${auxi_batch_size}_${max_norm}_${num_tasks}_${meta_inner_steps}_${meta_steps}_${lr_inner}_${lr_meta}
    OUTPUT_DIR="${OUT_ROOT}/results/${EXP_NAME}/${JOB_NAME}"

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
        if [ ${#subdirs[@]} -eq 1 ] && [ -f "${subdirs[0]}/metrics.yaml" ]; then
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
            --root_path $DATA_ROOT/traffic/ \
            --data_path traffic.csv \
            --model_id "${dst}_96_${pl}" \
            --model ${model_name} \
            --data_id $dst \
            --data custom \
            --features M \
            --seq_len 96 \
            --label_len 48 \
            --pred_len ${pl} \
            --enc_in 862 \
            --dec_in 862 \
            --c_out 862 \
            --e_layers 4 \
            --d_layers 1 \
            --factor 3 \
            --d_model 512 \
            --d_ff 512 \
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
            --fix_seed ${seed} \
            --checkpoints $CHECKPOINTS \
            --results $RESULTS \
            --test_results $TEST_RESULTS \
            --log_path $LOG_PATH \
            --rerun $rerun \
            --auxi_mode $auxi_mode \
            --auxi_type $auxi_type \
            --auxi_loss $auxi_loss \
            --module_first $module_first \
            --inner_lr $lr_inner \
            --meta_lr $lr_meta \
            --meta_inner_steps $meta_inner_steps \
            --overlap_ratio $overlap_ratio \
            --num_tasks $num_tasks \
            --max_norm $max_norm \
            --first_order $first_order \
            --warmup_steps $meta_steps \
            --auxi_batch_size ${auxi_batch_size}

        sleep 5
    # } 2>&1 | tee -a "${OUTPUT_DIR}/stdout.log" &
    } &
done
done
done
done
done
done
done
done
done
done
done
done
done






# hyper-parameters
dst=Weather
# pl_list=(96 192 336 720)
# lbd_list=(0.4 0.6)
# lr_list=(0.001 0.002)
# lradj_list=(type1)
# bs_list=(32)
# meta_lr_list=(0.1 0.2 0.05 0.02)
# meta_inner_steps_list=(1 4 5)
# inner_lr_list=(same)
# num_tasks_list=(3 4 5)
# auxi_bs_list=(64)
# max_norm_list=(5.0)
# meta_step_list=(700)
# auxi_loss_list=(MAE)

pl_list=(96 192 336 720)
lbd_list=(0.4 0.2 0.1 0.8 0.05 0.6)
lr_list=(0.001 0.0005 0.0002 0.0001)
lradj_list=(type1 type3)
bs_list=(32)
meta_lr_list=(0.1 0.05 0.02)
meta_inner_steps_list=(1 3 5 7)
inner_lr_list=(same)
num_tasks_list=(3 5 7)
auxi_bs_list=(64)
max_norm_list=(5.0)
meta_step_list=(500 300 100)
auxi_loss_list=(MAE)

train_epochs=30
patience=5
batch_size=32

rerun=0

for lr in ${lr_list[@]}; do
for inner_lr in ${inner_lr_list[@]}; do
if [[ $inner_lr == "double" ]]; then
    lr_inner=$(echo "scale=10; $lr * 2" | bc)
elif [[ $inner_lr == "half" ]]; then
    lr_inner=$(echo "scale=10; $lr / 2" | bc)
elif [[ $inner_lr == "same" ]]; then
    lr_inner=$lr
else
    lr_inner=$inner_lr
fi
[[ "$lr_inner" == .* ]] && lr_inner="0$lr_inner"
lr_inner=$(echo "$lr_inner" | sed 's/0*$//; s/\.$//')

for meta_lr in ${meta_lr_list[@]}; do
if [[ $meta_lr == "double" ]]; then
    lr_meta=$(echo "scale=10; $lr * 2" | bc)
elif [[ $meta_lr == "half" ]]; then
    lr_meta=$(echo "scale=10; $lr / 2" | bc)
elif [[ $meta_lr == "same" ]]; then
    lr_meta=$lr
else
    lr_meta=$meta_lr
fi
[[ "$lr_meta" == .* ]] && lr_meta="0$lr_meta"
lr_meta=$(echo "$lr_meta" | sed 's/0*$//; s/\.$//')

for meta_steps in ${meta_step_list[@]}; do
for max_norm in ${max_norm_list[@]}; do
for num_tasks in ${num_tasks_list[@]}; do
for meta_inner_steps in ${meta_inner_steps_list[@]}; do
for batch_size in ${bs_list[@]}; do
for auxi_batch_size in ${auxi_bs_list[@]}; do
for lradj in ${lradj_list[@]}; do
for lambda in ${lbd_list[@]}; do
for auxi_loss in ${auxi_loss_list[@]}; do
for pl in ${pl_list[@]}; do
    if ! [[ " ${datasets[@]} " =~ " ${dst} " ]]; then
        continue
    fi

    rl=$lambda
    ax=$(echo "1 - $lambda" | bc)
    decimal_places=$(echo "$lambda" | awk -F. '{print length($2)}')
    ax=$(printf "%.${decimal_places}f" $ax)

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lradj}_${train_epochs}_${patience}_${batch_size}_${auxi_mode}_${auxi_type}_${auxi_loss}_${module_first}_${first_order}_${overlap_ratio}_${auxi_batch_size}_${max_norm}_${num_tasks}_${meta_inner_steps}_${meta_steps}_${lr_inner}_${lr_meta}
    OUTPUT_DIR="${OUT_ROOT}/results/${EXP_NAME}/${JOB_NAME}"

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
        if [ ${#subdirs[@]} -eq 1 ] && [ -f "${subdirs[0]}/metrics.yaml" ]; then
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
            --e_layers 3 \
            --d_layers 1 \
            --factor 3 \
            --d_model 512 \
            --d_ff 512 \
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
            --fix_seed ${seed} \
            --checkpoints $CHECKPOINTS \
            --results $RESULTS \
            --test_results $TEST_RESULTS \
            --log_path $LOG_PATH \
            --rerun $rerun \
            --auxi_mode $auxi_mode \
            --auxi_type $auxi_type \
            --auxi_loss $auxi_loss \
            --module_first $module_first \
            --inner_lr $lr_inner \
            --meta_lr $lr_meta \
            --meta_inner_steps $meta_inner_steps \
            --overlap_ratio $overlap_ratio \
            --num_tasks $num_tasks \
            --max_norm $max_norm \
            --first_order $first_order \
            --warmup_steps $meta_steps \
            --auxi_batch_size ${auxi_batch_size}

        sleep 5
    # } 2>&1 | tee -a "${OUTPUT_DIR}/stdout.log" &
    } &
done
done
done
done
done
done
done
done
done
done
done
done
done






# hyper-parameters
dst=PEMS03
# pl_list=(12 24 36 48)
# lbd_list=(0.4 0.6 0.8)
# lr_list=(0.001 0.002)
# lradj_list=(type1)
# bs_list=(32)
# meta_lr_list=(0.015 0.05)
# meta_inner_steps_list=(1 2)
# inner_lr_list=(same)
# num_tasks_list=(3)
# auxi_bs_list=(64)
# max_norm_list=(5.0)
# meta_step_list=(200)
# auxi_loss_list=(MAE)

# pl_list=(12 24 36 48)
# lbd_list=(0.1 0.6 0.8 0.9)
# lr_list=(0.001 0.0005 0.0002)
# lradj_list=(type1)
# bs_list=(32)
# meta_lr_list=(0.1 0.05 0.2)
# meta_inner_steps_list=(1 2 3)
# inner_lr_list=(same)
# num_tasks_list=(1 3 5)
# auxi_bs_list=(64)
# max_norm_list=(5.0)
# meta_step_list=(200 300)
# auxi_loss_list=(MAE)

pl_list=(12 24 36 48)
lbd_list=(0.7 0.75 0.85 0.95 0.8 0.9)
lr_list=(0.001 0.0015 0.0008 0.002)
lradj_list=(type1 type3)
bs_list=(32)
meta_lr_list=(0.1 0.05 0.02)
meta_inner_steps_list=(1 2)
inner_lr_list=(same)
num_tasks_list=(2 3)
auxi_bs_list=(64)
max_norm_list=(5.0)
meta_step_list=(200 100 50)
auxi_loss_list=(MAE)

train_epochs=30
patience=5
batch_size=32

rerun=0

for lr in ${lr_list[@]}; do
for inner_lr in ${inner_lr_list[@]}; do
if [[ $inner_lr == "double" ]]; then
    lr_inner=$(echo "scale=10; $lr * 2" | bc)
elif [[ $inner_lr == "half" ]]; then
    lr_inner=$(echo "scale=10; $lr / 2" | bc)
elif [[ $inner_lr == "same" ]]; then
    lr_inner=$lr
else
    lr_inner=$inner_lr
fi
[[ "$lr_inner" == .* ]] && lr_inner="0$lr_inner"
lr_inner=$(echo "$lr_inner" | sed 's/0*$//; s/\.$//')

for meta_lr in ${meta_lr_list[@]}; do
if [[ $meta_lr == "double" ]]; then
    lr_meta=$(echo "scale=10; $lr * 2" | bc)
elif [[ $meta_lr == "half" ]]; then
    lr_meta=$(echo "scale=10; $lr / 2" | bc)
elif [[ $meta_lr == "same" ]]; then
    lr_meta=$lr
else
    lr_meta=$meta_lr
fi
[[ "$lr_meta" == .* ]] && lr_meta="0$lr_meta"
lr_meta=$(echo "$lr_meta" | sed 's/0*$//; s/\.$//')

for meta_steps in ${meta_step_list[@]}; do
for max_norm in ${max_norm_list[@]}; do
for num_tasks in ${num_tasks_list[@]}; do
for meta_inner_steps in ${meta_inner_steps_list[@]}; do
for batch_size in ${bs_list[@]}; do
for auxi_batch_size in ${auxi_bs_list[@]}; do
for lradj in ${lradj_list[@]}; do
for lambda in ${lbd_list[@]}; do
for auxi_loss in ${auxi_loss_list[@]}; do
for pl in ${pl_list[@]}; do
    if ! [[ " ${datasets[@]} " =~ " ${dst} " ]]; then
        continue
    fi

    rl=$lambda
    ax=$(echo "1 - $lambda" | bc)
    decimal_places=$(echo "$lambda" | awk -F. '{print length($2)}')
    ax=$(printf "%.${decimal_places}f" $ax)

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lradj}_${train_epochs}_${patience}_${batch_size}_${auxi_mode}_${auxi_type}_${auxi_loss}_${module_first}_${first_order}_${overlap_ratio}_${auxi_batch_size}_${max_norm}_${num_tasks}_${meta_inner_steps}_${meta_steps}_${lr_inner}_${lr_meta}
    OUTPUT_DIR="${OUT_ROOT}/results/${EXP_NAME}/${JOB_NAME}"

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
        if [ ${#subdirs[@]} -eq 1 ] && [ -f "${subdirs[0]}/metrics.yaml" ]; then
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
            --e_layers 4 \
            --d_layers 1 \
            --factor 3 \
            --d_model 512 \
            --d_ff 512 \
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
            --fix_seed ${seed} \
            --checkpoints $CHECKPOINTS \
            --results $RESULTS \
            --test_results $TEST_RESULTS \
            --log_path $LOG_PATH \
            --rerun $rerun \
            --auxi_mode $auxi_mode \
            --auxi_type $auxi_type \
            --auxi_loss $auxi_loss \
            --module_first $module_first \
            --inner_lr $lr_inner \
            --meta_lr $lr_meta \
            --meta_inner_steps $meta_inner_steps \
            --overlap_ratio $overlap_ratio \
            --num_tasks $num_tasks \
            --max_norm $max_norm \
            --first_order $first_order \
            --warmup_steps $meta_steps \
            --auxi_batch_size ${auxi_batch_size}

        sleep 5
    # } 2>&1 | tee -a "${OUTPUT_DIR}/stdout.log" &
    } &
done
done
done
done
done
done
done
done
done
done
done
done
done








# hyper-parameters
dst=PEMS08

pl_list=(12 24 36 48)
lbd_list=(0.7 0.8 0.6)
lr_list=(0.001 0.002 0.0005)
lradj_list=(type1 type3)
bs_list=(32)
meta_lr_list=(0.01 0.05)
meta_inner_steps_list=(2 3)
inner_lr_list=(same)
num_tasks_list=(3 4)
auxi_bs_list=(64)
max_norm_list=(5.0)
meta_step_list=(300 100)
auxi_loss_list=(MAE)

train_epochs=30
patience=5
batch_size=32

rerun=0

for lr in ${lr_list[@]}; do
for inner_lr in ${inner_lr_list[@]}; do
if [[ $inner_lr == "double" ]]; then
    lr_inner=$(echo "scale=10; $lr * 2" | bc)
elif [[ $inner_lr == "half" ]]; then
    lr_inner=$(echo "scale=10; $lr / 2" | bc)
elif [[ $inner_lr == "same" ]]; then
    lr_inner=$lr
else
    lr_inner=$inner_lr
fi
[[ "$lr_inner" == .* ]] && lr_inner="0$lr_inner"
lr_inner=$(echo "$lr_inner" | sed 's/0*$//; s/\.$//')

for meta_lr in ${meta_lr_list[@]}; do
if [[ $meta_lr == "double" ]]; then
    lr_meta=$(echo "scale=10; $lr * 2" | bc)
elif [[ $meta_lr == "half" ]]; then
    lr_meta=$(echo "scale=10; $lr / 2" | bc)
elif [[ $meta_lr == "same" ]]; then
    lr_meta=$lr
else
    lr_meta=$meta_lr
fi
[[ "$lr_meta" == .* ]] && lr_meta="0$lr_meta"
lr_meta=$(echo "$lr_meta" | sed 's/0*$//; s/\.$//')

for meta_steps in ${meta_step_list[@]}; do
for max_norm in ${max_norm_list[@]}; do
for num_tasks in ${num_tasks_list[@]}; do
for meta_inner_steps in ${meta_inner_steps_list[@]}; do
for batch_size in ${bs_list[@]}; do
for auxi_batch_size in ${auxi_bs_list[@]}; do
for lradj in ${lradj_list[@]}; do
for lambda in ${lbd_list[@]}; do
for auxi_loss in ${auxi_loss_list[@]}; do
for pl in ${pl_list[@]}; do
    if ! [[ " ${datasets[@]} " =~ " ${dst} " ]]; then
        continue
    fi

    rl=$lambda
    ax=$(echo "1 - $lambda" | bc)
    decimal_places=$(echo "$lambda" | awk -F. '{print length($2)}')
    ax=$(printf "%.${decimal_places}f" $ax)

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lradj}_${train_epochs}_${patience}_${batch_size}_${auxi_mode}_${auxi_type}_${auxi_loss}_${module_first}_${first_order}_${overlap_ratio}_${auxi_batch_size}_${max_norm}_${num_tasks}_${meta_inner_steps}_${meta_steps}_${lr_inner}_${lr_meta}
    OUTPUT_DIR="${OUT_ROOT}/results/${EXP_NAME}/${JOB_NAME}"

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
        if [ ${#subdirs[@]} -eq 1 ] && [ -f "${subdirs[0]}/metrics.yaml" ]; then
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
            --e_layers 3 \
            --d_layers 1 \
            --factor 3 \
            --d_model 512 \
            --d_ff 512 \
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
            --fix_seed ${seed} \
            --checkpoints $CHECKPOINTS \
            --results $RESULTS \
            --test_results $TEST_RESULTS \
            --log_path $LOG_PATH \
            --rerun $rerun \
            --auxi_mode $auxi_mode \
            --auxi_type $auxi_type \
            --auxi_loss $auxi_loss \
            --module_first $module_first \
            --inner_lr $lr_inner \
            --meta_lr $lr_meta \
            --meta_inner_steps $meta_inner_steps \
            --overlap_ratio $overlap_ratio \
            --num_tasks $num_tasks \
            --max_norm $max_norm \
            --first_order $first_order \
            --warmup_steps $meta_steps \
            --auxi_batch_size ${auxi_batch_size}

        sleep 5
    # } 2>&1 | tee -a "${OUTPUT_DIR}/stdout.log" &
    } &
done
done
done
done
done
done
done
done
done
done
done
done
done




wait