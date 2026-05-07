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

DATA_ROOT=$USRDIR/dataset
OUT_ROOT=/mnt/tidalfs-bdsz01/dataset/llm_ckpt/plc_data/FreDF
EXP_NAME=long_term
seed=2023
des='iTransformer'

model_name=iTransformer

auxi_mode=rfft
auxi_type=complex
module_first=1
first_order=1
overlap_ratio=0.0
test_batch_size=1

datasets=(ETTh1 ETTh2 ETTm1 ETTm2 ECL Traffic Weather PEMS03 PEMS08)



# hyper-parameters
dst=ETTh1
pl_list=(96 192 336 720)
lbd_list=(0.2)
lr_list=(0.001 0.005 0.002)
lradj_list=(type1)
bs_list=(32)
meta_lr_list=(0.1 0.2 0.05)
meta_inner_steps_list=(1 2 3 4)
inner_lr_list=(same)
num_tasks_list=(3)
auxi_bs_list=(64)
max_norm_list=(5.0)
meta_step_list=(300)
auxi_loss_list=(MAE MSE)

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
            --task_name long_term_forecast \
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
pl_list=(96 192 336 720)
lbd_list=(0.2)
lr_list=(0.001 0.005 0.002)
lradj_list=(type1)
bs_list=(32)
meta_lr_list=(0.1 0.2 0.05)
meta_inner_steps_list=(1 2 3 4)
inner_lr_list=(same)
num_tasks_list=(3)
auxi_bs_list=(64)
max_norm_list=(5.0)
meta_step_list=(300)
auxi_loss_list=(MAE MSE)

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

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lradj}_${train_epochs}_${patience}_${batch_size}_${auxi_mode}_${auxi_type}_${auxi_loss}_${module_first}
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
            --task_name long_term_forecast \
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
pl_list=(96 192 336 720)
lbd_list=(0.2)
lr_list=(0.001 0.005 0.002)
lradj_list=(type1)
bs_list=(32)
meta_lr_list=(0.1 0.2 0.05)
meta_inner_steps_list=(1 2 3 4)
inner_lr_list=(same)
num_tasks_list=(3)
auxi_bs_list=(64)
max_norm_list=(5.0)
meta_step_list=(300)
auxi_loss_list=(MAE MSE)

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

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lradj}_${train_epochs}_${patience}_${batch_size}_${auxi_mode}_${auxi_type}_${auxi_loss}_${module_first}
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
            --task_name long_term_forecast \
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
pl_list=(96 192 336 720)
lbd_list=(0.2)
lr_list=(0.001 0.005 0.002)
lradj_list=(type1)
bs_list=(32)
meta_lr_list=(0.1 0.2 0.05)
meta_inner_steps_list=(1 2 3 4)
inner_lr_list=(same)
num_tasks_list=(3)
auxi_bs_list=(64)
max_norm_list=(5.0)
meta_step_list=(300)
auxi_loss_list=(MAE MSE)

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

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lradj}_${train_epochs}_${patience}_${batch_size}_${auxi_mode}_${auxi_type}_${auxi_loss}_${module_first}
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
            --task_name long_term_forecast \
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
pl_list=(96 192 336 720)
lbd_list=(0.2)
lr_list=(0.001 0.005 0.002)
lradj_list=(type1)
bs_list=(32)
meta_lr_list=(0.1 0.2 0.05)
meta_inner_steps_list=(1 2 3 4)
inner_lr_list=(same)
num_tasks_list=(3)
auxi_bs_list=(64)
max_norm_list=(5.0)
meta_step_list=(300)
auxi_loss_list=(MAE MSE)

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

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lradj}_${train_epochs}_${patience}_${batch_size}_${auxi_mode}_${auxi_type}_${auxi_loss}_${module_first}
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
            --task_name long_term_forecast \
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
pl_list=(96 192 336 720)
lbd_list=(0.2)
lr_list=(0.001 0.005 0.002)
lradj_list=(type1)
bs_list=(32)
meta_lr_list=(0.1 0.2 0.05)
meta_inner_steps_list=(1 2 3 4)
inner_lr_list=(same)
num_tasks_list=(3)
auxi_bs_list=(64)
max_norm_list=(5.0)
meta_step_list=(300)
auxi_loss_list=(MAE MSE)

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

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lradj}_${train_epochs}_${patience}_${batch_size}_${auxi_mode}_${auxi_type}_${auxi_loss}_${module_first}
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
            --task_name long_term_forecast \
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
pl_list=(96 192 336 720)
lbd_list=(0.2)
lr_list=(0.001 0.005 0.002)
lradj_list=(type1)
bs_list=(32)
meta_lr_list=(0.1 0.2 0.05)
meta_inner_steps_list=(1 2 3 4)
inner_lr_list=(same)
num_tasks_list=(3)
auxi_bs_list=(64)
max_norm_list=(5.0)
meta_step_list=(300)
auxi_loss_list=(MAE MSE)

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

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lradj}_${train_epochs}_${patience}_${batch_size}_${auxi_mode}_${auxi_type}_${auxi_loss}_${module_first}
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
            --task_name long_term_forecast \
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
pl_list=(12 24 36 48)
lbd_list=(0.2)
lr_list=(0.001 0.005 0.002)
lradj_list=(type1)
bs_list=(32)
meta_lr_list=(0.1 0.2 0.05)
meta_inner_steps_list=(1 2 3 4)
inner_lr_list=(same)
num_tasks_list=(3)
auxi_bs_list=(64)
max_norm_list=(5.0)
meta_step_list=(300)
auxi_loss_list=(MAE MSE)

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

    case $pl in
        12) lambda=0.4;;
        24) lambda=0.4;;
        36) lambda=0.6;;
        48) lambda=0.6;;
    esac

    rl=$lambda
    ax=$(echo "1 - $lambda" | bc)
    decimal_places=$(echo "$lambda" | awk -F. '{print length($2)}')
    ax=$(printf "%.${decimal_places}f" $ax)

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lradj}_${train_epochs}_${patience}_${batch_size}_${auxi_mode}_${auxi_type}_${auxi_loss}_${module_first}
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
            --task_name long_term_forecast \
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
lbd_list=(0.2)
lr_list=(0.001 0.005 0.002)
lradj_list=(type1)
bs_list=(32)
meta_lr_list=(0.1 0.2 0.05)
meta_inner_steps_list=(1 2 3 4)
inner_lr_list=(same)
num_tasks_list=(3)
auxi_bs_list=(64)
max_norm_list=(5.0)
meta_step_list=(300)
auxi_loss_list=(MAE MSE)

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

    case $pl in
        12) lambda=0.9;;
        24) lambda=0.8;;
        36) lambda=0.9;;
        48) lambda=0.7;;
    esac

    rl=$lambda
    ax=$(echo "1 - $lambda" | bc)
    decimal_places=$(echo "$lambda" | awk -F. '{print length($2)}')
    ax=$(printf "%.${decimal_places}f" $ax)

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lradj}_${train_epochs}_${patience}_${batch_size}_${auxi_mode}_${auxi_type}_${auxi_loss}_${module_first}
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
            --task_name long_term_forecast \
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