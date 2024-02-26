export CUDA_VISIBLE_DEVICES=0

set -e

PROC_NUM=2
FIFO_FILE="/tmp/$$.fifo"
mkfifo ${FIFO_FILE}
exec 9<>${FIFO_FILE}
for process_num in $(seq ${PROC_NUM}); do
    echo "$(date +%F\ %T) Processor-${process_num} Start " >&9
done

DATA_ROOT=./dataset
OUTPUT_DIR=./exp_results/imp_autoencoder

model_name=DLinear
seed=2023
des='ImpAE'

auxi_loss="MAE"
module_first=1
lambda=1.0
rl=$lambda
ax=$(echo "1 - $lambda" | bc)
lr=0.001
auxi_mode='rfft-fill'
reconstruction_type='autoencoder'

JOB_DIR=$OUTPUT_DIR/${model_name}_ETTh1_${lambda}
mkdir -p $JOB_DIR

CHECKPOINTS=$JOB_DIR/checkpoints/
RESULTS=$JOB_DIR/results/
TEST_RESULTS=$JOB_DIR/test_results/
LOG_PATH=$JOB_DIR/result_imputation.txt

mask_rate_list=(0.125 0.25 0.375 0.5)

for mask_rate in ${mask_rate_list[@]}; do
    read -u 9
    {
        echo "${P}"
        python -u run.py \
            --task_name imputation \
            --is_training 1 \
            --root_path $DATA_ROOT/ETT-small/ \
            --data_path ETTh1.csv \
            --model_id "ETTh1_mask_${mask_rate}" \
            --mask_rate ${mask_rate} \
            --model $model_name \
            --data ETTh1 \
            --features M \
            --seq_len 96 \
            --label_len 0 \
            --pred_len 96 \
            --e_layers 2 \
            --d_layers 1 \
            --factor 3 \
            --enc_in 7 \
            --dec_in 7 \
            --c_out 7 \
            --batch_size 16 \
            --d_model 128 \
            --d_ff 128 \
            --des ${des} \
            --itr 1 \
            --top_k 5 \
            --learning_rate 0.001 \
            --auxi_lambda ${ax} \
            --rec_lambda ${rl} \
            --auxi_loss ${auxi_loss} \
            --module_first ${module_first} \
            --fix_seed ${seed} \
            --checkpoints $CHECKPOINTS \
            --results $RESULTS \
            --test_results $TEST_RESULTS \
            --log_path $LOG_PATH \
            --auxi_mode ${auxi_mode} \
            --reconstruction_type ${reconstruction_type}
        echo ${P} >&9
    } &
done
wait
exec 9>&-
