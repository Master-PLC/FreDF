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

model_name=PatchTST
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

JOB_DIR=$OUTPUT_DIR/${model_name}_Weather_${lambda}
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
            --root_path $DATA_ROOT/weather/ \
            --data_path weather.csv \
            --model_id "weather_mask_${mask_rate}" \
            --mask_rate ${mask_rate} \
            --model $model_name \
            --data custom \
            --features M \
            --seq_len 96 \
            --label_len 0 \
            --pred_len 0 \
            --e_layers 3 \
            --d_layers 1 \
            --factor 3 \
            --enc_in 21 \
            --dec_in 21 \
            --c_out 21 \
            --batch_size 128 \
            --des ${des} \
            --itr 1 \
            --n_heads 2 \
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
echo "All finished ..."