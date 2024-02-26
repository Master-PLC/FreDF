export CUDA_VISIBLE_DEVICES=0


set -e

PROC_NUM=1
FIFO_FILE="/tmp/$$.fifo"
mkfifo ${FIFO_FILE}
exec 9<>${FIFO_FILE}
for process_num in $(seq ${PROC_NUM}); do
    echo "$(date +%F\ %T) Processor-${process_num} Start " >&9
done

DATA_ROOT=./dataset
OUTPUT_DIR=./exp_results/stf_overall

model_name=DLinear
seed=2023
des='STFAll'

auxi_loss="MAE"
module_first=1
lambda=1.0
rl=$lambda
ax=$(echo "1 - $lambda" | bc)
lr=0.001
auxi_mode='rfft'

JOB_DIR=$OUTPUT_DIR/${model_name}_M4_${lambda}
mkdir -p $JOB_DIR

CHECKPOINTS=$JOB_DIR/checkpoints/
RESULTS=$JOB_DIR/results/
TEST_RESULTS=$JOB_DIR/test_results/
LOG_PATH=$JOB_DIR/result_long_term_forecast.txt

span_list=("Yearly" "Quarterly" "Monthly" "Weekly" "Daily" "Hourly")

for span in ${span_list[@]}; do
    read -u 9
    {
        echo "${P}"
        python -u run.py \
            --task_name short_term_forecast \
            --is_training 1 \
            --root_path $DATA_ROOT/m4 \
            --seasonal_patterns ${span} \
            --model_id "m4_${span}" \
            --model $model_name \
            --data m4 \
            --features M \
            --e_layers 2 \
            --d_layers 1 \
            --factor 3 \
            --enc_in 1 \
            --dec_in 1 \
            --c_out 1 \
            --batch_size 16 \
            --d_model 512 \
            --des ${des} \
            --itr 1 \
            --learning_rate 0.001 \
            --loss 'SMAPE' \
            --auxi_lambda ${ax} \
            --rec_lambda ${rl} \
            --auxi_loss ${auxi_loss} \
            --module_first ${module_first} \
            --fix_seed ${seed} \
            --checkpoints $CHECKPOINTS \
            --results $RESULTS \
            --test_results $TEST_RESULTS \
            --log_path $LOG_PATH \
            --auxi_mode ${auxi_mode}
        echo ${P} >&9
    } &
done
wait
exec 9>&-
echo "All finished ..."
