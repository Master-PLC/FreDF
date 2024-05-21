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
OUTPUT_DIR=./exp_results/gene_mode

CHECKPOINTS=$OUTPUT_DIR/checkpoints/
RESULTS=$OUTPUT_DIR/results/
TEST_RESULTS=$OUTPUT_DIR/test_results/
LOG_PATH=$OUTPUT_DIR/result_long_term_forecast.txt

model_name=iTransformer
seed=2023
des='GeneMode'

auxi_loss="MAE"
module_first=1
lambda=0.2
rl=$lambda
ax=$(echo "1 - $lambda" | bc)
lr=0.0005

auxi_mode_list=("rfft-D" "rfft-2D")
pl_list=(96 192 336 720)

for auxi_mode in ${auxi_mode_list[@]}; do
    JOB_DIR=$OUTPUT_DIR/${model_name}_ETTm1_${auxi_mode}
    mkdir -p $JOB_DIR

    CHECKPOINTS=$JOB_DIR/checkpoints/
    RESULTS=$JOB_DIR/results/
    TEST_RESULTS=$JOB_DIR/test_results/
    LOG_PATH=$JOB_DIR/result_long_term_forecast.txt

    for pl in ${pl_list[@]}; do
        read -u 9
        {
            echo "${P}"
            python -u run.py \
                --task_name long_term_forecast \
                --is_training 1 \
                --root_path $DATA_ROOT/ETT-small/ \
                --data_path ETTm1.csv \
                --model_id "ETTm1_96_${pl}_${auxi_loss}" \
                --model ${model_name} \
                --data ETTm1 \
                --features M \
                --seq_len 96 \
                --label_len 48 \
                --pred_len ${pl} \
                --e_layers 2 \
                --d_layers 1 \
                --factor 3 \
                --enc_in 7 \
                --dec_in 7 \
                --c_out 7 \
                --des ${des} \
                --d_model 128 \
                --d_ff 128 \
                --learning_rate ${lr} \
                --itr 1 \
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
done
wait
exec 9>&-
echo "All finished ..."
