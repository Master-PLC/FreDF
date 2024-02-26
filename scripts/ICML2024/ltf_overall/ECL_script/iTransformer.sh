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
OUTPUT_DIR=./exp_results/ltf_overall

model_name=iTransformer
seed=2023
des='LTFAll'

auxi_loss="MAE"
module_first=1
lr=0.0001
auxi_mode='rfft'

lbd_list=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
pl_list=(96 192 336 720)

for lambda in ${lbd_list[@]}; do
    JOB_DIR=$OUTPUT_DIR/${model_name}_ECL_${lambda}
    mkdir -p $JOB_DIR

    CHECKPOINTS=$JOB_DIR/checkpoints/
    RESULTS=$JOB_DIR/results/
    TEST_RESULTS=$JOB_DIR/test_results/
    LOG_PATH=$JOB_DIR/result_long_term_forecast.txt

    rl=$lambda
    ax=$(echo "1 - $lambda" | bc)

    for pl in ${pl_list[@]}; do
        read -u 9
        {
            echo "${P}"
            python -u run.py \
                --task_name long_term_forecast \
                --is_training 1 \
                --root_path $DATA_ROOT/electricity/ \
                --data_path electricity.csv \
                --model_id "ECL_96_${pl}" \
                --model ${model_name} \
                --data custom \
                --features M \
                --seq_len 96 \
                --label_len 48 \
                --pred_len ${pl} \
                --e_layers 3 \
                --d_layers 1 \
                --factor 3 \
                --enc_in 321 \
                --dec_in 321 \
                --c_out 321 \
                --des ${des} \
                --d_model 512 \
                --d_ff 512 \
                --batch_size 16 \
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
