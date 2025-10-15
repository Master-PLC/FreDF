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
OUTPUT_DIR=./exp_results/aba_type

model_name=iTransformer
seed=2023
des='AbaType'

module_first=1
lambda=0.1
rl=$lambda
ax=$(echo "1 - $lambda" | bc)
lr=0.001
auxi_mode='rfft'
auxi_loss="MAE"

auxi_type_list=("mag" "phase" "mag-phase" "complex-phase" "complex-mag-phase")
pl_list=(96 192 336 720)


for auxi_type in ${auxi_type_list[@]}; do
    JOB_DIR=$OUTPUT_DIR/${model_name}_ECL_${auxi_type}
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
                --root_path $DATA_ROOT/electricity/ \
                --data_path electricity.csv \
                --model_id "ECL_96_${pl}_${auxi_type}" \
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
                --auxi_mode ${auxi_mode} \
                --auxi_type ${auxi_type}
            echo ${P} >&9
        } &
    done
done
wait
exec 9>&-
echo "All finished ..."
