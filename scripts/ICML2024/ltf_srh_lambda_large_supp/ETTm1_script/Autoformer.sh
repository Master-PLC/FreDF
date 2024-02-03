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
lambda=$1

CHECKPOINTS=$AMLT_OUTPUT_DIR/checkpoints/
RESULTS=$AMLT_OUTPUT_DIR/results/
TEST_RESULTS=$AMLT_OUTPUT_DIR/test_results/
LOG_PATH=$AMLT_OUTPUT_DIR/result_long_term_forecast.txt

model_name=Autoformer
seed=2023
des='Srh-lambda'
auxi_loss="MAE"
module_first=1

rl=$lambda
ax=$(echo "1 - $lambda" | bc)

if [[ $lambda == "0" ]] || [[ $lambda == "0.1" ]] || [[ $lambda == "0.3" ]] || [[ $lambda == "0.5" ]] || [[ $lambda == "0.9" ]]; then
    pl_values=(96 192 336 720)
elif [[ $lambda == "0.2" ]]; then
    pl_values=(96 336 720)
elif [[ $lambda == "0.4" ]] || [[ $lambda == "0.6" ]] || [[ $lambda == "0.8" ]]; then
    pl_values=(96 192 720)
elif [[ $lambda == "0.7" ]]; then
    pl_values=(192 336)
elif [[ $lambda == "1" ]]; then
    pl_values=(96 336)
fi

for pl in "${pl_values[@]}"; do
    read -u 9
    {
        echo "${P}"
        python -u run.py \
            --task_name long_term_forecast \
            --is_training 1 \
            --root_path $DATA_ROOT/ETT-small/ \
            --data_path ETTm1.csv \
            --model_id "ETTm1_96_${pl}" \
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
            --itr 1 \
            --auxi_lambda ${ax} \
            --rec_lambda ${rl} \
            --auxi_loss ${auxi_loss} \
            --module_first ${module_first} \
            --fix_seed ${seed} \
            --checkpoints $CHECKPOINTS \
            --results $RESULTS \
            --test_results $TEST_RESULTS \
            --log_path $LOG_PATH
        echo ${P} >&9
    } &
done
wait
exec 9>&-
echo "All finished ..."
