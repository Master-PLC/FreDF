export CUDA_VISIBLE_DEVICES=$4


set -e

PROC_NUM=4
FIFO_FILE="/tmp/$$.fifo"
mkfifo ${FIFO_FILE}
exec 9<>${FIFO_FILE}
for process_num in $(seq ${PROC_NUM}); do
    echo "$(date +%F\ %T) Processor-${process_num} Start " >&9
done

DATA_ROOT=./dataset
AMLT_OUTPUT_DIR=$3

CHECKPOINTS=$AMLT_OUTPUT_DIR/checkpoints/
RESULTS=$AMLT_OUTPUT_DIR/results/
TEST_RESULTS=$AMLT_OUTPUT_DIR/test_results/
LOG_PATH=$AMLT_OUTPUT_DIR/result_long_term_forecast.txt

model_name=iTransformer
seed=2023
des='Stress-test'
auxi_loss="MAE"
module_first=1

lambda=$1
rl=$lambda
ax=$(echo "1 - $lambda" | bc)
auxi_mode='rfft'
data_percentage=$5
lr=$2

pl_list=("${@:6}")
echo "pl_list: ${pl_list[@]}"

for pl in ${pl_list[@]}; do
    read -u 9
    {
        echo "${P}"
        python -u run.py \
            --task_name long_term_forecast \
            --is_training 1 \
            --root_path $DATA_ROOT/weather/ \
            --data_path weather.csv \
            --model_id "weather_96_${pl}" \
            --model ${model_name} \
            --data custom \
            --features M \
            --seq_len 96 \
            --label_len 48 \
            --pred_len ${pl} \
            --e_layers 3 \
            --d_layers 1 \
            --factor 3 \
            --enc_in 21 \
            --dec_in 21 \
            --c_out 21 \
            --des ${des} \
            --d_model 512 \
            --d_ff 512 \
            --itr 1 \
            --learning_rate ${lr} \
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
            --data_percentage ${data_percentage}
        echo ${P} >&9
    } &
done
wait
exec 9>&-
echo "All finished ..."
