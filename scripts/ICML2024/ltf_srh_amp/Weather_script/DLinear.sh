export CUDA_VISIBLE_DEVICES=$4

# +-----------------------------------------------------------------------------+
# | Processes:                                                                  |
# |  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
# |        ID   ID                                                   Usage      |
# |=============================================================================|
# |    0   N/A  N/A      1763      G   /usr/lib/xorg/Xorg                 46MiB |
# |    0   N/A  N/A    317726      C   long_term_forecast               1327MiB |
# |    0   N/A  N/A    317728      C   long_term_forecast               1379MiB |
# |    0   N/A  N/A    317730      C   long_term_forecast               1475MiB |
# |    0   N/A  N/A    317731      C   long_term_forecast               1723MiB |
# +-----------------------------------------------------------------------------+

set -e

PROC_NUM=2
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

model_name=DLinear
seed=2023
des='Srh-amp'
auxi_loss="MAE"
module_first=1

rl=1
ax=0
noise_amp=$1
noise_type='sin'
noise_freq_percentage=$2
auxi_mode='rfft'

pl_list=("${@:5}")
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
            --e_layers 2 \
            --d_layers 1 \
            --factor 3 \
            --enc_in 21 \
            --dec_in 21 \
            --c_out 21 \
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
            --log_path $LOG_PATH \
            --add_noise \
            --noise_amp ${noise_amp} \
            --noise_type ${noise_type} \
            --noise_freq_percentage ${noise_freq_percentage} \
            --cutoff_freq_percentage ${cutoff_freq_percentage} \
            --auxi_mode ${auxi_mode}
        echo ${P} >&9
    } &
done
wait
exec 9>&-
echo "All finished ..."