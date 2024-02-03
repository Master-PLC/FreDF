export CUDA_VISIBLE_DEVICES=0

# +-----------------------------------------------------------------------------+
# | Processes:                                                                  |
# |  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
# |        ID   ID                                                   Usage      |
# |=============================================================================|
# |    0   N/A  N/A      1763      G   /usr/lib/xorg/Xorg                 46MiB |
# |    0   N/A  N/A    652613      C   long_term_forecast               2035MiB |
# |    0   N/A  N/A    652615      C   long_term_forecast               2033MiB |
# |    0   N/A  N/A    652616      C   long_term_forecast               2057MiB |
# |    0   N/A  N/A    652617      C   long_term_forecast               2109MiB |
# +-----------------------------------------------------------------------------+

set -e

PROC_NUM=4
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

model_name=PatchTST
seed=2023
des='Srh-lambda'
auxi_loss="MAE"
module_first=1

rl=$lambda
ax=$(echo "1 - $lambda" | bc)

for pl in 96 192 336 720; do
    if [[ "$pl" -eq 96 ]]; then
        n_heads=2
    elif [[ "$pl" -eq 192 || "$pl" -eq 336 ]]; then
        n_heads=8
    else
        n_heads=16
    fi

    read -u 9
    {
        echo "${P}"
        python -u run.py \
            --task_name long_term_forecast \
            --is_training 1 \
            --root_path $DATA_ROOT/ETT-small/ \
            --data_path ETTh1.csv \
            --model_id "ETTh1_96_${pl}" \
            --model ${model_name} \
            --data ETTh1 \
            --features M \
            --seq_len 96 \
            --label_len 48 \
            --pred_len ${pl} \
            --e_layers 1 \
            --d_layers 1 \
            --factor 3 \
            --enc_in 7 \
            --dec_in 7 \
            --c_out 7 \
            --des ${des} \
            --n_heads ${n_heads} \
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
