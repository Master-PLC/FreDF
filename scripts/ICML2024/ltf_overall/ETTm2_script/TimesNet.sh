export CUDA_VISIBLE_DEVICES=0

# +-----------------------------------------------------------------------------+
# | Processes:                                                                  |
# |  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
# |        ID   ID                                                   Usage      |
# |=============================================================================|
# |    0   N/A  N/A      1763      G   /usr/lib/xorg/Xorg                 46MiB |
# |    0   N/A  N/A    728072      C   long_term_forecast               1991MiB |
# |    0   N/A  N/A    728074      C   long_term_forecast               2089MiB |
# |    0   N/A  N/A    728076      C   long_term_forecast               2221MiB |
# |    0   N/A  N/A    728077      C   long_term_forecast               2489MiB |
# +-----------------------------------------------------------------------------+

set -e

PROC_NUM=1
FIFO_FILE="/tmp/$$.fifo"
mkfifo ${FIFO_FILE}
exec 9<>${FIFO_FILE}
for process_num in $(seq ${PROC_NUM}); do
    echo "$(date +%F\ %T) Processor-${process_num} Start " >&9
done

DATA_ROOT=./dataset
OUTPUT_DIR=./exp_results/ltf_overall

model_name=TimesNet
seed=2023
des='LTFAll'

auxi_loss="MAE"
module_first=1
lambda=1.0
rl=$lambda
ax=$(echo "1 - $lambda" | bc)
lr=0.0001
auxi_mode='rfft'

JOB_DIR=$OUTPUT_DIR/${model_name}_ETTm2_${lambda}
mkdir -p $JOB_DIR

CHECKPOINTS=$JOB_DIR/checkpoints/
RESULTS=$JOB_DIR/results/
TEST_RESULTS=$JOB_DIR/test_results/
LOG_PATH=$JOB_DIR/result_long_term_forecast.txt

pl_list=(96 192 336 720)

for pl in ${pl_list[@]}; do
    if [[ "$pl" -eq 192 ]]; then
        d_model=32
        train_epochs=1
    elif [[ "$pl" -eq 720 ]]; then
        d_model=16
        train_epochs=1
    else
        d_model=32
        train_epochs=10
    fi
    read -u 9
    {
        echo "${P}"
        python -u run.py \
            --task_name long_term_forecast \
            --is_training 1 \
            --root_path $DATA_ROOT/ETT-small/ \
            --data_path ETTm2.csv \
            --model_id "ETTm2_96_${pl}" \
            --model ${model_name} \
            --data ETTm2 \
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
            --d_model $d_model \
            --d_ff 32 \
            --top_k 5 \
            --des ${des} \
            --itr 1 \
            --train_epochs $train_epochs \
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
