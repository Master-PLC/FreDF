export CUDA_VISIBLE_DEVICES=$4

# +-----------------------------------------------------------------------------+
# | Processes:                                                                  |
# |  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
# |        ID   ID                                                   Usage      |
# |=============================================================================|
# |    0   N/A  N/A      1763      G   /usr/lib/xorg/Xorg                 46MiB |
# |    0   N/A  N/A    773246      C   long_term_forecast               6575MiB |
# |    0   N/A  N/A    773248      C   long_term_forecast               6563MiB |
# |    0   N/A  N/A    773250      C   long_term_forecast               6595MiB |
# |    0   N/A  N/A    773251      C   long_term_forecast               6663MiB |
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
AMLT_OUTPUT_DIR=$3

CHECKPOINTS=$AMLT_OUTPUT_DIR/checkpoints/
RESULTS=$AMLT_OUTPUT_DIR/results/
TEST_RESULTS=$AMLT_OUTPUT_DIR/test_results/
LOG_PATH=$AMLT_OUTPUT_DIR/result_long_term_forecast.txt

model_name=PatchTST
seed=2023
des='Srh-lambda-lr'
auxi_loss="MAE"
module_first=1

lambda=$1
rl=$lambda
ax=$(echo "1 - $lambda" | bc)
lr=$2
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
            --root_path $DATA_ROOT/traffic/ \
            --data_path traffic.csv \
            --model_id "traffic_96_${pl}" \
            --model ${model_name} \
            --data custom \
            --features M \
            --seq_len 96 \
            --label_len 48 \
            --pred_len ${pl} \
            --e_layers 2 \
            --d_layers 1 \
            --factor 3 \
            --enc_in 862 \
            --dec_in 862 \
            --c_out 862 \
            --d_model 512 \
            --d_ff 512 \
            --top_k 5 \
            --des ${des} \
            --batch_size 4 \
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
wait
exec 9>&-
echo "All finished ..."
