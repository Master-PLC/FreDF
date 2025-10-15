export CUDA_VISIBLE_DEVICES=$4

set -e

PROC_NUM=2
FIFO_FILE="/tmp/$$.fifo"
mkfifo ${FIFO_FILE}
exec 9<>${FIFO_FILE}
for process_num in $(seq ${PROC_NUM}); do
    echo "$(date +%F\ %T) Processor-${process_num} Start " >&9
done

DATA_ROOT=./dataset
OUTPUT_DIR=./exp_results/gene_ortho

model_name=iTransformer
seed=2023
des='GeneOrtho'

auxi_loss="MAE"
module_first=1
lr=0.0001

lbd_list=(0.0 0.2 0.4 0.6 0.8)
auxi_mode_list=("legendre" "chebyshev" "laguerre")
pl_list=(96 192 336 720)

for lambda in ${lbd_list[@]}; do
    for auxi_mode in ${auxi_mode_list[@]}; do
        JOB_DIR=$OUTPUT_DIR/${model_name}_ETTh1_${lambda}_${auxi_mode}
        mkdir -p $JOB_DIR

        CHECKPOINTS=$JOB_DIR/checkpoints/
        RESULTS=$JOB_DIR/results/
        TEST_RESULTS=$JOB_DIR/test_results/
        LOG_PATH=$JOB_DIR/result_long_term_forecast.txt

        rl=$lambda
        ax=$(echo "1 - $lambda" | bc)

        for pl in ${pl_list[@]}; do
            if [[ $pl == 96 ]]; then
                degree_list=(2 25 50 75 95)
            elif [[ $pl == 192 ]]; then
                degree_list=(2 50 100 150 191)
            elif [[ $pl == 336 ]]; then
                degree_list=(2 75 150 225 335)
            else
                degree_list=(2 175 350 525 719)
            fi

            for leg_degree in ${degree_list[@]}; do
                read -u 9
                {
                    echo "${P}"
                    python -u run.py \
                        --task_name long_term_forecast \
                        --is_training 1 \
                        --root_path $DATA_ROOT/ETT-small/ \
                        --data_path ETTh1.csv \
                        --model_id "ETTh1_96_${pl}_${leg_degree}" \
                        --model ${model_name} \
                        --data ETTh1 \
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
                        --auxi_mode ${auxi_mode} \
                        --leg_degree ${leg_degree}
                    echo ${P} >&9
                } &
            done
        done
    done
done
wait
exec 9>&-
echo "All finished ..."
