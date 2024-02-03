export CUDA_VISIBLE_DEVICES=0

DATA_ROOT=./dataset
ax=$1

CHECKPOINTS=$AMLT_OUTPUT_DIR/checkpoints/
RESULTS=$AMLT_OUTPUT_DIR/results/
TEST_RESULTS=$AMLT_OUTPUT_DIR/test_results/
LOG_PATH=$AMLT_OUTPUT_DIR/result_long_term_forecast.txt

model_name=DLinear
rl=0
seed=2023
des='Srh-rl0'
auxi_loss="MAE"
module_first=1


python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path $DATA_ROOT/electricity/ \
    --data_path electricity.csv \
    --model_id ECL_96_96 \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 96 \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 321 \
    --dec_in 321 \
    --c_out 321 \
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
    --log_path $LOG_PATH &


python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path $DATA_ROOT/electricity/ \
    --data_path electricity.csv \
    --model_id ECL_96_192 \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 192 \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 321 \
    --dec_in 321 \
    --c_out 321 \
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
    --log_path $LOG_PATH &


python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path $DATA_ROOT/electricity/ \
    --data_path electricity.csv \
    --model_id ECL_96_336 \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 336 \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 321 \
    --dec_in 321 \
    --c_out 321 \
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
    --log_path $LOG_PATH &


python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path $DATA_ROOT/electricity/ \
    --data_path electricity.csv \
    --model_id ECL_96_720 \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 720 \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 321 \
    --dec_in 321 \
    --c_out 321 \
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
    --log_path $LOG_PATH &

wait