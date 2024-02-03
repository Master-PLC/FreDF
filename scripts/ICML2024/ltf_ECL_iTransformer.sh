export CUDA_VISIBLE_DEVICES=0

DATA_ROOT=./dataset

CHECKPOINTS=$AMLT_OUTPUT_DIR/checkpoints/
RESULTS=$AMLT_OUTPUT_DIR/results/
TEST_RESULTS=$AMLT_OUTPUT_DIR/test_results/
LOG_PATH=$AMLT_OUTPUT_DIR/result_long_term_forecast.txt

model_name=iTransformer
ax=0
rl=1
seed=2021
des='Exp'

# python -u run.py \
#     --task_name long_term_forecast \
#     --is_training 1 \
#     --root_path $DATA_ROOT/electricity/ \
#     --data_path electricity.csv \
#     --model_id ECL_96_96 \
#     --model $model_name \
#     --data custom \
#     --features M \
#     --seq_len 96 \
#     --label_len 48 \
#     --pred_len 96 \
#     --e_layers 3 \
#     --d_layers 1 \
#     --factor 3 \
#     --enc_in 321 \
#     --dec_in 321 \
#     --c_out 321 \
#     --des ${des} \
#     --d_model 512 \
#     --d_ff 512 \
#     --batch_size 16 \
#     --learning_rate 0.0005 \
#     --itr 1 \
#     --auxi_lambda ${ax} \
#     --rec_lambda ${rl} \
#     --fix_seed ${seed} &

# python -u run.py \
#     --task_name long_term_forecast \
#     --is_training 1 \
#     --root_path $DATA_ROOT/electricity/ \
#     --data_path electricity.csv \
#     --model_id ECL_96_192 \
#     --model $model_name \
#     --data custom \
#     --features M \
#     --seq_len 96 \
#     --label_len 48 \
#     --pred_len 192 \
#     --e_layers 3 \
#     --d_layers 1 \
#     --factor 3 \
#     --enc_in 321 \
#     --dec_in 321 \
#     --c_out 321 \
#     --des ${des} \
#     --d_model 512 \
#     --d_ff 512 \
#     --batch_size 16 \
#     --learning_rate 0.0005 \
#     --itr 1 \
#     --auxi_lambda ${ax} \
#     --rec_lambda ${rl} \
#     --fix_seed ${seed} &


# python -u run.py \
#     --task_name long_term_forecast \
#     --is_training 1 \
#     --root_path $DATA_ROOT/electricity/ \
#     --data_path electricity.csv \
#     --model_id ECL_96_336 \
#     --model $model_name \
#     --data custom \
#     --features M \
#     --seq_len 96 \
#     --label_len 48 \
#     --pred_len 336 \
#     --e_layers 3 \
#     --d_layers 1 \
#     --factor 3 \
#     --enc_in 321 \
#     --dec_in 321 \
#     --c_out 321 \
#     --des ${des} \
#     --d_model 512 \
#     --d_ff 512 \
#     --batch_size 16 \
#     --learning_rate 0.0005 \
#     --itr 1 \
#     --auxi_lambda ${ax} \
#     --rec_lambda ${rl} \
#     --fix_seed ${seed} &


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
    --learning_rate 0.0005 \
    --itr 1 \
    --auxi_lambda ${ax} \
    --rec_lambda ${rl} \
    --fix_seed ${seed} \
    --checkpoints $CHECKPOINTS \
    --results $RESULTS \
    --test_results $TEST_RESULTS \
    --log_path $LOG_PATH &

wait