model_name=CNN
train_epochs=100
learning_rate=0.02

master_port=1234
num_process=1
batch_size=64
num_entries=10000
num_lags=20

comment='CNN-aapl'

accelerate launch --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port "../experiments/run_main.py" \
    --root_path ../data/ \
    --data_path AAPL.csv \
    --model_id AAPL \
    --model $model_name \
    --data aapl \
    --num_lags $num_lags \
    --target "Close" \
    --batch_size $batch_size \
    --learning_rate $learning_rate \
    --train_epochs $train_epochs \
    --model_comment $comment \
    --num_entries $num_entries \
    --lradj 'type3' \
    --pct_start 0.2 \
    --pred_len 40 \
    --seq_step 1
