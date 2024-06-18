model_name=CNN
train_epochs=100
learning_rate=0.02

master_port=1234
num_process=1
batch_size=32
num_entries=10000
num_lags=20

comment='CNN-house_sales'

accelerate launch --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port "../experiments/run_main.py" \
    --root_path ../data/ \
    --data_path house_sales.csv \
    --model_id HOUSE \
    --model $model_name \
    --data house_sales \
    --num_lags $num_lags \
    --target "price" \
    --batch_size $batch_size \
    --learning_rate $learning_rate \
    --train_epochs $train_epochs \
    --model_comment $comment \
    --num_entries $num_entries \
    --lradj 'type3' \
    --pct_start 0.3
