model_name=CNN
train_epochs=100
learning_rate=0.02

master_port=1234
num_process=1
batch_size=64
num_entries=40000
num_lags=64

comment='CNN-gbpcad'

accelerate launch --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port "../experiments/run_main.py" \
    --root_path ../data/ \
    --data_path gbpcad_one_hour_202311210827.csv \
    --model_id GBPCAD \
    --model $model_name \
    --data gbpcad \
    --num_lags $num_lags \
    --target "close" \
    --batch_size $batch_size \
    --learning_rate $learning_rate \
    --train_epochs $train_epochs \
    --model_comment $comment \
    --num_entries $num_entries \
    --lradj 'type3' \
    --pct_start 0.2 \
    --pred_len 10 \
    --seq_step 7
