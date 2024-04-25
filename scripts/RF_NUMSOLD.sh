model_name=RF

master_port=1234
num_process=1
num_entries=40000 # TODO num_entries

comment='SVM-testing'

accelerate launch --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port "../experiments/run_main.py" \
    --root_path ../data/ \
    --data_path numsold.csv \
    --model_id NUMSOLD \
    --model $model_name \
    --data numsold \
    --num_lags 20 \
    --target "number_sold" \
    --n_estimators 100 \
    --max_features 8 \
    --criterion "log_loss" \
    --model_comment $comment
