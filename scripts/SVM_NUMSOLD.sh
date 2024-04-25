model_name=SVM

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
    --num_lags 30 \
    --target "number_sold" \
    --kernel "poly" \
    --gamma "scale" \
    --C 1.0 \
    --model_comment $comment
