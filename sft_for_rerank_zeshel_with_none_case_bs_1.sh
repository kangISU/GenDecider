cd /work/LAS/qli-lab/kangzhou/envs
source vicuna/bin/activate

cd /work/LAS/qli-lab/kangzhou/GenDecider

log_file="sft_for_rerank_zeshel_with_none_case_bs_1_log.txt"
error_log_file="sft_for_rerank_zeshel_with_none_case_bs_1_error_log.txt"

deepspeed fastchat/train/train_lora.py \
    --model_name_or_path models/vicuna-7b-v1.5 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --data_path finetune/rerank/zeshel_instruct_data/bm25/zeshel_train_with_none_case.json \
    --fp16 True \
    --output_dir models/vicuna-7b-v1.5-peft-rerank-zeshel-with-none-case-bs-1 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing True \
    --evaluation_strategy "steps" \
    --eval_steps 10000  \
    --save_strategy "steps" \
    --save_steps 10000 \
    --save_total_limit 100 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_strategy "steps" \
    --logging_steps 1 \
    --model_max_length 4096 \
    --q_lora True \
    --deepspeed finetune/deepspeed_config_s2.json >> "$log_file" 2>> "$error_log_file"