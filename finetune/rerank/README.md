```python
python finetune/rerank/create_zeshel_instruct_data.py \
    --with_none_case \
    --sft_data_path finetune/rerank/zeshel_instruct_data/bm25/zeshel_train_with_none_case.json \
    --data_path data/zeshel/blink_format/train.jsonl \
    --dict_path dictionaries/zeshel/train/train_dict.jsonl \
    --mention_id_path retrieval_result/zeshel/bm25/train/train_mention_id.json \
    --retrieval_path retrieval_result/zeshel/bm25/train/train.json \
    --candidate_path candidates_for_prompt/zeshel/bm25/zeshel_train_candidates_with_none_case.json
```
