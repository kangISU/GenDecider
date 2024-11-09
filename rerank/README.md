```python
python rerank/rerank_zeshel_topk.py \
    --model-path models/vicuna-7b-v1.5-peft-rerank-zeshel-with-none-case-bs-1/checkpoint-50000 \
    --data_path data/zeshel/blink_format/forgotten_realms.jsonl \
    --dict_path dictionaries/zeshel/test/forgotten_realms_dict.jsonl \
    --mention_id_path retrieval_result/zeshel/bm25/test/forgotten_realms_mention_id.json \
    --retrieval_path retrieval_result/zeshel/bm25/test/forgotten_realms.json \
    --candidate_file_dir candidates_for_prompt/zeshel/bm25/top64 \
    --candidate_file_name forgotten_realms_candidates_ckp50000 \
    --chunk_preds_path rerank/reranked/zeshel/top64/forgotten_realms_chunk_preds_ckp50000.json \
    --reranked_file_dir rerank/reranked/zeshel/top64 \
    --reranked_file_name forgotten_realms_reranked_ckp50000
```