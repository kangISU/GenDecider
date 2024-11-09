import copy
import json
import argparse
import random
import re
import os
from typing import OrderedDict
import torch
import wandb
from tqdm import tqdm
from fastchat.model import load_model, get_conversation_template, add_model_args


INSTRUCT = '''Entity Mention: {}\nEntity Mention Context: {}\n\nBased on the above entity mention and its context, identify the ID of the candidate in the following to which the entity mention refers:{}'''

INSTRUCT_WITH_NONE_CASE = '''Entity Mention: {}\nEntity Mention Context: {}\n\nBased on the above entity mention and its context, identify the ID of the candidate in the following to which the entity mention refers (if none of them, assign the ID as "None"):{}'''

# # Only for in-context learning
# INSTRUCT = '''Entity Mention: {}\nEntity Mention Context: {}\n\nBased on the above entity mention and its context, identify the ID of the candidate in the following to which the entity mention refers:{}\nOnly output the ID in this format {{"ID": ""}}'''

# INSTRUCT_WITH_NONE_CASE = '''Entity Mention: {}\nEntity Mention Context: {}\n\nBased on the above entity mention and its context, identify the ID of the candidate in the following to which the entity mention refers:{}\nOnly output the ID in this format {{"ID": ""}} (if none of them, assign the ID as "None")'''


CANDIDATE_NUM = 10

MAX_INPUT_LENGTH = 4000


def load_json_data(path):
    with open(path, 'r', encoding='utf-8') as fp:
        data = json.load(fp)
    return data


def load_jsonl_data(path):
    data = []
    with open(path, 'r', encoding='utf-8') as fp:
        for line in fp:
            obj = json.loads(line.strip())
            data.append(obj)
    return data


def dump_json_data(data, path):
    with open(path, 'w', encoding='utf-8') as fp:
        json.dump(data, fp, ensure_ascii=True,
                  indent=2, separators=(", ", ": "))


def prepare_intermediate_context_candidates(
        data_path, dict_path, mention_id_path, retrieval_path, candidate_path, start):
    data = load_jsonl_data(data_path)
    mention_id_to_idx = {}
    for i, each in enumerate(data):
        mention_id_to_idx[each["mention_id"]] = i

    entities = load_jsonl_data(dict_path)
    zeshel_id_to_idx = {}
    for i, each in enumerate(entities):
        zeshel_id = each["id"]
        zeshel_id_to_idx[zeshel_id] = i

    mention_ids = load_json_data(mention_id_path)

    gt_with_candidates_list = load_json_data(retrieval_path)

    context_candidates_list = []
    for mention_id, gt_with_candidates in zip(mention_ids, gt_with_candidates_list):
        mention_idx = mention_id_to_idx[mention_id]
        mention_obj = data[mention_idx]
        label_id = mention_obj["label_id"]
        title = mention_obj["label_title"]
        assert label_id == gt_with_candidates["label"]

        if gt_with_candidates["label"] not in gt_with_candidates["context_preds"][:args.topk]:
            label_id = "None"
            title = "None"

        left_offset = 128
        right_offset = 128
        if len(mention_obj["context_left"].split(' ')) < 128:
            right_offset = 256 - len(mention_obj["context_left"].split(' '))
        if len(mention_obj["context_right"].split(' ')) < 128:
            left_offset = 256 - len(mention_obj["context_right"].split(' '))

        context_left = ' '.join(
            mention_obj["context_left"].split(' ')[-left_offset:])
        context_right = ' '.join(
            mention_obj["context_right"].split(' ')[:right_offset])

        mention_context = ' '.join([context_left, "[MENTION_START]",
                                   mention_obj["mention"], "[MENTION_END]", context_right])

        candidates = []
        for candidate_zeshel_id in gt_with_candidates["context_preds"][start: start + CANDIDATE_NUM]:
            candidate_entity = entities[zeshel_id_to_idx[candidate_zeshel_id]]
            assert candidate_zeshel_id == candidate_entity["id"]
            candidate = {
                "zeshel_id": candidate_zeshel_id,
                "title": candidate_entity["title"],
                "entity_description": candidate_entity["text"].strip()
            }
            candidates.append(candidate)

        context_candidates = {
            "mention_id": mention_id,
            "mention": mention_obj["mention"],
            "mention_context": mention_context,
            "zeshel_id": label_id,
            "title": title,
            "candidates": candidates
        }
        context_candidates_list.append(context_candidates)

    dump_json_data(context_candidates_list, candidate_path)
    return context_candidates_list


def prepare_final_context_candidates(
        data_path, dict_path, mention_id_path, retrieval_path, chunk_preds_path, candidate_path):
    data = load_jsonl_data(data_path)
    mention_id_to_idx = {}
    for i, each in enumerate(data):
        mention_id_to_idx[each["mention_id"]] = i

    entities = load_jsonl_data(dict_path)
    zeshel_id_to_idx = {}
    for i, each in enumerate(entities):
        zeshel_id = each["id"]
        zeshel_id_to_idx[zeshel_id] = i

    mention_ids = load_json_data(mention_id_path)
    gt_with_candidates_list = load_json_data(chunk_preds_path)

    ori_gt_with_candidates_list = load_json_data(retrieval_path)

    context_candidates_list = []
    for mention_id, gt_with_candidates, ori_gt_with_candidates in zip(mention_ids, gt_with_candidates_list, ori_gt_with_candidates_list):
        mention_idx = mention_id_to_idx[mention_id]
        mention_obj = data[mention_idx]
        label_id = mention_obj["label_id"]
        title = mention_obj["label_title"]
        assert mention_id == gt_with_candidates[
            "mention_id"], f"{mention_id}, {gt_with_candidates['mention_id']}"
        assert label_id == ori_gt_with_candidates["label"]

        if ori_gt_with_candidates["label"] not in ori_gt_with_candidates["context_preds"][:args.topk]:
            label_id = "None"
            title = "None"

        left_offset = 128
        right_offset = 128
        if len(mention_obj["context_left"].split(' ')) < 128:
            right_offset = 256 - len(mention_obj["context_left"].split(' '))
        if len(mention_obj["context_right"].split(' ')) < 128:
            left_offset = 256 - len(mention_obj["context_right"].split(' '))

        context_left = ' '.join(
            mention_obj["context_left"].split(' ')[-left_offset:])
        context_right = ' '.join(
            mention_obj["context_right"].split(' ')[:right_offset])

        mention_context = ' '.join([context_left, "[MENTION_START]",
                                   mention_obj["mention"], "[MENTION_END]", context_right])

        candidates = []
        for candidate_zeshel_id in gt_with_candidates["context_preds"]:
            if candidate_zeshel_id == "None":
                continue
            if candidate_zeshel_id not in zeshel_id_to_idx:
                continue
            candidate_entity = entities[zeshel_id_to_idx[candidate_zeshel_id]]
            assert candidate_zeshel_id == candidate_entity["id"]
            candidate = {
                "zeshel_id": candidate_zeshel_id,
                "title": candidate_entity["title"],
                "entity_description": candidate_entity["text"].strip()
            }
            candidates.append(candidate)

        context_candidates = {
            "mention_id": mention_id,
            "mention": mention_obj["mention"],
            "mention_context": mention_context,
            "zeshel_id": label_id,
            "title": title,
            "candidates": candidates
        }
        context_candidates_list.append(context_candidates)

    dump_json_data(context_candidates_list, candidate_path)
    return context_candidates_list


def shorten_entity_description(entity_description, max_len):
    entity_description_tokens = entity_description.split(" ")
    entity_description = ' '.join(entity_description_tokens[: max_len])
    return entity_description


def formulate_candidates(candidate_list, max_len):
    candidates = ""
    candidate_template = '\n\nID: {}\nEntity: {}\nEntity Description: {}'
    for candidate_obj in candidate_list:
        entity_description = shorten_entity_description(
            candidate_obj["entity_description"], max_len)
        candidate = candidate_template.format(
            candidate_obj["zeshel_id"], candidate_obj["title"], entity_description)
        candidates += candidate

    return candidates


def compute_metrics(reranked):
    print("\ncomputing metrics...")
    predictions = []
    labels = []
    for each in reranked:
        predictions.append(each["prediction"])
        labels.append(each["ground_truth"])

    total = 0
    label_p = 0
    label_n = 0

    pred_p = 0
    pred_n = 0

    true_p = 0
    false_p_label_p = 0
    false_p_label_n = 0
    true_n = 0
    false_n = 0

    for pred, label in zip(predictions, labels):
        total += 1
        try:
            pred_id = pred["ID"].lower()
            label_id = label["ID"].lower()

            is_label_positive = label_id != "none"

            is_pred_positive = pred_id != "none"

            pred_correct = pred_id == label_id

            if is_label_positive:
                label_p += 1
            if not is_label_positive:
                label_n += 1

            if is_pred_positive:
                pred_p += 1
            if not is_pred_positive:
                pred_n += 1

            if is_label_positive and pred_correct:
                true_p += 1
            elif is_label_positive and is_pred_positive and not pred_correct:
                false_p_label_p += 1
            elif not is_label_positive and is_pred_positive:
                false_p_label_n += 1
            elif not is_label_positive and not is_pred_positive:
                true_n += 1
            elif is_label_positive and not is_pred_positive:
                false_n += 1

        except Exception:
            total -= 1
            continue

    assert label_p == true_p + false_p_label_p + false_n
    assert label_n == true_n + false_p_label_n
    assert total == label_p + label_n

    assert pred_p == true_p + false_p_label_p + false_p_label_n
    assert pred_n == true_n + false_n
    assert total == pred_p + pred_n

    precision = true_p / pred_p if pred_p > 0 else 0
    recall = true_p / label_p if label_p > 0 else 0
    f1 = 2 * (precision * recall) / (precision +
                                     recall) if (precision + recall) > 0 else 0

    print(
        f"total examples: {total}\nground truth positive: {label_p}\nground truth negative: {label_n}\n")

    print(
        f"total examples: {total}\nprediction positive: {pred_p}\npredition negative: {pred_n}\n")

    print(f"true positive: {true_p}\nfalse positive (gt positive): {false_p_label_p}\nfalse positive (gt negative): {false_p_label_n}\ntrue negative: {true_n}\nfalse negative: {false_n}\n")

    print(f"retrieval recall@k: {round(label_p / total, 4) * 100}")

    print(
        f"precision: {precision}\nrecall: {recall}\nf1: {f1}")


@torch.inference_mode()
def rerank(args, chunk_preds, start=0, final=False):
    print("reranking...")
    model, tokenizer = load_model(
        args.model_path,
        args.device,
        args.num_gpus,
        args.max_gpu_memory,
        args.load_8bit,
        args.cpu_offloading,
        debug=args.debug,
    )
    max_length = model.config.max_position_embeddings
    print(f"vicuna max length: {max_length}")

    print("preparing candidates...")

    if final:
        mention_candidates = prepare_final_context_candidates(
            args.data_path, args.dict_path, args.mention_id_path, args.retrieval_path, args.chunk_preds_path, args.candidate_path)
    else:
        mention_candidates = prepare_intermediate_context_candidates(
            args.data_path, args.dict_path, args.mention_id_path, args.retrieval_path, args.candidate_path, start)

    print("done preparing candidates")

    batch_prompts = []
    batch_mention_candidates = []
    reranked = []

    if args.with_none_case:
        instruct = INSTRUCT_WITH_NONE_CASE
    else:
        instruct = INSTRUCT

    shorten_len = 32

    for step, mention_candidate in tqdm(enumerate(mention_candidates)):
        entity_description_max_len = 256

        keep_shorten = True
        while keep_shorten:
            msg = instruct.format(
                mention_candidate["mention"],
                mention_candidate["mention_context"],
                formulate_candidates(mention_candidate["candidates"], entity_description_max_len))

            vicuna_convo = get_conversation_template(args.model_path)
            vicuna_convo.append_message(vicuna_convo.roles[0], msg)
            vicuna_convo.append_message(vicuna_convo.roles[1], None)
            prompt = vicuna_convo.get_prompt()

            inputs = tokenizer([prompt])
            input_length = len(inputs["input_ids"][0])

            if input_length <= MAX_INPUT_LENGTH:
                keep_shorten = False
            else:
                entity_description_max_len -= shorten_len

        batch_prompts.append(prompt)
        batch_mention_candidates.append(mention_candidate)

        # If the batch is full or it's the last mention
        if len(batch_prompts) == args.batch_size or step == len(mention_candidates)-1:

            tokenizer.padding_side = 'left'

            inputs = tokenizer(batch_prompts, return_tensors='pt',
                               return_attention_mask=True, padding='longest')

            inputs = {k: v.to(args.device) for k, v in inputs.items()}

            outputs = model.generate(
                **inputs,
                do_sample=True,
                temperature=args.temperature,
                repetition_penalty=args.repetition_penalty,
                max_new_tokens=args.max_new_tokens,
            )

            for idx, (output_ids, mention_candidate) in enumerate(zip(outputs, batch_mention_candidates)):
                output_ids = output_ids[len(inputs["input_ids"][idx]):]

                output = tokenizer.decode(
                    output_ids, skip_special_tokens=True, spaces_between_special_tokens=False)

                new_mention_candidate = copy.deepcopy(mention_candidate)
                new_mention_candidate["ground_truth"] = {
                    "ID": new_mention_candidate["zeshel_id"],
                    "Title": new_mention_candidate["title"]
                }
                try:
                    if final and len(new_mention_candidate["candidates"]) == 0:
                        new_mention_candidate["prediction"] = {"ID": "None"}
                    else:
                        print(f"output: {output}")
                        output = json.loads(
                            re.search("\{.*\}", output, re.DOTALL).group(0))
                        new_mention_candidate["prediction"] = output
                    reranked.append(new_mention_candidate)

                except:
                    print(f"invalid output: {output}")
                    new_mention_candidate["prediction"] = {"ID": "None"}
                    reranked.append(new_mention_candidate)

                try:
                    pred_id = new_mention_candidate["prediction"]["ID"]
                except:
                    print(f"wrong key, suppose to be ID")
                    pred_id = "None"

                if not final:
                    if new_mention_candidate["mention_id"] not in chunk_preds:
                        chunk_preds[new_mention_candidate["mention_id"]] = {
                            "label": new_mention_candidate["zeshel_id"],
                            "context_preds": [pred_id]
                        }
                    else:
                        chunk_preds[new_mention_candidate["mention_id"]
                                    ]["context_preds"].append(pred_id)

            batch_prompts.clear()
            batch_mention_candidates.clear()

        wandb.log({"processed": len(reranked)})

    print(f'num of reranked: {len(reranked)}')

    dump_json_data(reranked, args.reranked_path)
    compute_metrics(reranked)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_model_args(parser)

    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--debug", action="store_true")

    parser.add_argument("--with_none_case", action="store_true")

    parser.add_argument("--batch_size", type=int, default=4)

    parser.add_argument("--data_path", type=str)
    parser.add_argument("--dict_path", type=str)
    parser.add_argument("--mention_id_path", type=str)
    parser.add_argument("--retrieval_path", type=str)

    parser.add_argument("--topk", type=int, default=64)
    parser.add_argument("--candidate_file_dir", type=str)
    parser.add_argument("--candidate_file_name", type=str)
    parser.add_argument("--chunk_preds_path", type=str)
    parser.add_argument("--reranked_file_dir", type=str)
    parser.add_argument("--reranked_file_name", type=str)

    args = parser.parse_args()

    if not os.path.exists(args.candidate_file_dir):
        os.makedirs(args.candidate_file_dir)

    if not os.path.exists(args.reranked_file_dir):
        os.makedirs(args.reranked_file_dir)

    print(f"dataset: {args.data_path.rsplit('/', 1)[-1].split('.')[0]}")
    print(f"model: {args.model_path}")

    starts = [idx for idx in range(0, args.topk, CANDIDATE_NUM)]

    wandb.init(project="gendecider")

    chunk_preds = OrderedDict()
    for start in starts:
        args.candidate_path = f"{args.candidate_file_dir}/{args.candidate_file_name}_top{str(args.topk)}_{str(start)}.json"
        args.reranked_path = f"{args.reranked_file_dir}/{args.reranked_file_name}_top{str(args.topk)}_{str(start)}.json"
        rerank(args, chunk_preds, start)

    chunk_preds_list = []
    for k, v in chunk_preds.items():
        chunk_preds_list.append({
            "mention_id": k,
            "label": v["label"],
            "context_preds": v["context_preds"]
        })

    dump_json_data(chunk_preds_list, args.chunk_preds_path)

    args.candidate_path = f"{args.candidate_file_dir}/{args.candidate_file_name}_top{str(args.topk)}_final.json"
    args.reranked_path = f"{args.reranked_file_dir}/{args.reranked_file_name}_top{str(args.topk)}_final.json"
    rerank(args, None, final=True)

    wandb.finish()
