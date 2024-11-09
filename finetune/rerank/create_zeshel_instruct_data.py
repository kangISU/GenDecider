import json
import argparse
import random
from tqdm import tqdm
from transformers import AutoTokenizer
from fastchat.model import get_conversation_template


INSTRUCT = '''Entity Mention: {}\nEntity Mention Context: {}\n\nBased on the above entity mention and its context, identify the ID of the candidate in the following to which the entity mention refers:{}'''

INSTRUCT_WITH_NONE_CASE = '''Entity Mention: {}\nEntity Mention Context: {}\n\nBased on the above entity mention and its context, identify the ID of the candidate in the following to which the entity mention refers (if none of them, assign the ID as "None"):{}'''


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


def prepare_context_candidates(
        data_path, dict_path, mention_id_path, retrieval_path, candidate_path, with_none_case=False):
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

    none_case_num = 0
    context_candidates_list = []
    for mention_id, gt_with_candidates in zip(mention_ids, gt_with_candidates_list):
        mention_idx = mention_id_to_idx[mention_id]
        mention_obj = data[mention_idx]
        label_id = mention_obj["label_id"]
        assert label_id == gt_with_candidates["label"]

        if gt_with_candidates["label"] not in gt_with_candidates["context_preds"][:CANDIDATE_NUM]:
            if with_none_case:
                label_id = "None"
                none_case_num += 1
            else:
                continue

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

        mention_context = ' '.join(
            [context_left, mention_obj["mention"], context_right])

        candidates = []
        for candidate_zeshel_id in gt_with_candidates["context_preds"][:CANDIDATE_NUM]:
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
            "candidates": candidates
        }
        context_candidates_list.append(context_candidates)

    print(f"num of none case: {none_case_num}")

    dump_json_data(context_candidates_list, candidate_path)
    return context_candidates_list


def shorten_entity_description(entity_description, max_len):
    entity_description_tokens = entity_description.split(" ")
    entity_description = ' '.join(entity_description_tokens[: max_len])
    return entity_description


def formulate_candidates(candidate_list, max_len):
    candidates = ""
    candidate_template = '\n\nID: {}\nEntity: {}\nEntity Description: {}'
    random.shuffle(candidate_list)
    for i, candidate_obj in enumerate(candidate_list):
        entity_description = shorten_entity_description(
            candidate_obj["entity_description"], max_len)
        candidate = candidate_template.format(
            candidate_obj["zeshel_id"], candidate_obj["title"], entity_description)
        candidates += candidate

    return candidates


def is_length_valid(model_path, human_value, gpt_value):
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    vicuna_convo = get_conversation_template(model_path)
    vicuna_convo.append_message(vicuna_convo.roles[0], human_value)
    vicuna_convo.append_message(vicuna_convo.roles[1], gpt_value)
    prompt = vicuna_convo.get_prompt()

    inputs = tokenizer([prompt])
    input_length = len(inputs["input_ids"][0])

    if random.randint(1, 100) == 1:
        print(f"vicuna input length + output length = {input_length}")

    if input_length > MAX_INPUT_LENGTH:
        return False

    return True


def create_sft_data(context_candidates_list, sft_data_path, model_path, with_none_case=False):
    if with_none_case:
        instruct = INSTRUCT_WITH_NONE_CASE
    else:
        instruct = INSTRUCT
    sft_data = []
    shorten_len = 32
    for each in tqdm(context_candidates_list):
        exp = {"mention_id": each["mention_id"],
               "mention": each["mention"]}

        entity_description_max_len = 256
        keep_shorten = True
        while keep_shorten:
            human_value = instruct.format(
                each["mention"],
                each["mention_context"],
                formulate_candidates(
                    each["candidates"], entity_description_max_len)
            )

            ground_truth = {"ID": each["zeshel_id"]}
            gpt_value = json.dumps(ground_truth)

            if is_length_valid(model_path, human_value, gpt_value):
                keep_shorten = False
            else:
                entity_description_max_len -= shorten_len

        human_turn = {"from": "human", "value": human_value}
        gpt_turn = {"from": "gpt", "value": gpt_value}
        exp["conversations"] = [human_turn, gpt_turn]
        sft_data.append(exp)

    dump_json_data(sft_data, sft_data_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--with_none_case", action="store_true")
    parser.add_argument("--sft_data_path", type=str)    # must provide

    parser.add_argument("--model_path", type=str)   # must provide

    parser.add_argument("--data_path", type=str)    # must provide

    parser.add_argument("--dict_path", type=str)    # must provide

    parser.add_argument("--mention_id_path", type=str)   # must provide

    parser.add_argument("--retrieval_path", type=str)  # must provide

    parser.add_argument("--candidate_path", type=str)   # must provide

    args = parser.parse_args()

    context_candidates_list = prepare_context_candidates(
        args.data_path, args.dict_path, args.mention_id_path, args.retrieval_path, args.candidate_path, args.with_none_case
    )

    create_sft_data(context_candidates_list,
                    args.sft_data_path, args.model_path, args.with_none_case)
