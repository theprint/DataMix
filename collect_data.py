import random
from datasets import load_dataset
from build_data import new_dataset, config

def get_samples(weight: float) -> int:
    my_samples = round(config["total_samples"] * weight)
    return my_samples


# instruction - response formatted data
def get_instruction_response(setlist):
    for dataset, weight in setlist:
        ds = load_dataset(dataset, split="train", token=config["hf_token"])
        ds = ds.shuffle(seed=config["seed"])
        print(f"DATA SET: {dataset.upper()} | LOADING {get_samples(weight)} of {len(ds)} ENTRIES")
        d = 0
        while d < len(ds) and d < get_samples(weight):
            # print(f"ENTRY {d+1}:\n\t{ds[d]}")
            qa_set = {
                "source": dataset,
                "instruction": ds[d]["instruction"],
                "input": "",
                "output": ds[d]["response"]
            }
            new_dataset.append(qa_set)
            d += 1


# prompt - chosen format
def get_prompt_chosen(setlist):
    for dataset, weight in setlist:
        ds = load_dataset(dataset, split="train", token=config["hf_token"])
        ds = ds.shuffle(seed=config["seed"])
        print(f"DATA SET: {dataset.upper()} | LOADING {get_samples(weight)} of {len(ds)} ENTRIES")
        d = 0
        while d < len(ds) and d < get_samples(weight):
            # print(f"ENTRY {d+1}:\n\t{ds[d]}")
            qa_set = {
                "source": dataset,
                "instruction": ds[d]["prompt"],
                "input": "",
                "output": ds[d]["chosen"]
            }
            new_dataset.append(qa_set)
            d += 1


# input - output format
def get_input_output(setlist):
    for dataset, weight in setlist:
        ds = load_dataset(dataset, split="train", token=config["hf_token"])
        ds = ds.shuffle(seed=config["seed"])
        print(f"DATA SET: {dataset.upper()} | LOADING {get_samples(weight)} of {len(ds)} ENTRIES")
        d = 0
        while d < len(ds) and d < get_samples(weight):
            # print(f"ENTRY {d+1}:\n\t{ds[d]}")
            qa_set = {
                "source": dataset,
                "instruction": ds[d]["input"],
                "input": "",
                "output": ds[d]["output"]
            }
            new_dataset.append(qa_set)
            d += 1


# prompt-output format
def get_prompt_output(setlist):
    for dataset, weight in setlist:
        print(f"DATA SET: {dataset.upper()} | LOADING {get_samples(weight)} of {len(ds)} ENTRIES")
        ds = load_dataset(dataset, split="train", token=config["hf_token"])
        ds = ds.shuffle(seed=config["seed"])
        d = 0
        while d < len(ds) and d < get_samples(weight):
            qa_set = {
                "source": dataset,
                "instruction": ds[d]["prompt"],
                "input": "",
                "output": ds[d]["output"]
            }
            new_dataset.append(qa_set)
            d += 1


# Instruction and chosen_response.
def get_instr_chosen_response(setlist):
    for dataset, weight in setlist:
        ds = load_dataset(dataset, split="train", token=config["hf_token"])
        ds = ds.shuffle(seed=config["seed"])
        print(f"DATA SET: {dataset.upper()} | LOADING {get_samples(weight)} of {len(ds)} ENTRIES")
        d = 0
        while d < len(ds) and d < get_samples(weight):
            # print(f"ENTRY {d+1}:\n\t{ds[d]}")
            qa_set = {
                "source": dataset,
                "instruction": ds[d]["instruction"],
                "input": "",
                "output": ds[d]["chosen_response"]
            }
            new_dataset.append(qa_set)
            d += 1


# Capitalized instruction - response formatted data
def get_cap_instr_resp(setlist):
    for dataset, weight in setlist:
        ds = load_dataset(dataset, split="train", token=config["hf_token"])
        ds = ds.shuffle(seed=config["seed"])
        print(f"DATA SET: {dataset.upper()} | LOADING {get_samples(weight)} of {len(ds)} ENTRIES")
        d = 0
        while d < len(ds) and d < get_samples(weight):
            # print(f"ENTRY {d+1}:\n\t{ds[d]}")
            qa_set = {
                "source": dataset,
                "instruction": ds[d]["INSTRUCTION"],
                "input": "",
                "output": ds[d]["RESPONSE"]
            }
            new_dataset.append(qa_set)
            d += 1


# Alpaca-style (instruction, input, output) formatted data / input is optional
def get_alpaca_response(setlist):
    for dataset, weight in setlist:
        ds = load_dataset(dataset, split="train", token=config["hf_token"])
        ds = ds.shuffle(seed=config["seed"])
        print(f"DATA SET: {dataset.upper()} | LOADING {get_samples(weight)} of {len(ds)} ENTRIES")
        d = 0
        while d < len(ds) and d < get_samples(weight):
            qa_set = {
                "source": dataset,
                "instruction": ds[d]["instruction"],
                "input": "",
                "output": ds[d]["output"]
            }
            if "input" in ds[d]:
                qa_set["input"] = ds[d]["input"]
            new_dataset.append(qa_set)
            d += 1


# question+choices-solution formatted data
def get_question_solution(setlist):
    for dataset, weight in setlist:
        print(f"DATA SET: {dataset.upper()} | LOADING {get_samples(weight)} of {len(ds)} ENTRIES")
        ds = load_dataset(dataset, split="train", token=config["hf_token"])
        ds = ds.shuffle(seed=config["seed"])
        d = 0
        while d < len(ds) and d < get_samples(weight):
            choices = ""
            for c in ds[d]["choices"]:
                choices += f"\n - {c}"
            qa_set = {
                "source": dataset,
                "instruction": ds[d]["question"],
                "input": choices,
                "output": ds[d]["solution"]
            }
            new_dataset.append(qa_set)
            d += 1


# question-best answer format
def get_question_bestansw(setlist):
    for dataset, weight in setlist:
        ds = load_dataset('truthfulqa/truthful_qa', 'generation', split="validation")
        ds = ds.shuffle(seed=config["seed"])
        print(f"DATA SET: {dataset.upper()} | LOADING {get_samples(weight)} of {len(ds)} ENTRIES")
        d = 0
        while d < len(ds) and d < get_samples(weight):
            qa_set = {
                "source": dataset,
                "instruction": ds[d]["question"],
                "input": "",
                "output": ds[d]["best_answer"]
            }
            if "input" in ds[d]:
                qa_set["input"] = ds[d]["input"]
            new_dataset.append(qa_set)
            d += 1


# problem-solution formatted data
def get_problem_solution(setlist):
    for dataset, weight in setlist:
        print(f"DATA SET: {dataset.upper()} | LOADING {get_samples(weight)} of {len(ds)} ENTRIES")
        ds = load_dataset(dataset, split="train", token=config["hf_token"])
        ds = ds.shuffle(seed=config["seed"])
        d = 0
        while d < len(ds) and d < get_samples(weight):
            qa_set = {
                "source": dataset,
                "instruction": ds[d]["problem"],
                "input": "",
                "output": ds[d]["solution"]
            }
            new_dataset.append(qa_set)
            d += 1


# user_input-model_output formatted data
def get_user_input_model_output(setlist):
    for dataset, weight in setlist:
        print(f"DATA SET: {dataset.upper()} | LOADING {get_samples(weight)} of {len(ds)} ENTRIES")
        ds = load_dataset(dataset, split="train", token=config["hf_token"])
        ds = ds.shuffle(seed=config["seed"])
        d = 0
        while d < len(ds) and d < get_samples(weight):
            qa_set = {
                "source": dataset,
                "instruction": ds[d]["user_input"],
                "input": "",
                "output": ds[d]["model_output"]
            }
            new_dataset.append(qa_set)
            d += 1


# info-summary format
def get_info_summary(setlist):
    for dataset, weight in setlist:
        print(f"DATA SET: {dataset.upper()} | LOADING {get_samples(weight)} of {len(ds)} ENTRIES")
        ds = load_dataset('openai/summarize_from_feedback', 'axis', split="validation")
        ds = ds.shuffle(seed=config["seed"])
        d = 0
        while d < len(ds) and d < get_samples(weight):
            entry = ds[d]  # json.loads(ds[d])
            info = entry["info"]["post"]
            summary = entry["summary"]["text"]
            instruction = random.choice(
                ["Summarize the following:", "Give me a summary of this:", "Please summarize this:",
                 "Create a summary of the following:"])
            qa_set = {
                "source": dataset,
                "instruction": instruction,
                "input": info,
                "output": summary
            }
            new_dataset.append(qa_set)
            d += 1


# GPT-style (conversations) formatted-data
def get_gpt_conversations(setlist):
    for dataset in setlist:
        ds = load_dataset(dataset[0], split="train", token=config["hf_token"])
        print(f"DATA SET: {dataset[0].upper()} | LOADING {get_samples(dataset[1])} of {len(ds)} ENTRIES")
        d = 0
        while d < get_samples(dataset[1]):
            for turn in ds["conversations"]:
                user_turn = turn[0]
                model_turn = turn[1]
                instruction = user_turn["value"]
                output = model_turn["value"]
                qa_set = {
                    "source": dataset[0],
                    "instruction": instruction,
                    "input": "",
                    "output": output
                }
                new_dataset.append(qa_set)
                d += 1
                if d == get_samples(dataset[1]):
                    break