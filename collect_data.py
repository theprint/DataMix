from datasets import load_dataset
from dataclasses import dataclass
import yaml
import random
from typing import Dict, Any

MAX_INSTRUCTION_LENGTH = 4000
MAX_OUTPUT_LENGTH = 4000

@dataclass
class DatasetFormat:
    instruction_key: str
    output_key: str
    input_key: str = None

def get_samples(total_samples: int, weight: float) -> int:
    return round(total_samples * weight)

def join_choices(entry):
    return "\n".join(f" - {c}" for c in entry["choices"])

def process_dataset(dataset_info, format_config, config, new_dataset, split="train"):
    dataset_name, weight = dataset_info
    ds = load_dataset(dataset_name, split=split, token=config["hf_token"])
    ds = ds.shuffle(seed=config["seed"])
    samples_count = min(get_samples(config["total_samples"], weight), len(ds))
    
    print(f"DATA SET: {dataset_name.upper()} | LOADING {samples_count} of {len(ds)} ENTRIES")
    
    for d in range(samples_count):
        entry = ds[d]
        input_text = ""
        
        # Special handling for question_solution format
        if format_config.input_key == "choices":
            input_text = join_choices(entry)
        elif format_config.input_key:
            input_text = entry[format_config.input_key]
            
        qa_set = {
            "source": dataset_name,
            "instruction": entry[format_config.instruction_key],
            "input": input_text,
            "output": entry[format_config.output_key]
        }
        if len(qa_set["instruction"]) > 0 and len(qa_set["instruction"]) < MAX_INSTRUCTION_LENGTH and len(qa_set["output"]) > 0 and len(qa_set["output"]) < MAX_OUTPUT_LENGTH:
            new_dataset.append(qa_set)

def process_gpt_conversations(dataset_info, config, new_dataset):
    dataset_name, weight = dataset_info
    ds = load_dataset(dataset_name, split="train", token=config["hf_token"])
    ds = ds.shuffle(seed=config["seed"])
    samples_count = get_samples(config["total_samples"], weight)
    
    print(f"DATA SET: {dataset_name.upper()} | LOADING {samples_count} of {len(ds)} ENTRIES")
    
    for d in range(min(samples_count, len(ds))):
        entry = ds[d]
        # print(f"DEBUG: {entry}")
        if "conversations" in entry:
            sub_entry = entry["conversations"]
            print(f"DEBUG: {sub_entry}")
            user_in = None
            ai_out = None
            
            for part in sub_entry:
                print(f"DEBUG: {part}")
                if "from" and "value" in part:
                    if part["from"] == "user":
                        user_in = part["value"]
                    elif part["from"] == "gpt":
                        ai_out = part["value"]
                else:
                    print("Error: Unsupported GPT data formatting.")
                    break
        
        qa_set = {
            "source": dataset_name,
            "instruction": user_in,
            "input": "",
            "output": ai_out
        }
        new_dataset.append(qa_set)

def process_datasets(dataset_list, format_name, config, new_dataset):
    with open("formats.yaml", "r") as f:
        formats_data = yaml.safe_load(f)["formats"]
    
    if format_name == "gpt-style":
        for dataset_info in dataset_list:
            process_gpt_conversations(dataset_info, config, new_dataset)
    else:
        format_config = DatasetFormat(**formats_data[format_name])
        for dataset_info in dataset_list:
            process_dataset(dataset_info, format_config, config, new_dataset)