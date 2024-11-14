from datasets import load_dataset
from dataclasses import dataclass
import yaml
import random
from typing import Dict, Any

@dataclass
class DatasetFormat:
    instruction_key: str
    output_key: str
    input_key: str = None

def get_samples(total_samples: int, weight: float) -> int:
    return round(total_samples * weight)

def join_choices(entry):
    return "\n".join(f" - {c}" for c in entry["choices"])

def process_dataset(dataset_info, format_config, config, new_dataset):
    dataset_name, weight = dataset_info
    ds = load_dataset(dataset_name, split="train", token=config["hf_token"])
    ds = ds.shuffle(seed=config["seed"])
    samples_count = get_samples(config["total_samples"], weight)
    
    print(f"DATA SET: {dataset_name.upper()} | LOADING {samples_count} of {len(ds)} ENTRIES")
    
    for d in range(min(samples_count, len(ds))):
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
        new_dataset.append(qa_set)

def process_gpt_conversations(dataset_info, config, new_dataset):
    dataset_name, weight = dataset_info
    ds = load_dataset(dataset_name, split="train", token=config["hf_token"])
    ds = ds.shuffle(seed=config["seed"])
    samples_count = get_samples(config["total_samples"], weight)
    
    print(f"DATA SET: {dataset_name.upper()} | LOADING {samples_count} of {len(ds)} ENTRIES")
    
    for d in range(min(samples_count, len(ds))):
        entry = ds[d]
        qa_set = {
            "source": dataset_name,
            "instruction": entry["human"],
            "input": "",
            "output": entry["assistant"]
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