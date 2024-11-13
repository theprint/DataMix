from dotenv import load_dotenv
from datetime import datetime
import collect_data as cld
import json
import os

# Don't forget to set your Huggingface token in the .env file as:
# HF_TOKEN = "WhateverYourTokenValueIs"
load_dotenv() 
hf_token = os.getenv("HF_TOKEN")

config = {
    "total_samples": 20000,  # Set this to your desired dataset size.
    "dataset_name": "MyDataSet",  # Set this to your desired dataset name.
    "hf_token": hf_token,  # Only needed for gated datasets.
    "seed": 18072005  # The seed value makes it so that random values can be recreated exactly.
}

def save_file(setname=config["dataset_name"]):
    now = datetime.now()
    timestamp = now.strftime("%d%m%y")
    filename = f"export/{setname}-{round(len(new_dataset) / 1000, 2)}k-{timestamp}.json"
    with open(filename, 'w', encoding='utf-8') as f:
        print(f"The {config['dataset_name']} data set saved with a total of {len(new_dataset)} entries.")
        json.dump(new_dataset, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    # Empty data structure
    new_dataset = []

    # Data sets with instruction-response format
    instruction_response_sets = [("grimulkan/physical-reasoning", 0.02), ("grimulkan/interpersonal-relational-reasoning", 0.01),
                                ("grimulkan/theory-of-mind", 0.02)]
    # Get data
    cld.get_instruction_response(instruction_response_sets)

    # Data sets with instruction-chosen_response format
    instr_chosen_response_sets = [("argilla/distilabel-math-preference-dpo", 0.1)]
    # Get data
    cld.get_instr_chosen_response(instr_chosen_response_sets)

    # Data sets with capitalized instruction-response format
    cap_instr_response_sets = [("KK04/LogicInference_OA", 0.05)]
    # Get data
    cld.get_cap_instr_resp(cap_instr_response_sets)

    # Data sets with alpaca-style format
    alpaca_output_sets = [("theprint/mindfulness-alpaca",0.11), ("iamtarun/python_code_instructions_18k_alpaca",0.1),
                            ("mlabonne/Evol-Instruct-Python-26k", 0.16), ("garage-bAInd/Open-Platypus", 0.04),
                            ("theprint/MysteryWriter", 0.15), ("totally-not-an-llm/EverythingLM-data-V3", 0.05),
                            ("theprint/gamedev_alpaca", 0.12)]
    # Get data
    cld.get_alpaca_response(alpaca_output_sets)

    # Data sets with question-solution format
    science_qa_sets = [("tasksource/ScienceQA_text_only", 0.07)]
    # Get data
    cld.get_question_solution(science_qa_sets)

    save_file()