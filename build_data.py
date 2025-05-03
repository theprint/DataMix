from dotenv import load_dotenv
import json
import os
from datetime import datetime

import collect_data as cld

# Load hugging face token key from .env file
load_dotenv()
hf_token = os.getenv("HF_TOKEN")

config = {
    "total_samples": 2000,
    "dataset_name": "theprint_mix",
    "hf_token": hf_token,
    "seed": 18072005
}

def save_file(new_dataset):
    now = datetime.now()
    timestamp = now.strftime("%d%m%y")
    os.makedirs("export", exist_ok=True)
    filename = f"export/{config['dataset_name']}-Alpaca-{round(len(new_dataset) / 1000, 2)}k-{timestamp}.json"
    
    with open(filename, 'w', encoding='utf-8') as f:
        print(f"The {config['dataset_name']} data set saved with a total of {len(new_dataset)} entries.")
        json.dump(new_dataset, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    # Empty data structure
    new_dataset = []
    
    # Dataset lists by format
    instruction_response_sets = []
    input_output_sets = []
    cap_instr_response_sets = []
    cap_context_response_sets = []
    cap_human_assistant_sets = []
    instr_chosen_response_sets = []
    instr_demonstration_sets = []
    query_answer_sets = []
    question_answer_sets = [("theprint/homicide-investigation", 0.1),("theprint/gamedev-nocode", 0.1)]
    question_response_sets = []
    problem_answer_sets = []
    problem_description_response_sets = []
    problem_gold_standard_sets = []
    prompt_question_sets = []
    prompt_response_sets = []
    prompt_chosen_sets = []
    alpaca_output_sets = [("theprint/Coach-1.2k", 0.3), ("theprint/Hemispheres-v0.3-Final", 0.2), ("theprint/Hemispheres-v0.3-Combo", 0.2)]
    question_choice_solution_sets = []
    gpt_style_sets = [("theprint/DEI_gpt", 0.1)]
    
    source_data = [
        (instruction_response_sets, "instruction_response"),
        (input_output_sets, "input_output"),
        (cap_instr_response_sets, "cap_instruction_response"),
        (cap_context_response_sets, "cap_context_response"),
        (cap_human_assistant_sets, "cap_human_assistant"),
        (query_answer_sets, "query_answer"),
        (question_answer_sets, "question_answer"),
        (question_response_sets, "question_response"),
        (instr_chosen_response_sets, "instr_chosen_resp"),
        (instr_demonstration_sets, "instr_demonstration"),
        (problem_answer_sets, "problem_answer"),
        (problem_description_response_sets, "problem_description_response"),
        (problem_gold_standard_sets, "problem_gold_standard"),
        (prompt_question_sets, "prompt_question"),
        (prompt_response_sets, "prompt_response"),
        (prompt_chosen_sets, "prompt_chosen"),
        (alpaca_output_sets, "alpaca_format"),
        (question_choice_solution_sets, "question_solution"),
        (gpt_style_sets, "gpt-style")
    ]
    
    for dataset_list, format_name in source_data:
        if len(dataset_list) > 0:
            cld.process_datasets(dataset_list, format_name, config, new_dataset)
    
    save_file(new_dataset)