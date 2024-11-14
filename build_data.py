from dotenv import load_dotenv
from datetime import datetime
import collect_data as cld
import json
import os

# Load environment variables
load_dotenv()
hf_token = os.getenv("HF_TOKEN")

config = {
    "total_samples": 80000,
    "dataset_name": "VanRossum",
    "hf_token": hf_token,
    "seed": 18072005
}

def save_file(new_dataset):
    now = datetime.now()
    timestamp = now.strftime("%d%m%y")
    os.makedirs("export", exist_ok=True)
    filename = f"export/{config['dataset_name']}-{round(len(new_dataset) / 1000, 2)}k-{timestamp}.json"
    
    with open(filename, 'w', encoding='utf-8') as f:
        print(f"The {config['dataset_name']} data set saved with a total of {len(new_dataset)} entries.")
        json.dump(new_dataset, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    # Empty data structure
    new_dataset = []
    
    # Dataset lists by format
    instruction_response_sets = [("anubrag/Python-CodeExercises-Evol", 0.2)]
    input_output_sets = [("Kaeyze/computer-science-synthetic-dataset", 0.1)]
    cap_instr_response_sets = [("Nan-Do/instructional_code-search-net-python", 0.1)]
    instr_chosen_response_sets = []
    question_answer_sets = []
    question_response_sets = [("cognitivecomputations/dolphin-coder", 0.2)]
    prompt_chosen_sets = []
    alpaca_output_sets = [("iamtarun/code_instructions_120k_alpaca", 0.2),("Vezora/Tested-22k-Python-Alpaca", 0.2)]
    question_choice_solution_sets = []
    gpt_style_sets = []
    
    source_data = [
        (instruction_response_sets, "instruction_response"),
        (input_output_sets, "input_output"),
        (cap_instr_response_sets, "cap_instruction_response"),
        (question_answer_sets, "question_answer"),
        (question_response_sets, "question_response"),
        (instr_chosen_response_sets, "instr_chosen_resp"),
        (prompt_chosen_sets, "prompt_chosen"),
        (alpaca_output_sets, "alpaca_format"),
        (question_choice_solution_sets, "question_solution"),
        (gpt_style_sets, "gpt-style")
    ]
    
    for dataset_list, format_name in source_data:
        if len(dataset_list) > 0:
            cld.process_datasets(dataset_list, format_name, config, new_dataset)
    
    save_file(new_dataset)