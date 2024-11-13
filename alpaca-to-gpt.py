import json
from datetime import datetime

SYSTEM_PROMPT = "You are an AI assistant that always delivers an accurate and objective response."

def save_file(data_name, dataset):
    now = datetime.now()
    timestamp = now.strftime("%d%m%y")
    filename = f"{data_name}-{round(len(dataset['conversations'])/1000,2)}k-{timestamp}.json"
    with open(filename, 'w', encoding='utf-8') as f:
        print(f"The {data_name} data set was saved with {len(dataset['conversations'])} entries.")
        json.dump(dataset, f, ensure_ascii=False, indent=2)

def read_alpaca(filename):
    """
    Reads a JSON file and returns its contents as a Python data object.

    Args:
        filename (str): The path to the JSON file to be read.

    Returns:
        dict or list: The parsed JSON data, which can be a dictionary, list, or any other JSON type.
    """
    try:
        with open(filename, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"The file {filename} was not found.")
        return None
    except json.JSONDecodeError as e:
        print(f"An error occurred while parsing the JSON: {e}")
        return None



def create_gpt_data(data):
    """
    Creates a new empty list called gpt_data, loops through each dictionary in the passed data object,
    creates a new dictionary with modified fields, and adds it to the gpt_data list.

    Args:
        data (dict or list): A list of dictionaries containing "instruction", "input", and "output" fields.

    Returns:
        list: The populated gpt_data list.
    """
    gpt_dict = []
    system_prompt = {"from": "system", "value": SYSTEM_PROMPT}
    user_prompt = {"from": "user", "value": data["instruction"] + " " + data["input"]}
    gpt_response = {"from": "gpt", "value": data["output"]}
    gpt_dict.append(system_prompt)
    gpt_dict.append(user_prompt)
    gpt_dict.append(gpt_response)
    return gpt_dict


data = read_alpaca("ReWiz-Data-20.0k-121024.json")
new_data = {"conversations": []}
for entry in data:
    new_data["conversations"].append(create_gpt_data(entry))
print(new_data)

save_file("ReWiz-GPT-Data-20.0k.json", new_data)