# DataMix

This is a simple utility that combines a weighted number of entries from various source data sets from *Huggingface* and creates a new dataset file (JSON) from the sampled data. With this tool, you can combined data sets that supplement each other and easily control the ratio of contribution from each source.

## Steps

If you are working with gated datasets, you will need a Huggingface token. Get this from HF and set it up as an environment variable by adding it to a .env file as follows:

```python
HF_TOKEN = ""wHat3v3rYoUrToK3nV4lU3isH3r3"
```
You set each run up in **build_data.py** and should not need to edit the other files.

### 1. Edit config
Set up the config in build_data.py by setting the size you want for the dataset, and the name you've chosen for the new set.

```python
config = {
    "total_samples": 80000,  # This the total size you want your new dataset to be
    "dataset_name": "VanRossum",  # This is the name of your new dataset
    "hf_token": hf_token,  # Should be set in the .env file
    "seed": 18072005  # A numerical seed is used for consistent generation
}
```

### 2. Add source sets
Organize your source huggingface datasets by format and assign weights. This is the most complicated part of the setup, and it's really not that hard. There are two things to keep an eye on:
- The dataset is added in the right format category. This always has to do with the naming of the columns in the source. Double check on Huggingface is necessary.
- When added up, all the weights should total 1.0.
Each source is represented by a tuple containing the two values listed above.

Here is an example:
```python
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
```

### 3. Adding formats
If you are using a source set that does not follow any of the formats listed in formats.yaml, you can add it yourself. Follow the same structure and manually edit formats.yaml with the information matching your source.

When adding a new format, it is important that you name the **instruction_key** and **output_key** exactly what those columns are called in the source data. 

For example, if the user input in your source data is called _user_input_, and the response is called _ai_response_, you would add it like this:
```yaml
  user_input_ai_response:
    instruction_key: user_input
    output_key: ai_response
```

### 4. Run build_data.py and wait for magic to happen.


