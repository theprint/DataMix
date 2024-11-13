# DataMix

This is a simple utility that combines a weighted number of entries from various source data sets from *Huggingface* and creates a new dataset file (JSON) from the sampled data. With this tool, you can combined data sets that supplement each other and easily control the ratio of contribution from each source.

## Steps

You set everything up in **build_data.py** and should not need to edit the other files.

1. Set up the config in build_data.py by setting the size you want for the dataset, and the name you've chosen for the new set.
2. Organize your source huggingface datasets by format and assign weights. When added up, all the weights should total 1.0.
3. For each type of dataset used, call the appropriate function from collect_data.py.
4. Run build_data.py and wait for magic to happen.


