from datasets import load_dataset
from datasets import DatasetDict


# load the ultrachat dataset 
ultra_dataset = load_dataset("HuggingFaceH4/ultrachat_200k")

indices = range(0,10)

dataset_dict = {
        'train':ultra_dataset['train_sft'].select(indices=indices),
        'test':ultra_dataset['test_sft'].select(indices=indices)
        }

raw_datasets = DatasetDict(dataset_dict)

# load the first example 
first_example = raw_datasets['train'][0]

# load the messages 
messages = first_example['messages']
# messages is the list of dictionary 

# The length of raw dataset 
print(len(raw_datasets['train']))
