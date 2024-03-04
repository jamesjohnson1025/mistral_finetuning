import re 
import random 
from multiprocessing import cpu_count 
from load_tokenizer import tokenizer
from load_dataset import raw_datasets
import os
import torch


def apply_chat_template(example,tokenizer):
    messages = example['messages']
    if messages[0]['role'] != 'system':
        messages.insert(0,{'role':'system','content':''})
    example['text'] = tokenizer.apply_chat_template(messages,tokenize=False)
    return example


column_names = list(raw_datasets['train'].features)
raw_datasets = raw_datasets.map(apply_chat_template,
                                num_proc=os.cpu_count(),
                                fn_kwargs={'tokenizer':tokenizer},
                                remove_columns=column_names,
                                desc='Applying chat template'
                            )

train_dataset = raw_datasets['train']
test_dataset = raw_datasets['test']