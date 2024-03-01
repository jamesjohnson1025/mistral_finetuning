from transformers import AutoTokenizer

model_id = 'mistralai/Mistral-7B-v0.1'

tokenizer = AutoTokenizer.from_pretrained(model_id)

# set pad token id to eos token id if pad token id is not set 
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

if tokenizer.model_max_length > 100_000:
    tokenizer.max_model_length = 2048

# set the chat template 
DEFAULT_CHAT_TEMPLATE = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n' + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>'}}\n{% endif %}\n{% endfor %}"
tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE

'''
Below is the structure for chat template 

{% for message in messages %}\n
{% if message['role'] == 'user' %}\n
{{ '<|user|>\n' + message['content'] + eos_token }}\n
{% elif message['role'] == 'system' %}\n
{{ '<|system|>\n' + message['content'] + eos_token }}\n
{% elif message['role'] == 'assistant' %}\n
{{ '<|assistant|>\n' + message['content'] + eos_token }}\n
{% endif %}\n
{% if loop.last and add_generation_prompt %}\n
{{ '<|assistant|>'}}\n
{% endif %}\n
{% endfor %}

'''







