from transformers import BitsAndBytesConfig
import torch 

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.bfloat16
)

device_map = {"":torch.cuda.current_device()} if torch.cuda.is_available() else None 

model_kwargs = dict(
    attn_implementation='flash_attention_2',
    torch_dtype='auto',
    use_cache=False,
    device_map=device_map,
    quantization_config=quantization_config
)
