from trl import SFTTrainer
from peft import LoraConfig
from transformers import TrainingArguments
from load_tokenizer import *
from model_arguments import *
from apply_chat_template import get_ultra_dataset

output_dir = 'data/zephyr-7b-sft-lora'

train_dataset,test_dataset = get_ultra_dataset()

training_args = TrainingArguments(
    fp16=True,
    do_eval=True,
    evaluation_strategy="epoch",
    gradient_accumulation_steps=128,
    gradient_checkpointing_kwargs={'use_reentrant':False},
    learning_rate=2.0e-05,
    log_level='info',
    logging_steps=5,
    lr_scheduler_type='cosine',
    max_steps=-1,
    num_train_epochs=1,
    output_dir=output_dir,
    overwrite_output_dir=True,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    save_strategy='no',
    save_total_limit=None,
    seed=42
    )


peft_config = LoraConfig(
    r=64,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj","k_proj","v_proj","o_proj"]
)

trainer = SFTTrainer(
    model=model_id,
    model_init_kwargs=model_kwargs,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    dataset_text_field='text',
    tokenizer=tokenizer,
    packing=True,
    peft_config=peft_config,
    max_seq_length=tokenizer.model_max_length
)
