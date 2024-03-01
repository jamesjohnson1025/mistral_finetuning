from sft_trainer import *

train_result = trainer.train()

# And save the model 
metrics = train_result.metrics
max_train_samples = training_args.max_train_samples if training_args.max_train_samples is not None else len(train_dataset)
metrics["train_samples"] = min(max_train_samples, len(train_dataset))
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()