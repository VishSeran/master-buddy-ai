import torch
from trl import DPOConfig, DPOTrainer

dpoConfig = DPOConfig(
    beta=0.1,
    output_dir="/model_output",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    learning_rate=1e-4,
    logging_steps=40,
    eval_steps=100,
    eval_strategy="steps",
    save_strategy="epoch",
    report_to="none",
    remove_unused_columns=False,
    
    )

def dpoTrainer(model, ref_model,train_dataset, eval_dataset,
               tokenizer,config=dpoConfig):
    
    try:
        
        if not model:
            raise ValueError("Model is empty or None")
        
        if not ref_model:
            raise ValueError("Ref model is empty or None")
        
        if not train_dataset:
            raise ValueError("Train dataset is empty or None")
        
        if not eval_dataset:
            raise ValueError("Evaluation dataset is empty or None")
        
        if not tokenizer:
            raise ValueError("Tokenizer is empty or None")
        
        dpo_trainer = DPOTrainer(
            model=model,
            ref_model=ref_model,
            args=config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer
        )
    
    except ValueError as e:
        print(f"Invalid input: {e}")
    
    except Exception as e:
        print(f"Unexpected error: {e}")
    
    return dpo_trainer