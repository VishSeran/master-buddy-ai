from model_config import load_ref_model, load_tokenizer,get_lora_model
from dataset_loader import get_dataset,dataset_preprocess,train_test_dataset_object
from dpo_config import dpoTrainer

def main():
    
    try:
        model_name = "meta-llama/Llama-3.2-1B-Instruct"
        
        if not model_name:
            raise ValueError("Model name is empty or None")
        ref_model = load_ref_model(model_name)
        tokenizer = load_tokenizer(model_name)

        
        file_path = "../dataset/dpo_study_assistant_1000.jsonl"
        dataset = get_dataset(file_path=file_path)

        processed_dataset = dataset_preprocess(dataset=dataset, tokenizer=tokenizer)
        train_dataset, eval_dataset = train_test_dataset_object(dataset=processed_dataset)
        
        peft_model = get_lora_model(model_name=model_name)
        
        dpo_trainer = dpoTrainer(model=peft_model, ref_model=ref_model,
                                train_dataset=train_dataset, eval_dataset=eval_dataset,
                                tokenizer=tokenizer)
        #train model
        train_results = dpo_trainer.train()
        eval_results = dpo_trainer.evaluate()
        
        #save model
        dpo_trainer.save_model("../model/master_buddy_v1.0")
        tokenizer.save_pretrained("../model/master_buddy_v1.0")
        dpo_trainer.save_state()
        
        #save train and eval metrics
        dpo_trainer.save_metrics(
            "train",
            train_results.metrics
        )
        
        dpo_trainer.save_metrics(
            "eval",
            eval_results.metrics
        )
        
        
        
    except ValueError as e:
        print(f"Invalid input: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
