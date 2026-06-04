from model_config import load_ref_model, load_tokenizer,get_lora_model, load_base_model
from dataset_loader import get_dataset,dataset_preprocess,train_test_dataset_object
from dpo_config import dpoTrainer
from peft import PeftModel

def main():
    
    try:
        model_name = "meta-llama/Llama-3.2-1B-Instruct"
        
        if not model_name:
            raise ValueError("Model name is empty or None")
        ref_model = load_ref_model(model_name)
        tokenizer = load_tokenizer(model_name)
        base_model = load_base_model(model_name)

        
        file_path = "../dataset/dpo_study_assistant_1000.jsonl"
        dataset = get_dataset(file_path=file_path)

        processed_dataset = dataset_preprocess(dataset=dataset, tokenizer=tokenizer)
        train_dataset, eval_dataset = train_test_dataset_object(dataset=processed_dataset)
        
        peft_model = get_lora_model(model_name=model_name)
        
        dpo_trainer = dpoTrainer(model=peft_model, ref_model=ref_model,
                                train_dataset=train_dataset, eval_dataset=eval_dataset,
                                tokenizer=tokenizer)
        
        save_path = "../model/master_buddy_v1.0"
        
        if not save_path:
            raise ValueError("save path is empty or None")
        
        #train model
        train_results = dpo_trainer.train()
        eval_results = dpo_trainer.evaluate()
        
        #save model
        
        
        dpo_trainer.save_model(save_path)
        print(f"dpo model saved to {save_path}")
        tokenizer.save_pretrained(save_path)
        print(f"tokenizer (dpo model) saved to {save_path}")
        dpo_trainer.save_state()
        print("dpo trainer state saved")
        
        #save train and eval metrics
        dpo_trainer.save_metrics(
            "train",
            train_results.metrics
        )
        print("dpo trainer train results saved")
        
        dpo_trainer.save_metrics(
            "eval",
            eval_results.metrics
        )
        print("dpo trainer evaluation results saved")
        
        print(f"Train loss: {train_results.metrics.get('train_loss', 'N/A')}")
        print(f"Eval loss: {eval_results.metrics.get('eval_loss', 'N/A')}")
        
        # merge the trained model with the base model
        model = PeftModel.from_pretrained(
            base_model,
            save_path
        )
        
        save_merge_model_path = "../model/master_buddy_merged_v1.0"
        if not save_merge_model_path:
            raise ValueError("merged save path is empty or None")
        
        merge_model = model.merge_and_unload()
        merge_model.save_pretrained(save_merge_model_path)
        tokenizer.save_pretrained(save_merge_model_path)
        print(f"Merged model saved to {save_merge_model_path}")
        
    except ValueError as e:
        print(f"Invalid input: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == '__main__':
    main()