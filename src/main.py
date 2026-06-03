from model_config import load_ref_model, load_tokenizer
from dataset_loader import get_dataset,dataset_preprocess,train_test_dataset_object

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

    except ValueError as e:
        print(f"Invalid input: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
