import json 
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.model_selection import train_test_split
from datasets import Dataset


def get_dataset(file_path):

    data = []
    
    try:
        with open(file_path, "r") as file:
            for line in file:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
            print(f"Dataset loaded successfully: {len(data)} samples")
    
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    
    except json.JSONDecodeError as e:
        print(f"Invalid JSON format: {e}")
        
    return data

def dataset_preprocess(dataset, tokenizer):
    try:
        if not dataset:
            raise ValueError("Dataset is empty or None")
        
        if tokenizer is None:
            raise ValueError("Tokenizer is None")
        
        for data in dataset:
            message = [{"role":"user", "content":data['prompt']}]
            
            prompt = tokenizer.apply_chat_template(
                message,
                tokenize = False,
                add_generation_prompt = True
            )
            
            data['prompt'] = prompt
            
    except ValueError as e:
        print(f"Invalid input: {e}")
    
    except KeyError as e:
        print(f"Missing expected field in dataset: {e}")
            
    except Exception as e:
        print(f"Unexpected error: {e}")
        
    return dataset

## convert list of dataset into a Huggingface Dataset Object
def train_test_dataset_object(dataset, test_percentage=0.2):
    
    train_dataset, test_dataset = None, None
    
    try:
        
        if not dataset:
            raise ValueError("Dataset is empty or None")
        
        if not (0 < test_percentage < 1):
            raise ValueError(f"test percentage must in between 0 and 1. got: {test_percentage}")
        
        train_set, test_set = train_test_split(dataset, test_size=test_percentage,
                                            shuffle=True, random_state=42)
        
        train_dataset = Dataset.from_list(train_set)
        test_dataset = Dataset.from_list(test_set)
        
    except ValueError as e:
        print(f"Invalid input: {e}")
    
    except Exception as e:
        print(f"Unexpected error: {e}")
    
    return train_dataset, test_dataset

