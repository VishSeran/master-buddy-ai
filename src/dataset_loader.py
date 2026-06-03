import json 
from transformers import AutoTokenizer, AutoModelForCausalLM
from model_config import load_tokenizer


def get_dataset(file_path):
    
    data = []
    
    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    
    return data



def dataset_preprocess(dataset, tokenizer):
    

    for data in dataset:
        message = [{"role":"user", "content":data['prompt']}]
        
        prompt = tokenizer.apply_chat_template(
            message,
            tokenize = False,
            add_generation_prompt = True
        )
        
        data['prompt'] = prompt
        
    return dataset