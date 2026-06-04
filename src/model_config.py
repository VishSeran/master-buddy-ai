from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_ref_model(model_name):
    ref_model = AutoModelForCausalLM.from_pretrained(model_name)
    for param in ref_model.parameters():
        param.requires_grad = False
    return ref_model.to(device)

def load_base_model (model_name):
    base_model = AutoModelForCausalLM.from_pretrained(model_name)
    return base_model

def load_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=4,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj']
    
)

def get_lora_model(model_name, 
                   bnb_config=bnb_config, 
                   lora_config = lora_config):

    try:
        qlora_model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                quantization_config=bnb_config)
        qlora_model.config.use_cache = False
        peft_model = get_peft_model(qlora_model, lora_config).to(device)
    
    except Exception as e:
        print(f"Unexpected error: {e}")
            
    return peft_model
    