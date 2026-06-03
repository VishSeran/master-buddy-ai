from transformers import AutoTokenizer, AutoModelForCausalLM

def load_ref_model(model_name):
    ref_model = AutoModelForCausalLM.from_pretrained(model_name)
    return ref_model

def load_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer

def bnb_config ()