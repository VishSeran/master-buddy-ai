# resume.py
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def resume():
    try:
        base_model_name = "meta-llama/Llama-3.2-1B-Instruct"
        save_path = "../model/master_buddy_v1.0"
        save_merge_model_path = "../model/master_buddy_merged_v1.0"

        print("Loading base model and tokenizer...")
        base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
        tokenizer = AutoTokenizer.from_pretrained(save_path)  # use saved tokenizer

        print("Loading trained adapter...")
        model = PeftModel.from_pretrained(base_model, save_path)

        print("Merging model...")
        merge_model = model.merge_and_unload()

        merge_model.save_pretrained(save_merge_model_path)
        tokenizer.save_pretrained(save_merge_model_path)
        print(f"Merged model saved to {save_merge_model_path}")

    except Exception as e:
        import traceback
        print(f"Unexpected error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    resume()