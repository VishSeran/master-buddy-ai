from transformers import AutoTokenizer, AutoModelForCausalLM


def model_push_hub():
    save_path = "../model/master_buddy_merged_v1.0"
    model = AutoModelForCausalLM.from_pretrained(save_path)
    tokenizer = AutoTokenizer.from_pretrained(save_path)

    model.push_to_hub("SeranVishwa/master-buddy-v1.0")
    tokenizer.push_to_hub("SeranVishwa/master-buddy-v1.0")
    
if __name__ == "__main__":
    model_push_hub()
