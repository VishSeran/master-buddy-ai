from transformers import AutoTokenizer, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("SeranVishwa/master-buddy-v1.0")
tokenizer = AutoTokenizer.from_pretrained("SeranVishwa/master-buddy-v1.0")