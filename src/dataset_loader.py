import json

def get_dataset(file_path):
    
    data = []
    
    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    
    return data

def dataset_preprocess(dataset):
    pass