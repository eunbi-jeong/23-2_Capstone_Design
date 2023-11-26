import json
from datasets import load_dataset

def save_squad_validation_as_json():
    dataset = load_dataset("squad", split="validation")

    data_to_save = []
    for example in dataset:
        data_to_save.append({
            "answers":example["answers"],
            "question": example["question"],
            "context": example["context"],
            "title": example["title"],
            "id": example["id"]
        })

    with open("squad.json", "w") as json_file:
        json.dump(data_to_save, json_file)

if __name__ == "__main__":
    save_squad_validation_as_json()

