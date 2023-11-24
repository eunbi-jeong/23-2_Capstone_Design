import json
from datasets import load_dataset

def save_qnli_validation_as_json():
    dataset = load_dataset("glue", "qnli", split="validation")

    data_to_save = []
    for example in dataset:
        data_to_save.append({
            "question": example["question"],
            "sentence": example["sentence"],
            "label": example["label"],
            "idx": example["idx"]
        })


    with open("glue.json", "w") as json_file:
        json.dump(data_to_save, json_file)

if __name__ == "__main__":
    save_qnli_validation_as_json()

