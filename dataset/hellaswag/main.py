from datasets import DownloadManager
from hellaswag import Hellaswag
import json

def save_validation_to_json():
    # Initialize your Hellaswag dataset builder
    builder = Hellaswag()

    # Initialize the DownloadManager
    dl_manager = DownloadManager()

    # Load the validation split using _split_generators method
    splits = builder._split_generators(dl_manager)
    validation_data = splits[2]  # Assuming the third split is the validation split

    # Get the filepath for the validation split
    validation_filepath = validation_data.gen_kwargs["filepath"]

    # Initialize an empty dictionary to store examples
    examples = {}

    # Use the _generate_examples method to load validation examples
    for key, example in builder._generate_examples(validation_filepath):
        examples[key] = example

    # Define the path where you want to save the JSON file for validation data
    validation_json_path = 'hellaswag.json'

    # Save the examples dictionary as JSON to the defined path
    with open(validation_json_path, 'w') as f:
        json.dump(examples, f)

# Call the function to save the validation data to a JSON file
save_validation_to_json()

