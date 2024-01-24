from scripts.get_prompts import get_prompt
import json

split = "val"

# print the length of the dataset
with open(f'data/{split}/labels.json', encoding="utf-8") as f:
    labels = json.load(f)
print(f"Length of {split} dataset: {len(labels)}")

# print a list of 10 randomly selected IDs for which the target is True
import random
random.seed(42)
print("15 randomly selected IDs for which the target is True:")
for i in range(15):
    id = random.randint(0, len(labels))
    while not labels[id]['target']:
        id = random.randint(0, len(labels))
    print(id)

label_print = False
print("[If you want to switch label printing, enter `l` instead of an ID.")
while True:
    response = input("\n------------\nPlease enter an ID: ")
    try:
        id = int(response)
    except ValueError:
        if response == "l":
            if label_print:
                label_print = False
            else:
                label_print = True
            continue
        else:
            raise ValueError("Unknown input.")
    print(get_prompt(id, split=split, dataset="data", label_print=label_print, max_input_tokens=1024)[0])
    