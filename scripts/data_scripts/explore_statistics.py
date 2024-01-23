import json    
import matplotlib.pyplot as plt

def print_stats(dataset="val"):

    # Load data
    with open(f'data/knowledge.json', encoding="utf-8") as f:
        knowledge = json.load(f)

    with open(f'data/{dataset}/labels.json', encoding="utf-8") as f:
        labels = json.load(f)

    with open(f'data/{dataset}/logs.json', encoding="utf-8") as f:
        logs = json.load(f)

    print(f"# Entries in {dataset} split")
    print(len(labels))

    # Create a histogram of the length of "knowledge" over all objects in the dataset "labels.json"
    knowledge_length = []
    for label in labels:
        if label['target']:
            knowledge_length.append(len(label['knowledge']))
    print(f"Average # snippets: {sum(knowledge_length)/len(knowledge_length)}")

    # Plot the histogram to a file

    plt.hist(knowledge_length, bins=range(0, 20))
    plt.savefig(f"snippet_numbers_{dataset}.png")
    plt.clf()

    # Tell me for which number I have covered 95% of the data
    knowledge_length.sort()
    print(f"95% of the data has less than {knowledge_length[round(len(knowledge_length)*0.95)]} knowledge snippets.")

for split in ["train", "val", "test"]:
    print_stats(split)
