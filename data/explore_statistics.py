import json

def print_stats(dataset="val"):

    # Load data
    with open(f'/u/nils.hilgers/setups/dstc11-track5/data/knowledge.json', encoding="utf-8") as f:
        knowledge = json.load(f)

    with open(f'/u/nils.hilgers/setups/dstc11-track5/data/{dataset}/labels.json', encoding="utf-8") as f:
        labels = json.load(f)

    with open(f'/u/nils.hilgers/setups/dstc11-track5/data/{dataset}/logs.json', encoding="utf-8") as f:
        logs = json.load(f)

    # Create a histogram of the length of "knowledge" over all objects in the dataset "labels.json"
    knowledge_length = []
    for label in labels:
        if label['target']:
            knowledge_length.append(len(label['knowledge']))
    print(f"Knowledge length: {knowledge_length}")
    print(f"Average knowledge length: {sum(knowledge_length)/len(knowledge_length)}")

    # Plot the histogram to a file
    import matplotlib.pyplot as plt
    plt.hist(knowledge_length, bins=range(0, 20))
    plt.savefig(f"knowledge_length_{dataset}.png")
    plt.clf()

    # Tell me for which number I have covered 95% of the data
    knowledge_length.sort()
    print(f"95% of the data is covered by {knowledge_length[round(len(knowledge_length)*0.95)]} knowledge items.")

print_stats()