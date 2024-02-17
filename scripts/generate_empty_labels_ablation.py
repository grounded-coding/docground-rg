import json

with open('data/val/labels.json', encoding="utf-8") as f:
    labels = json.load(f)

for entry in labels:
    if entry['target']:
        entry['knowledge'] = []

with open('data/val/empty_knowledge_labels.json', 'w', encoding="utf-8") as f:
    json.dump(labels, f, ensure_ascii=False, indent=4)