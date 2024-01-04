import json
from itertools import groupby
from operator import itemgetter

def get_prompt(id, label_print=False, split="val", max_turns=None, max_n_sent=None, dataset="data"):

    with open(f'{dataset}/{split}/knowledge.json', encoding="utf-8") as f:
        knowledge = json.load(f)

    with open(f'{dataset}/{split}/labels.json', encoding="utf-8") as f:
        labels = json.load(f)

    with open(f'{dataset}/{split}/logs.json', encoding="utf-8") as f:
        logs = json.load(f)

    # Check if target is False, return message
    if not labels[id]['target']:
        return "The target for this ID is set to 'False'. Please enter a different ID.", False

    # Generate sentences from knowledge.json
    sentences = []
    current_doc_id = None
    current_entity_id = None
    dstc9_map = False
    n_sent = 0

    q_key = "question"
    a_key = "answer"

    cur_knowledge_set = labels[id]['knowledge']
    if "doc_type" not in cur_knowledge_set[0]:
        dstc9_map = True
        for i, snippet in enumerate(cur_knowledge_set):
            snippet["sent_id"] = 0
            snippet["doc_type"] = "faq"
            cur_knowledge_set[i] = snippet


    # Group the items from labels[id]['knowledge'] by the same entity_id, doc_type and doc_id
    cur_knowledge_set.sort(key=lambda x: (x['entity_id'], x['doc_type'], x['doc_id'], x.get('sent_id', float('-inf'))))

    # group them by entity_id, doc_type, and doc_id
    grouped_data = []
    for key, group in groupby(cur_knowledge_set, key=itemgetter('entity_id', 'doc_type', 'doc_id')):
        grouped_data.append(list(group))

    for info_sentence_set in grouped_data:
        for sentence in info_sentence_set:
            domain = sentence['domain']
            entity_id = str(sentence['entity_id'])
            doc_id = str(sentence['doc_id'])
            doc_type = str(sentence["doc_type"]) + "s"
            if doc_type != "faqs":
                sent_id = str(sentence['sent_id'])

            if doc_type != "faqs":
                text = knowledge[domain][entity_id][doc_type][doc_id]['sentences'][sent_id]
            else:
                if dstc9_map:
                    doc_type = "docs"
                    q_key = "title"
                    a_key = "body"
                text = knowledge[domain][entity_id][doc_type][doc_id][q_key] + " " + knowledge[domain][entity_id][doc_type][doc_id][a_key]
                doc_type = "faqs"
            entity_name = str(knowledge[domain][entity_id]['name'])
            
            if max_n_sent is not None and n_sent + 1 > max_n_sent:
                break
            n_sent += 1

            # Add a separator if this is a new document
            #if entity_id != current_entity_id:
                # STYLE
            if doc_id != current_doc_id:
                if doc_type != "faqs":
                    sentences.append(f":Doc: ({entity_name})")
                else:
                    sentences.append(f":Doc: ({entity_name})")
            current_doc_id = doc_id
            current_entity_id = entity_id

            sentences.append(text)

    # Get all responses of all speakers separated by the speaker's name in the logs.json
    selected_turns = []
    turns = logs[id]
    n_turns = 0
    # If max_turns is set, only use the last max_turns turns
    for log in turns[-max_turns:]:
        if log['speaker'] == 'U':
            selected_turns.append("User: " + log['text'])
        elif log['speaker'] == 'S':
            selected_turns.append("Assistant: " + log['text'])
        n_turns += 1
        if max_turns is not None and n_turns + 1 > max_turns:
            break

    # Get the label
    label = labels[id]['response']
    if label_print:
        selected_turns.append(f":S: {label}")
    output = ""
    if max_n_sent > 0:
        output += '## Context\n' + ' '.join(sentences)
    if max_turns > 0:
        output += '\n\n' + '## Conversation\n' + ' '.join(selected_turns)

    return output, True
