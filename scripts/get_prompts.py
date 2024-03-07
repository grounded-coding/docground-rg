import json
from itertools import groupby
from operator import itemgetter

def truncate_output(knowledge_text, conversation_text, max_tokens=1024, char_per_token=4):
    entire_input_len = max_tokens * char_per_token if max_tokens else 1024 * 4

    # Calculate the length of knowledge text and conversation text
    entire_knowledge_len = len(knowledge_text)
    entire_history_len = len(conversation_text)

    # Calculate maximum length allowed for each
    max_conversation_length = int((entire_history_len * entire_input_len) / (entire_knowledge_len + entire_history_len))
    max_conversation_length = min(entire_history_len + entire_history_len, 512 * 4)
    max_knowledge_length = entire_input_len - max_conversation_length

    # Truncate knowledge text and conversation text if necessary
    if max_knowledge_length < entire_knowledge_len:
        truncated_knowledge = knowledge_text[:max_knowledge_length].rsplit(' ', 1)[0]
    else:
        truncated_knowledge = knowledge_text

    # Truncate from the start of the conversation text if necessary
    remaining_chars_for_conversation = len(conversation_text) - max_conversation_length
    if remaining_chars_for_conversation > 0:
        truncated_conversation = conversation_text[remaining_chars_for_conversation:].lstrip()
    else:
        truncated_conversation = conversation_text

    # Combine truncated parts into final output
    truncated_output = truncated_knowledge
    if truncated_conversation:
        truncated_output += '\n\n ## Conversation\n' + truncated_conversation

    return truncated_output


def get_prompt(id, label_print=False, split="val", max_turns=10000, max_n_sent=10000, dataset="data", max_input_tokens=None, labels=None):

    with open(f'{dataset}/{split}/knowledge.json', encoding="utf-8") as f:
        knowledge = json.load(f)

    if not labels:
        labels = f'{dataset}/{split}/labels.json'
    with open(labels, encoding="utf-8") as f:
        labels = json.load(f)

    with open(f'{dataset}/{split}/logs.json', encoding="utf-8") as f:
        logs = json.load(f)

    # Check if target is False, return message
    if not labels[id]['target']:
        return "The target for this ID is set to 'False'. Please enter a different ID.", False

    # Generate sentences from knowledge.json
    sentences = []
    current_doc_id = None
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

    for knowledge_snippet_set in grouped_data:
        for snippet in knowledge_snippet_set:
            domain = snippet['domain']
            entity_id = str(snippet['entity_id'])
            doc_id = str(snippet['doc_id'])
            doc_type = str(snippet["doc_type"]) + "s"
            if doc_type != "faqs":
                sent_id = str(snippet['sent_id'])

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
            
            if n_sent + 1 > max_n_sent:
                break
            n_sent += 1

            if doc_id != current_doc_id:
                if doc_type != "faqs":
                    sentences.append(f":Doc: ({entity_name})")
                else:
                    sentences.append(f":Doc: ({entity_name})")
            current_doc_id = doc_id

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
        if n_turns + 1 > max_turns:
            break

    # Get the label
    label = labels[id]['response']
    if label_print:
        selected_turns.append(f":S: {label}")

    knowledge_text = ' '.join(sentences) if max_n_sent > 0 else ""
    conversation_text = ' '.join(selected_turns) if max_turns > 0 else ""

    output = truncate_output(knowledge_text, conversation_text, max_tokens=max_input_tokens)
    return output, True
