import sys
from openai import OpenAI

# Iterate over the inputs in the dataset
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import json
import os, time
from scripts.get_prompts import get_prompt


class InputData:
    def __init__(self, dataset, dataset_split):
        self.dataset = dataset
        self.dataset_split = dataset_split
        with open(f'{dataset}/{dataset_split}/labels.json', encoding="utf-8") as f:
            labels = json.load(f)
            self.dataset_len = len(labels)

def create_output_structure(input_data: InputData, model):
    output_path = f"rg.review_{model}_{input_data.dataset_split}.json"
    return output_path

sys_part = "You are a helpful assistant who can make bookings and reservations. Below are documents like reviews and FAQs for hotels and restaurants, paired with a conversation between Assistant and User. If the provided documents are insufficient or contradictory, inform User."
task_part = "\n\n## Task\nNow give a concise answer as Assistant.\n\n Assistant:"

llama2_str = "[INST] <<SYS>> You are a helpful assistant who can make bookings and reservations. Below are documents like reviews and FAQs for hotels and restaurants, paired with a conversation between Assistant and User. If the provided documents are insufficient or contradictory, inform User. <</SYS>> {}\n\n## Task\nNow give a concise answer as Assistant. [/INST]Assistant:"

wizardlm_str = "You are a helpful assistant who can make bookings and reservations. Below are documents like customer reviews and FAQs for hotels and restaurants, paired with a conversation between Assistant and User. If the provided documents are insufficient or contradictory, inform User.\n\n{}\n\n## Task\nNow give a concise answer as Assistant.\n\n### Response: Assistant:"

prompt_llama1_str = "You are a helpful assistant who can make bookings and reservations. Below are documents like customer reviews and FAQs for hotels and restaurants, paired with a conversation between Assistant and User. If the provided documents are insufficient or contradictory, inform User.\n\n{}\n\n## Task\nNow give a concise answer as Assistant.\n\n Assistant:"

def prompt_llama(input, url='http://gpu-19.apptek.local:8090/generate'):
    data = {}
    data["parameters"] = {  "do_sample": True, "min_new_tokens": 5, "max_new_tokens": 50, "num_beams": 5}
    data["inputs"] = wizardlm_str.format(input)

    headers = {
    'Content-Type': 'application/json'
    }

    response_text = requests.post(url, json=data, headers=headers).text
    response_text = json.loads(response_text)["generated_text"]
    return response_text



def prompt_openai(context, model, client):
    assert isinstance(context, str)

    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": sys_part}, {"role": "user", "content": context + task_part}],
        max_tokens=50,
    )
    return completion.choices[0].message.content


def process_request(dataset, i, model, client):
    context, target = get_prompt(i, False, dataset=dataset.dataset, split=dataset.dataset_split,
                                 max_input_tokens=1024, labels=None)
    if not target:
        return {"target": False, "id": i}

    # print(f"Sending request {i} with content\n\n {sys_part} \n " + context + task_part + "\n\n------------------")

    response = prompt_openai(context, model, client)
    return {"target": target, "response": response.replace("Assistant: ", ""), "id": i}


if __name__ == "__main__":
    dataset = InputData("data", "test")
    model = "gpt-3.5-turbo-0613"
    load_dotenv()
    client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"))

    out_path = create_output_structure(dataset, model)

    responses = []
    num_threads = 8

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(process_request, dataset, i, model, client) for i in range(dataset.dataset_len)]
        # regularly save in case of crash
        for i, future in enumerate(as_completed(futures)):

            responses.append(future.result())
            print("Response saved.\n------------------")
            if i % 50 == 0:
                with open(out_path, 'w') as file:
                    responses = sorted(responses, key=lambda resp: resp["id"])
                    json.dump(responses, file)

    with open(out_path, 'w') as file:
        responses = sorted(responses, key=lambda resp: resp["id"])
        json.dump(responses, file)

