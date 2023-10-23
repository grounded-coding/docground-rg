import sys
sys.path.append("/home/nhilgers/setups/dstc11-track5")


# Iterate over the inputs in the dataset
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import json
import os, time
import openai
from scripts.data_scripts.get_prompts import get_prompt


responses = []
num_requests = 4173
num_threads = 16
file_name = "rg.review_wizardlm1-7b_zero_shot_val.json"
# openai.api_key = os.environ["OPENAI_API_KEY"]

dataset = "val"

responses = []

prompt_llama2_str = "[INST] <<SYS>> Below is a context with customer reviews and FAQs for hotels and restaurants, paired with a conversation between Assistant and User. Answer the question from User appropriately as Assistant. <</SYS>> {} [/INST] Assistant:"
wizardlm_str = "Below is a context with customer reviews and FAQs for hotels and restaurants, paired with a conversation between Assistant and User. Answer the question from User appropriately as Assistant.\n\n{}\n\n### Response: Assistant:"
prompt_llama1_str = "Below is a context with customer reviews and FAQs for hotels and restaurants, paired with a conversation between Assistant and User. Answer the question from User appropriately as Assistant.\n\n{} Assistant:"

def prompt_llama(input, url='http://gpu-19.apptek.local:8090/generate'):
    data = {}
    data["parameters"] = {  "do_sample": True, "min_new_tokens": 5, "max_new_tokens": 150, "num_beams": 5}
    data["inputs"] = wizardlm_str.format(input)

    headers = {
    'Content-Type': 'application/json'
    }

    # print(data)
    response_text = requests.post(url, json=data, headers=headers).text
    response_text = json.loads(response_text)["generated_text"]
    return response_text

def prompt_openai(input, prompt, url):    
    # Split the input text based on the token :U: but keep the token
    context = input.split("User:")[0]
    # Remove the first element of the list which is the context
    turns = input.split("User:")[1:]
    system_turns = turns[0].split("Assistant:")[1:]

    try:
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages= [{"role": "system", "content": prompt + context}, {"role": "user", "content": turns[0]},
                    {"role": "assistant", "content": system_turns[0]},
                    {"role": "user", "content": turns[1]}],
            max_tokens=512,
        )
    except openai.error.RateLimitError:
        time.sleep(10)
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages= [{"role": "system", "content": prompt + context}, {"role": "user", "content": turns[0]},
                    {"role": "assistant", "content": system_turns[0]},
                    {"role": "user", "content": turns[1]}],
            max_tokens=512,
        )
    return completion["choices"][0]["message"]["content"]


def process_request(i):
    input, target = get_prompt(i, False, dataset, max_turns=3, max_n_sent=15)
    if not target:
        return {"target": False, "id": i}

    print(f"Sending request {i} with content \n{input}\n------------------")

    response = prompt_llama(input)
    print(response)

    return {"target": target, "response": response.replace("Assistant: ", ""), "id": i}

with ThreadPoolExecutor(max_workers=num_threads) as executor:
    futures = [executor.submit(process_request, i) for i in range(num_requests)]

    for i, future in enumerate(as_completed(futures)):
        responses.append(future.result())
        print("Response saved.\n------------------")

with open(file_name, 'w') as file:
    responses = sorted(responses, key=lambda resp: resp["id"])
    json.dump(responses, file)

