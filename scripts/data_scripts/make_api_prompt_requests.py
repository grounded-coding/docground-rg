import requests
import json
import os, time
# import openai
from llama_eval.prompt_llama_eval import prompt_llama_eval

# openai.api_key = os.environ["OPENAI_API_KEY"]

from scripts.data_scripts.get_prompts import get_prompt

# Your dataset with lists of inputs
dataset = "val"

responses = []

prompt_llama_str = "[INST] <<SYS>> You are a helpful assistant who can make bookings and reservations. You are provided with a context and must only use the context to answer the users. The context is related to the provided hotel or restaurant names. Context can be reviews from customers or FAQs. FAQs start after token :F: and each new review starts after token :R:. If the context info is not sufficient or contradictory, inform the user.<</SYS>> {}\n\n## Task\nNow give a concise answer as assistant.[/INST]Assistant:"

def prompt_llama(input, url='http://gpu-19.apptek.local:8080/generate'):
    data = {}
    data["parameters"] = {"max_new_tokens":256, "do_sample": True, "temperature": 0.95, "top_p": 0.95, "top_k": 250, "repetition_penalty": 1.15, "repetition_penalty_sustain": 256, "token_repetition_penalty_decay": 128}
    data["inputs"] = prompt_llama_str.format(input)

    headers = {
    'Content-Type': 'application/json'
    }

    # print(data)
    response_text = requests.post(url, json=data, headers=headers).text
    response_text = json.loads(response_text)["generated_text"]
    return response_text

def prompt_openai(input, prompt, url):
    # Send a payload chat completion request to the OpenAI API for gpt3.5
    
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

# Iterate over the inputs in the dataset
from concurrent.futures import ThreadPoolExecutor, as_completed

def process_request(i):
    input, target = get_prompt(i, False, dataset, max_turns=3, max_n_sent=15)
    if not target:
        return {"target": False, "id": i}

    print(f"Sending request {i} with content \n{input}\n------------------")

    response = prompt_llama(input)
    print(response)

    return {"target": target, "response": response.replace("Assistant: ", ""), "id": i}

responses = []
num_requests = 4173
num_threads = 16
file_name = "rg.review_llama_13b_zero_shot_val.json"

with ThreadPoolExecutor(max_workers=num_threads) as executor:
    futures = [executor.submit(process_request, i) for i in range(num_requests)]

    for i, future in enumerate(as_completed(futures)):
        responses.append(future.result())
        print("Response saved.\n------------------")

with open(file_name, 'w') as file:
    responses = sorted(responses, key=lambda resp: resp["id"])
    json.dump(responses, file)

