import requests
import json
import os, time
import openai

openai.api_key = os.environ["OPENAI_API_KEY"]

from get_prompts import get_prompt

# Your dataset with lists of inputs
dataset = "val"

# The URL of the API endpoint
url = "https://375cead61e4db124.gradio.app/run/predict"

responses = []

prompt_general = "You must complete the given chat history between assistant and user based only on the context about hotels and restaurants. You take the assistant role, and you can make bookings and reservations. The context is followed by the corresponding entity names. Context can be reviews from customers or FAQs. FAQs start after token :F: and each new review starts after token :R:. User turns start after token :U: and assistant turns after token :S:. If the context info is not sufficient or contradictory, inform the user. Always answer concisely."

prompt_openai = "You are a helpful assistant who can make bookings and reservations. You are provided with context information and must only use this information. The context is related to the provided hotel or restaurant name. Context can be reviews from customers or FAQs. FAQs start after token :F: and each new review starts after token :R:. If the context info is not sufficient or contradictory, inform the user. Always answer as concisely as possible. Context: "

prompt = prompt_openai
print(f"PROMPT\n{prompt}\n------------------")

def prompt_gradio(input, prompt, url):
    payload = {
        "data": [
            f"{prompt} INPUT: {input}",  # The instruction input
            1,  # Temperature
            0.95,  # Top p
            40,  # Top k
            1,  # Beams
            2048,  # Max tokens
        ]
    }

    # Send a POST request to the API and save the response
    print(f"Sending request {i} with content \n{input}\n------------------")
    response = requests.post(url, json=payload)

    if response.status_code == 200:
        data = response.json()["data"][0]
    else:
        data = f"Request failed with status code {response.status_code}"
        print("REQUEST FAILED, CHECK RESPONSES.JSON FOR DETAILS\n------------------")
    return data

def prompt_openai(input, prompt, url):
    # Send a payload chat completion request to the OpenAI API for gpt3.5
    
    # Split the input text based on the token :U: but keep the token
    context = input.split(":U:")[0]
    # Remove the first element of the list which is the context
    turns = input.split(":U:")[1:]
    system_turns = turns[0].split(":S:")[1:]

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
for i in range(1681, 4173):
    # The payload to send to the API
    input, target = get_prompt(i, False, dataset, max_turns=3, max_n_sent=8)
    if not target:
        responses.append({"target": False})
        continue

    print(f"Sending request {i} with content \n{input}\n------------------")

    response = prompt_openai(input, prompt, url)

    # Check if the request was successful

        # Save the response data to the list
    # Retrieve the first response from the list of responses but remove ":S: " from the beginning if it exists
    responses.append({"input": input, "response": response.replace(":S: ", ""), "target": target})
    print("Response saved.\n------------------")

    # Save the responses to the JSON file every 20 steps
    if i % 20 == 0 and i > 0:
        with open('responses.json', 'w') as file:
            json.dump(responses, file)

# Save the remaining responses to the JSON file
with open('responses.json', 'w') as file:
    json.dump(responses, file)


# Print the list of responses
print(responses)
