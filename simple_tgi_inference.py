import requests
import json

url = "http://gpu-19.apptek.local:8090/generate"
headers = {"Content-Type": "application/json"}

def read_json_file(file_path):
    with open(file_path, "r") as file:
        return json.load(file)

# Read the JSON file as a dictionary

context = "## Context\nthe cleveland browns ' last playoff win was in 1995 against the new england patriots and their head coach was bill belichick . this was bill 's first playoff win as head coach and the only playoff win as head coach outside the patriots .\n\n## Conversation\nAssistant: i wonder whether letting women into the game would create interest , get more fans . User: oh , it absolutely would . it would be like ... and i know i am going to butcher the name here , but when manon reheaume played in the nhl . my only concern is that if mishandled it might come off like a gimmick . Assistant: i see your point . long term , will women stay ? true . \n i feel like ... there are enough talented women athletes out there , that a woman kicker or qb is possible , perhaps a wide receiver . User: women can be tough too . would a coach like bill belichick hire a woman if she could help him win the super bowl ? Assistant: this is going to come off sounding bad , but it is n't meant to . i think anyone in the front office would look at it from the perspective of generating revenue and views . the sponsors would love the coverage . User: belichick has done well in the post season , so i think he 's looking at it from a strategic point of view . the sponsors would tie her presence into a bunch of social issues and products. Assistant: yes , i agree . the last playoff win was in 1995 against the patriots ."

input_string = "Hello world"
data = {
    "inputs": "[INST] <<SYS>> For the following context, you will be given a multi-turn conversation between User and Assistant and a potential response for the next turn. Rate how accurate the conversation is in total. An accurate conversation is factually correct with respect to the context. An inaccurate conversation invents facts, which cannot be cited or derived from the context. Carefully assess the score if some parts of the conversation are correct and other parts are wrong. Do not rate helpfulness or the style of the conversation. <</SYS>> {}\n\n## Task\nFIRST provide a one-sentence explanation of your score. Double-check the context you use to verify information. SECOND, state only the score on a scale of 1 to 5. Follow the template.\n\n## Template\nExplanation: <one-sentence explanation>\nAccurate Score: <1-5>[/INST]Explanation:".format(context),
    "parameters": {
        "max_new_tokens": 256,
        "do_sample": False
    },
}

response = requests.post(url, data=json.dumps(data), headers=headers)

print(response.text)