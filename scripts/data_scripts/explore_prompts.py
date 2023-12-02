from scripts.data_scripts.get_prompts import get_prompt

label_print = False
print("[If you want to switch label printing, enter `l` instead of an ID.")
while True:
    response = input("\n------------\nPlease enter an ID: ")
    try:
        id = int(response)
    except ValueError:
        if response == "l":
            if label_print:
                label_print = False
            else:
                label_print = True
            continue
        else:
            raise ValueError("Unknown input.")
    print(get_prompt(id, split="val", dataset="data", label_print=label_print, max_turns=15, max_n_sent=15)[0])
    