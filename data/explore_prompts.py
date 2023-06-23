from get_prompts import get_prompt

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
    print(get_prompt(id, label_print, max_turns=3, max_n_sent=100)[0])
    