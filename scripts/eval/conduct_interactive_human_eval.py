import json
import random

from PyQt5.QtWidgets import QApplication, QInputDialog
import textwrap

def get_user_input(text_display):
    app = QApplication([])

    input_dialog = QInputDialog()
    input_dialog.setInputMode(QInputDialog.TextInput)
    input_dialog.setWindowTitle('Input Dialog')
    input_dialog.setLabelText('\n'.join(textwrap.fill(line, width=79) for line in text_display.split('\n')))
    input_dialog.resize(500, 500)
    
    ok = input_dialog.exec_()
    text = input_dialog.textValue()

    app.quit()
    return text


# Load the data
with open('/u/nils.hilgers/setups/dialog_setup/output/dstc/dstc11_track5/generation/merged_predictions--gpt3.5-turbo/zero_shot.json', 'r') as f:
    data_model1 = json.load(f)
with open('/u/nils.hilgers/setups/dialog_setup/output/dstc/dstc11_track5/generation/merged_predictions--t5-large/ft.json', 'r') as f:
    data_model2 = json.load(f)

# Randomly select n samples with "target" = True
samples_model1 = [sample for i, sample in enumerate(data_model1) if sample['target']]
samples_model2 = [sample for i, sample in enumerate(data_model2) if sample['target']]

# Create a list of indices
indices = list(range(len(samples_model1)))

# Shuffle the list of indices
random.shuffle(indices)

ratings_model1 = []
ratings_model2 = []

# if a evaluation_model1_model2.json exists, load it
try:
    with open('evaluation_model1_model2.json', 'r') as f:
        ratings = json.load(f)
        ratings_model1 = ratings["Model1"]
        ratings_model2 = ratings["Model2"]
except FileNotFoundError:
    pass

# Print the explanation of metrics
expl = ""
expl += "\n" + "Appropriateness: whether the response is fluent and naturally connected to the dialogue context."
expl += "\n" + "Aspect Accuracy: whether the response provides relevant and useful information to the aspect that the user queried. Context provided."
expl += "\n" + "Sentiment Accuracy: whether the sentiment proportion provided by the response is accordant with that of the subjective knowledge. Context provided."

# Ask user for number of samples to evaluate
n = int(get_user_input(expl + "\n" + "\n" + 'How many samples would you like to evaluate? '))

# Loop over the samples and ask the user to rate each one
for z,id1 in enumerate(indices):
    if z >= n:
        break
    sample1, sample2 = samples_model1[id1], samples_model2[id1]
    print(f"\nSample ID: {id1}\n")
    dialog_str = "Dialog\n\n" + ':U:'.join(sample1['input'].split(':U:')[1:])

    models = ['Model1', 'Model2']
    responses = [sample1['response'], sample2['response']]
    random_order = random.sample(range(2), 2) # randomly order the models for each sample

    for i in random_order:
        resp = f"\n\nModel Response\n\n" + responses[i]
        rating_appropriate = int(get_user_input(dialog_str + resp + f'\n\nRate the response for Appropriateness (1-5): '))
        context_str = "\nContext\n\n" + sample1['input'].split(':U:')[0]
        rating_aspect = int(get_user_input(dialog_str + resp + "\n" + context_str + '\n\nRate the response for Accuracy (1-5): '))

        if models[i] == 'Model1':
            ratings_model1.append((id1, rating_appropriate, rating_aspect))
        else:
            ratings_model2.append((id1, rating_appropriate, rating_aspect))

        with open('evaluation_model1_model2.json', 'w') as f:
            json.dump({'Model1': ratings_model1, 'Model2': ratings_model2}, f)


# Calculate average ratings
avg_rating_model1 = [sum(rating[i] for rating in ratings_model1) / len(ratings_model1) for i in range(1,2)]
avg_rating_model2 = [sum(rating[i] for rating in ratings_model2) / len(ratings_model2) for i in range(1,2)]

print(f"\nAverage rating for Model1: {avg_rating_model1}") # [Appropriateness, Accuracy]
print(f"Average rating for Model2: {avg_rating_model2}") # [Appropriateness, Accuracy]

# Store ratings in a new .json file
with open('evaluation_model1_model2.json', 'w') as f:
    json.dump({'Model1': ratings_model1, 'Model2': ratings_model2}, f)
