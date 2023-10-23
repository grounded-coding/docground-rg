

from uni_eval.utils import convert_to_json
from uni_eval.evaluator import get_evaluator

task = 'dialogue'

# a list of dialogue histories
src_list = ['Do you want to book a room at Frankie and Bennys or at the Pizza Hut? \n Do any of them have free wifi? \n\n']
# a list of model outputs to be evaluated
output_list1 = ["Unfortunately the Pizza Hut doesn't have good reviews about the wifi and it seems to cost money. I can recommend Frankie and Bennys which has free wifi."]

output_list2 = ["They do not have any wifi at all."]


# Initialize evaluator for a specific task
evaluator = get_evaluator(task)

context_list = ["(FRANKIE AND BENNYS) The wifi here was free and in general I was happy with the service. \n(FRANKIE AND BENNYS) Quality of the wifi was excellent. \n(Pizza Hut) Terrible wifi. \n(Pizza Hut) The wifi is more than 1GBP per hour.\n"]

output_list2 = ["Frankie and Bennys has more complaints than good reviews saying service was slow and inattentive. The Pizza Hut has some complaints about slow, inattentive service but had more good reviews than bad.\n"]

# Prepare data for pre-trained evaluators
data = convert_to_json(output_list=output_list1, 
                       src_list=src_list, context_list=context_list)

data2 = convert_to_json(output_list=output_list2, 
                       src_list=src_list, context_list=context_list)

eval_scores = evaluator.evaluate(data, print_result=True)
eval_scores = evaluator.evaluate(data2, print_result=True)