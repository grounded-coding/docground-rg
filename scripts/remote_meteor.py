# remote_meteor.py

import json
import sys
from summ_eval.meteor_metric import MeteorMetric


def main():
    # Read the predicted and reference responses from the command line arguments
    pred_responses = json.loads(sys.argv[1])
    ref_responses = json.loads(sys.argv[2])

    # Create a MeteorMetric instance
    metric = MeteorMetric()

    # Compute the METEOR metric
    result = metric.evaluate_batch(pred_responses, ref_responses)

    # Print the METEOR metric so it can be read by the script on the local machine
    print(result['meteor'])


if __name__ == "__main__":
    main()
