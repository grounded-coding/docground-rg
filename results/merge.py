import pandas as pd
import json
import os
import glob

# Step 1: Parse the CSV file results/score.csv with pandas. The columns team_id and entry_id are integers
scores = pd.read_csv('results/score.csv', dtype={'team_id': int, 'entry_id': int})

# Step 2: Parse JSON files
data_list = []

for team_dir in glob.glob('results/team*'):  # iterate over each team directory
    team_id = os.path.basename(team_dir)
    # Team ID is the last 2 digits of the directory name
    team_id = int(team_id[-2:])
    
    for metadata_file in glob.glob(f'{team_dir}/entry*.metadata.json'):  # iterate over metadata files
        entry_id = os.path.basename(metadata_file).split('.')[0]
        # Entry ID is the last digit of the file name
        entry_id = int(entry_id[-1:])
        
        # open and load json file
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        subtask3_data = metadata.get('subtask3', {})
        
        data_list.append({
            'team_id': team_id,
            'entry_id': entry_id,
            'pretrained': subtask3_data.get('pretrained', None),
            'external_api': subtask3_data.get('external_api', None),
            'ensemble': subtask3_data.get('ensemble', None),
            'desc': subtask3_data.get('desc', None),
        })

metadata_df = pd.DataFrame(data_list)

# Step 3: Now using keys team_id and entry_id, merge the remaining information
result = pd.merge(scores, metadata_df, on=['team_id', 'entry_id'])

# Write the result to result.csv
result.to_csv('results/result.csv', index=False)
