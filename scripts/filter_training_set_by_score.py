import json

def load_json_file(file_path):
    """Load and return the content of a JSON file."""
    with open(file_path, 'r') as file:
        return json.load(file)

def filter_indices_by_accuracy(ratings, min_accuracy=4.0):
    """Return a list of indices where accuracy ratings meet or exceed the minimum accuracy threshold."""
    return [rating['response_index'] for rating in ratings if rating['accurate'] >= min_accuracy]

def filter_data_by_indices(data, indices):
    """Filter and return data entries that correspond to the specified indices."""
    return [data[i] for i in range(len(data)) if i in indices]

def main(data_file_path_1, data_file_path_2, ratings_file_path):
    # Load the JSON data from the files
    data_1 = load_json_file(data_file_path_1)
    data_2 = load_json_file(data_file_path_2)
    ratings = load_json_file(ratings_file_path)
    
    # Filter indices by accuracy from the ratings
    filtered_indices = filter_indices_by_accuracy(ratings)
    
    # Filter the data based on the filtered indices
    filtered_data_1 = filter_data_by_indices(data_1, filtered_indices)
    filtered_data_2 = filter_data_by_indices(data_2, filtered_indices)
    
    return filtered_data_1, filtered_data_2

# Example usage
filtered_data_1, filtered_data_2 = main('data/train/labels.json', 'data/train/logs.json', '/u/nils.hilgers/setups/docground_eval/outputs/dstc11/train/humanref/geval4_turbo.json')

# Optionally, you can write the filtered data back to new JSON files
with open('data/train_filtered/labels.json', 'w') as file:
    json.dump(filtered_data_1, file, indent=4)

with open('data/train_filtered/logs.json', 'w') as file:
    json.dump(filtered_data_2, file, indent=4)
