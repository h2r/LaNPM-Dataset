import json
import os
import re
import random

def get_ground_truths():
    # Define the directory containing the JSON files (replace 'your_directory_path' with the actual path)
    directory_path = ''

    # Initialize a list to store the ground truth actions
    ground_truth_actions = []

    # Regular expression to match file names
    file_pattern = re.compile(r'^data_chunk_(\d{1,2})\.json$')

    # Iterate through files in the specified directory
    for file_name in os.listdir(directory_path):
        # Check if the file name matches the pattern
        if file_pattern.match(file_name):
            # Construct the full file path
            file_path = os.path.join(directory_path, file_name)
            
            # Open and read the JSON file
            with open(file_path, 'r') as file:
                data = json.load(file)
                
                # Assuming each file contains an 'action' key at the root level
                action = data.get('action')
                if action:
                    ground_truth_actions.append(action)

    return ground_truth_actions



actions = ["MoveRight", "MoveLeft", "MoveAhead", "MoveBack", "LookUp", "LookDown", "RotateRight", "RotateLeft"]

ground_truth_data = get_ground_truths()

# Initialize a counter for the correct predictions
correct_predictions = 0

for gt_action in ground_truth_data:
    predicted_action = random.choice(actions)
    
    if predicted_action == gt_action:
        correct_predictions += 1

# Calculate the total accuracy
accuracy = (correct_predictions / len(ground_truth_data)) * 100


print(f"Total accuracy: {accuracy}%")