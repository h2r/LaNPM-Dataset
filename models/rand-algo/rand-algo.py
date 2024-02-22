import json
import os
import random
import glob

def get_ground_truths():
    directory_path = '/home/ajaafar/h2r/NPM-Dataset/data/'
    actions = []
    # Use glob to find all JSON files in the directory
    file_pattern = os.path.join(directory_path, '*.json')
    for file_path in glob.glob(file_pattern):
        try:
            with open(file_path, 'r') as file:
                data = json.load(file)
                actions.append(data['steps'][0]['action'])
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from {file_path}: {e}")
        except Exception as e:
            print(f"Unexpected error with file {file_path}: {e}")
    return actions


actions = ["MoveRight", "MoveLeft", "MoveAhead", "MoveBack", "LookUp", "LookDown", "RotateRight", "RotateLeft"]
accuracy_lst = []
ground_truth_data = get_ground_truths()

for i in range(10):
    print(i+1)
    correct_predictions = 0

    for gt_action in ground_truth_data:
        predicted_action = random.choice(actions)
        
        if predicted_action == gt_action:
            correct_predictions += 1

    # Calculate the total accuracy
    accuracy = (correct_predictions / len(ground_truth_data)) * 100
    print(f"Accuracy: {accuracy}%")
    accuracy_lst.append(accuracy)


print(f"Avg accuracy: {sum(accuracy_lst)/len(accuracy_lst)}%")