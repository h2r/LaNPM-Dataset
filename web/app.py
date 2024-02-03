from flask import Flask, request, jsonify, render_template
import pandas as pd
import drive_upload
import os


app = Flask(__name__)

all_scene_data = []

with open("user.txt", "r") as file:
        user_id = file.read().strip()

@app.route('/')
def index():
    return render_template('index.html', user_id=user_id)

@app.route('/scene1')
def scene1():
    return render_template('scene1.html')

@app.route('/scene2')
def scene2():
    return render_template('scene5.html') #change back to 2

@app.route('/end')
def end():
    return render_template('end.html')


@app.route('/submit', methods=['POST'])
def submit_data():
    scene_data = request.json
    all_scene_data.append(scene_data)
    return jsonify({'message': 'Data received successfully'})

@app.route('/save', methods=['POST'])
def save_data():
    scene_data = request.json
    all_scene_data.append(scene_data)

    # Save the DataFrame as a CSV file with the name "commands_{number}.csv"
    csv_filename = f'commands_participant{user_id}.csv'
    df = pd.DataFrame(all_scene_data)
    df.to_csv(csv_filename, index=False)

    # Increment the number and update 'user.txt'
    new_user_id = int(user_id) + 1
    with open('user.txt', 'w') as file:
        file.write(str(new_user_id))

    #upload csv to google drive shared folder
    service = drive_upload.service_account_login()
    folder_id = '18sVFbyUGcmnRavVPgJTiW2Pa5D3gzubp'
    file_path = csv_filename  
    file_name = csv_filename

    file_id = drive_upload.upload_file(service, file_path, file_name, folder_id)
    print(f"Uploaded file ID: {file_id}")

    if os.path.exists(file_name):
        os.remove(file_name)
        print(f"{file_name} has been deleted.")

    return jsonify({'message': 'Data saved successfully'})


if __name__ == '__main__':
    app.run(debug=True)
