from flask import Flask, request, jsonify, render_template, session
import pandas as pd
import drive_upload
import os
from dynamodb import read_user_id, write_user_id

application = Flask(__name__)
application.secret_key = 'af@93k$j392}a' 

all_scene_data = []

@application.route('/')
def index():
    user_id = read_user_id(1)
    return render_template('index.html', user_id=user_id)

@application.route('/scene1')
def scene1():
    return render_template('scene1.html')

@application.route('/scene2')
def scene2():
    return render_template('scene2.html')

@application.route('/scene3')
def scene3():
    return render_template('scene3.html')

@application.route('/scene4')
def scene4():
    return render_template('scene4.html')

@application.route('/scene5')
def scene5():
    return render_template('scene5.html')

@application.route('/end')
def end():
    return render_template('end.html')

def append_scene_data(scene_data):
    all_scene_data.append(scene_data)
    return {'message': 'Data received successfully'}

@application.route('/submit', methods=['POST'])
def submit_data():
    scene_data = request.json
    result = append_scene_data(scene_data)
    return jsonify(result)

@application.route('/save', methods=['POST'])
def save_data():
    scene_data = request.json
    all_scene_data.append(scene_data)
    user_id = read_user_id(1)

    # Save the DataFrame as a CSV file with the name "commands_{number}.csv"
    csv_filename = f'commands_participant{user_id}.csv'
    df = pd.DataFrame(all_scene_data)
    df.to_csv(csv_filename, index=False)

    # Increment the number and update 'user.txt'
    new_user_id = int(user_id) + 1
    write_user_id(1, new_user_id)

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
    application.run()
