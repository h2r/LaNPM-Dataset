from flask import Flask, request, jsonify, render_template, session
import pandas as pd
import drive_upload
import os
from dynamodb import generate_unique_id, insert_unique_id

application = Flask(__name__)
application.secret_key = 'af@93k$j392}a' 

@application.route('/')
def index():
    if 'user_id' not in session:
        unique_id = generate_unique_id()
        insert_unique_id(unique_id)
        session['user_id'] = unique_id
    else:
        unique_id = session['user_id']
    return render_template('index.html', user_id=unique_id)

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

@application.route('/submit', methods=['POST'])
def submit_data():
    scene_data = request.json

    if 'all_scene_data' not in session:
        session['all_scene_data'] = []

    session['all_scene_data'].append(scene_data)

    # Flask sessions are stored client-side in cookies by default,
    # so you must explicitly mark the session as modified to ensure it gets saved
    session.modified = True

    return jsonify({'message': 'Data received and stored in session.'})

@application.route('/save', methods=['POST'])
def save_data():
    scene_data = request.json
    session['all_scene_data'].append(scene_data)
    session.modified = True

    unique_id = session['user_id']

    all_data = session.get('all_scene_data', [])

    # Save the DataFrame as a CSV file with the name "commands_{number}.csv"
    csv_filename = f'commands_participant{unique_id}.csv'
    df = pd.DataFrame(all_data)
    df.to_csv(csv_filename, index=False)

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

    session.pop('all_scene_data', None)

    return jsonify({'message': 'Data saved successfully'})


if __name__ == '__main__':
    application.run()
