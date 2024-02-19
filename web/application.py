from flask import Flask, request, jsonify, render_template, session
import pandas as pd
import os
from s3 import save_csv_to_s3

application = Flask(__name__)
application.secret_key = 'af@93k$j392}a' 

@application.route('/')
def index():  
    session['prolific_pid'] = request.args.get('PROLIFIC_PID', default=None)
    session['study_id'] = request.args.get('STUDY_ID', default=None)
    session['session_id'] = request.args.get('SESSION_ID', default=None)

    return render_template('index.html')

@application.route('/scenes')
def scenes():
    return render_template('scenes.html')

@application.route('/end')
def end():
    return render_template('end.html')

@application.route('/save', methods=['POST'])
def save_data():
    scene_data = request.json

    prolific_pid = session.get('prolific_pid', None)
    study_id = session.get('study_id', None)
    session_id = session.get('session_id', None)

    csv_filename = f'commands_participant_{prolific_pid}.csv'
    df = pd.DataFrame(scene_data)
    df.insert(0, 'session_id', session_id)
    df.insert(0, 'study_id', study_id)
    df.insert(0, 'prolific_pid', prolific_pid)
    df.to_csv(csv_filename, index=False)

    save_csv_to_s3(csv_filename)

    if os.path.exists(csv_filename):
        os.remove(csv_filename)
        print(f"{csv_filename} has been deleted.")

    #cleaning the session
    prolific_pid = session.pop('prolific_pid', None)
    study_id = session.pop('study_id', None)
    session_id = session.pop('session_id', None)

    return jsonify({'message': 'Data saved successfully'})


if __name__ == '__main__':
    application.run()