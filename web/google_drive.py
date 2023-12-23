from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
import os

# Define the scope
SCOPES = ['https://www.googleapis.com/auth/drive']

# The path to your service account key file
SERVICE_ACCOUNT_FILE = 'key.json'

def service_account_login():
    credentials = service_account.Credentials.from_service_account_file(
            SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    return build('drive', 'v3', credentials=credentials)

def upload_file(service, file_path, file_name, folder_id):
    # Ensure that folder_id is not None and is a string
    if not isinstance(folder_id, str):
        raise ValueError("folder_id must be a string representing the Google Drive folder ID.")
    
    file_metadata = {
        'name': file_name,
        'parents': [folder_id]  # The ID of the shared folder
    }
    media = MediaFileUpload(file_path, mimetype='text/csv')
    file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
    print('File ID: %s' % file.get('id'))

def main():
    service = service_account_login()
    # The ID of the shared folder, this should be replaced with your actual shared folder ID
    folder_id = '18sVFbyUGcmnRavVPgJTiW2Pa5D3gzubp'
    # Path to the CSV file you want to upload, and the name you want it to have in Google Drive
    file_path = 'scene_commands.csv'
    file_name = 'scene_commands.csv'
    
    # Make sure the file_path exists and is a file
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"The file {file_path} was not found.")
    
    upload_file(service, file_path, file_name, folder_id)

if __name__ == '__main__':
    main()