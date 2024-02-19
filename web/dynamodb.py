import boto3

dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
table = dynamodb.Table('userID')

def save_to_dynamodb(prolific_pid, study_id, session_id):
    try:
        response = table.put_item(
           Item={
                'prolific_pid': prolific_pid,
                'study_id': study_id,
                'session_id': session_id
            }
        )
        print("Save to DynamoDB successful")
    except Exception as e:
        print(f"Error saving to DynamoDB: {e}")