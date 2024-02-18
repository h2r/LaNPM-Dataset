import boto3
from botocore.exceptions import ClientError
from nanoid import generate


dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
table = dynamodb.Table('userID')

def id_exists(unique_id):
    try:
        response = table.get_item(Key={'user_id': unique_id})
        return 'Item' in response
    except ClientError as e:
        print(e.response['Error']['Message'])
        return False

def generate_unique_id():
    while True:
        new_id = generate(size=10)
        if not id_exists(new_id):
            return new_id

def insert_unique_id(unique_id):
    try:
        table.put_item(
            Item={
                'user_id': unique_id,
            }
        )
    except ClientError as e:
        print(e.response['Error']['Message'])