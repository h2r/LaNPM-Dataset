import boto3
from botocore.exceptions import ClientError


dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
table = dynamodb.Table('userID')


def read_user_id(user_id):
    try:
        # Attempt to fetch the item by its primary key
        # response = table.get_item(Key={'user_id': user_id})
        response = table.get_item(Key={'user_id': user_id}, ConsistentRead=True)
        if 'Item' in response:
            # The item exists, return the associated value
            return response['Item']['value']
        else:
            # The item does not exist, initialize it with 0
            write_user_id(user_id, 0)  # Insert user_id with a value of 0
            return 0
    except ClientError as e:
        print(f"Error reading from DynamoDB: {e}")
        return None  # Or handle the error as needed

def write_user_id(user_id, new_value):
    try:
        # Write or update the item in the table
        table.put_item(
            Item={
                'user_id': user_id,   # Primary key column name
                'value': new_value    # The attribute/column where the user_id value is stored
            }
        )
    except ClientError as e:
        print(f"Error writing to DynamoDB: {e}")