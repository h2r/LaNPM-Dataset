import boto3

# Initialize a boto3 S3 client without specifying AWS credentials
# Make sure to specify the correct region your S3 bucket is in
s3_client = boto3.client('s3', region_name='us-east-1')

bucket_name = 'lanmp-data'  # Your S3 bucket name
file_key = 'user.txt'  # The path to your file within the S3 bucket

def check_and_create_file_if_not_exists():
    # Check if the file exists
    try:
        s3_client.head_object(Bucket=bucket_name, Key=file_key)
        print("File exists.")
    except:
        print("File does not exist. Creating the file with initial value 0.")
        # If the file does not exist, create it and write 0 to it
        s3_client.put_object(Bucket=bucket_name, Key=file_key, Body=b'0')

def read_user_id_from_s3():
    # Ensure the file exists
    check_and_create_file_if_not_exists()
    
    # Fetch the file from S3
    response = s3_client.get_object(Bucket=bucket_name, Key=file_key)
    
    # Read the file's content
    user_id = response['Body'].read().decode('utf-8').strip()
    return user_id

def write_user_id_to_s3(new_user_id):
    # Convert the new user ID to a string and encode it to bytes
    new_user_id_bytes = str(new_user_id).encode('utf-8')
    
    # Upload the new content to S3
    s3_client.put_object(Bucket=bucket_name, Key=file_key, Body=new_user_id_bytes)