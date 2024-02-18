import boto3

def save_csv_to_s3(csv_content, file_name):
    """
    Save CSV content to a file in an S3 bucket.
    
    :param csv_content: String representation of CSV data.
    :param bucket_name: The name of the S3 bucket.
    :param file_name: The S3 key under which to save the file.
    """
    # Initialize an S3 client
    s3_client = boto3.client('s3')
    
    # Upload the CSV content
    try:
        s3_client.put_object(Bucket="lanmp-participant-data", Key=file_name, Body=csv_content)
        print(f"File {file_name} saved to bucket.")
    except Exception as e:
        print(f"Error saving file to S3: {e}")
