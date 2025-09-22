"""

AWS Lambda Function Name: AiroAbscInvoiceProcessing

This function performs the core processing logic of ABSC invoices from the following vendors:
    1. ASIS
    2. DR
    3. LGBGood
    4. LGParts
    5. SGBGood
    6. SGParts
    7. SRA

Main Logic Steps:
    1. POST request sent via API Gateway
        - Inputs Args: string (file type), string (object key)
    2. The ZIP file is downloaded and sent to an S3 bucket (airo-invoice-processing/input)
    3. The ZIP file is retrieved by its key and processing is done on each PDF file within the ZIP, resulting in an Excel file
    4. Pre-signed URL is generated for the processed Excel file (for download), which is held in the S3 bucket (airo-invoice-outputs/output)
    5. Pre-signed URL is returned to the client

"""


print('Lambda function starting...')

import json
import boto3
import zipfile
import io
from datetime import datetime
import pandas as pd
import openpyxl
import utils.parser

# Establish the S3 Client
s3 = boto3.client('s3')

def lambda_handler(event, context):
    """
    Lambda handler for processing invoice ZIP files via API Gateway POST requests.
    
    Expected POST request body:
    {
        "filekey": "1234567890.zip",  # Timestamp-based unique filename
        "filetype": "ASIS"            # Type of invoice files to process
    }
    """
    print('Main starting...')
    try:
        # Print received event for debugging
        print("Received event:", json.dumps(event, indent=2))
            
        # Validate required fields in the raw event
        if 'filekey' not in event or 'filetype' not in event:
            return {
                'statusCode': 400,
                'body': json.dumps('Missing required fields: filekey and filetype')
            }
            
        # Set bucket names
        input_bucket = 'airo-invoice-processing'
        output_bucket = 'airo-invoice-outputs'
        
        # Construct the full input key (input/filename.zip)
        # Ensure the filekey ends with .zip
        filekey = event['filekey'] if event['filekey'].endswith('.zip') else f"{event['filekey']}.zip"
        object_key = f"input/{filekey}"
        print(f"Bucket: {input_bucket}, Key: {object_key}")
        
        # Retrieve the file from S3
        try:
            response = s3.get_object(Bucket=input_bucket, Key=object_key)
            print(f"Response: {response}")
            file_content = response['Body'].read()
            print(f"File size: {len(file_content)} bytes")
        except s3.exceptions.NoSuchKey:
            return {
                'statusCode': 404,
                'body': json.dumps(f"File not found: {object_key}")
            }
        
        # Return an error if not a ZIP file
        if not zipfile.is_zipfile(io.BytesIO(file_content)):
            return {
                'statusCode': 400,
                'body': json.dumps(f"Error: The file '{object_key}' is not a ZIP file")
            }
        
        # Process the list of PDF files into an Excel spreadsheet
        try: 
            output = utils.parser.zip_file_handler(file_content, event['filetype'])
        except ValueError as e:
            print(f"Error: {e}")
            return {
                'statusCode': 400,
                'body': json.dumps(f"Invalid file type: {e}")
            }
        except Exception as e:
            print(f"Error: {e}")
            return {
                'statusCode': 500,
                'body': json.dumps(f"Error processing the ZIP file: {str(e)}")
            }

        # In-memory buffer for Excel file
        print('Writing file to Excel...')
        try:
            excel_buffer = io.BytesIO()
        except Exception as e:
            print(f"Error: {e}")

        # Write the output to the buffer as an Excel file
        try:
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                output.to_excel(writer, index=False, sheet_name=event['filetype'].lower())
            excel_buffer.seek(0)
        except Exception as e:
            print(f"Error: {e}")

        # Write the key for the output file, which should be placed in the output/ folder
        dt = datetime.now().strftime('%m%d%y.%H%M%S')
        s3_output_key = f"output/{event['filetype'].upper()}_{dt}.xlsx"

        # Upload the Excel file to S3
        print('Uploading output to S3...')
        s3.put_object(Bucket=output_bucket, Key=s3_output_key, Body=excel_buffer.getvalue())

        # Delete the input file after processing is complete
        print('Deleting old input...')
        s3.delete_object(Bucket=input_bucket, Key=object_key)

        # Generate a pre-signed URL to download the processed Excel file
        print('Generating pre-signed URL...')
        presigned_url = s3.generate_presigned_url(
            'get_object',
            Params={
                'Bucket': output_bucket,
                'Key': s3_output_key
            },
            ExpiresIn=3600 # URL valid for 1 hour
        )
        print(f"Generated URL: {presigned_url}")

        # Return the pre-signed URL
        print('Returning pre-signed URL...')
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': f"Successfully processed {object_key}",
                'download_url': presigned_url
            })
        }
                


    # Return the error message for debugging
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps(f"Error processing the file: {str(e)}")
        }