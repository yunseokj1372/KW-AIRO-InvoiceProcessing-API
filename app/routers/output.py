"""
FastAPI Router for Invoice Processing

This router performs the core processing logic of ABSC invoices from the following vendors:
    1. ASIS
    2. DR
    3. LGBGood
    4. LGParts
    5. SGBGood
    6. SGParts
    7. SRA

Main Logic Steps:
    1. POST request with file key and file type
    2. The ZIP file is retrieved from S3 bucket (airo-invoice-processing/input)
    3. Processing is done on each PDF file within the ZIP, resulting in an Excel file
    4. Excel file is uploaded to S3 bucket (airo-invoice-outputs/output)
    5. Pre-signed URL is generated for download and returned to client
"""

import json
import boto3
import zipfile
import io
from datetime import datetime
import pandas as pd
import openpyxl
from app.utils import parser
from app.core.config import settings
from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from botocore.exceptions import ClientError, NoCredentialsError

# Set up file logging
import os

# Ensure the log directory exists
log_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'output.log')
os.makedirs(os.path.dirname(log_file), exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='a', encoding='utf-8'),
        logging.StreamHandler()  # Also print to console
    ]
)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/output", tags=["output"])

# API Key authentication
api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)

async def get_api_key(api_key: str = Depends(api_key_header)):
    if api_key != settings.api_key:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )
    return api_key

class SingleTypeInput(BaseModel):
    filekey: str    # Changed from fileKey to match Lambda function
    filetype: str   # Changed from fileType to match Lambda function

@router.post("/single")
async def single_type_output(request: SingleTypeInput, api_key: str = Depends(get_api_key)):
    """
    Process invoice ZIP files from S3, exactly matching Lambda function behavior.
    
    Expected request body:
    {
        "filekey": "1234567890.zip",  # Timestamp-based unique filename (S3 key)
        "filetype": "ASIS"            # Type of invoice files to process
    }
    """
    logger.info('Processing invoice request...')
    
    try:
        # Log received request for debugging
        logger.info(f"Received request: filekey={request.filekey}, filetype={request.filetype}")
        
        # Validate required fields
        if not request.filekey or not request.filetype:
            raise HTTPException(
                status_code=400,
                detail="Missing required fields: filekey and filetype"
            )
        
        # Set bucket names (matching Lambda function)
        input_bucket = settings.input_bucket
        output_bucket = settings.output_bucket
        
        # Construct the full input key (input/filename.zip)
        # Ensure the filekey ends with .zip
        filekey = request.filekey if request.filekey.endswith('.zip') else f"{request.filekey}.zip"
        object_key = f"input/{filekey}"
        logger.info(f"Bucket: {input_bucket}, Key: {object_key}")
        
        # Initialize S3 client
        try:
            # Let boto3 use the instance's IAM role automatically
            s3_client = boto3.client('s3')
        except NoCredentialsError:
            raise HTTPException(
                status_code=500,
                detail="AWS credentials not configured"
            )
        
        # Retrieve the file from S3
        try:
            response = s3_client.get_object(Bucket=input_bucket, Key=object_key)
            logger.info(f"S3 Response received")
            file_content = response['Body'].read()
            logger.info(f"File size: {len(file_content)} bytes")
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                raise HTTPException(
                    status_code=404,
                    detail=f"File not found: {object_key}"
                )
            else:
                raise HTTPException(
                    status_code=500,
                    detail=f"Error retrieving file from S3: {str(e)}"
                )
        
        # Return an error if not a ZIP file
        if not zipfile.is_zipfile(io.BytesIO(file_content)):
            raise HTTPException(
                status_code=400,
                detail=f"Error: The file '{object_key}' is not a ZIP file"
            )
        
        # Process the list of PDF files into an Excel spreadsheet
        logger.info('Processing ZIP file...')
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as pool:
            try:
                # Run zip_file_handler in a thread pool since it's CPU-bound
                output = await loop.run_in_executor(
                    pool,
                    lambda: parser.zip_file_handler(file_content, request.filetype)
                )
            except ValueError as e:
                logger.error(f"Invalid file type error: {e}")
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid file type: {e}"
                )
            except Exception as e:
                logger.error(f"Processing error: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Error processing the ZIP file: {str(e)}"
                )

        # In-memory buffer for Excel file
        logger.info('Writing file to Excel...')
        try:
            excel_buffer = io.BytesIO()
        except Exception as e:
            logger.error(f"Buffer creation error: {e}")
            raise HTTPException(status_code=500, detail=f"Error creating Excel buffer: {e}")

        # Write the output to the buffer as an Excel file
        try:
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                output.to_excel(writer, index=False, sheet_name=request.filetype.lower())
            excel_buffer.seek(0)
        except Exception as e:
            logger.error(f"Excel writing error: {e}")
            raise HTTPException(status_code=500, detail=f"Error writing Excel file: {e}")

        # Write the key for the output file, which should be placed in the output/ folder
        dt = datetime.now().strftime('%m%d%y.%H%M%S')
        s3_output_key = f"output/{request.filetype.upper()}_{dt}.xlsx"

        # Upload the Excel file to S3
        logger.info('Uploading output to S3...')
        try:
            s3_client.put_object(
                Bucket=output_bucket, 
                Key=s3_output_key, 
                Body=excel_buffer.getvalue()
            )
        except ClientError as e:
            logger.error(f"S3 upload error: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Error uploading to S3: {str(e)}"
            )

        # Delete the input file after processing is complete
        logger.info('Deleting old input...')
        try:
            s3_client.delete_object(Bucket=input_bucket, Key=object_key)
        except ClientError as e:
            logger.warning(f"Warning: Could not delete input file: {e}")
            # Don't fail the request if we can't delete the input file

        # Generate a pre-signed URL to download the processed Excel file
        logger.info('Generating pre-signed URL...')
        try:
            presigned_url = s3_client.generate_presigned_url(
                'get_object',
                Params={
                    'Bucket': output_bucket,
                    'Key': s3_output_key
                },
                ExpiresIn=3600  # URL valid for 1 hour
            )
            logger.info(f"Generated URL: {presigned_url}")
        except ClientError as e:
            logger.error(f"Presigned URL generation error: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Error generating download URL: {str(e)}"
            )

        # Return the pre-signed URL (matching Lambda function response)
        logger.info('Returning pre-signed URL...')
        return {
            'message': f"Successfully processed {object_key}",
            'download_url': presigned_url
        }
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        # Catch any other unexpected errors
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing the file: {str(e)}"
        )

    