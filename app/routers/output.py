import json
import zipfile
import io
from datetime import datetime
import pandas as pd
import openpyxl
import utils.parser
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse  # Add this import
from pydantic import BaseModel
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Set up file logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('output.log'),
        logging.StreamHandler()  # Also print to console
    ]
)
logger = logging.getLogger(__name__)


router = APIRouter(prefix="/output", tags=["output"])

class SingleTypeInput(BaseModel):
    fileKey: str
    fileType: str


@router.post("/single")
async def single_type_output(request: SingleTypeInput):
    try:
        file_content = request.fileKey
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Error: {e}")
    
    # Return an error if not a ZIP file
    if not zipfile.is_zipfile(io.BytesIO(file_content)):
        raise HTTPException(status_code=401, detail=f"Error: The file '{request.fileKey}' is not a ZIP file")
        
    # Create a ThreadPoolExecutor for CPU-bound operations
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as pool:
        try:
            # Run zip_file_handler in a thread pool since it's CPU-bound
            output = await loop.run_in_executor(
                pool,
                lambda: utils.parser.zip_file_handler(file_content, request.fileType)
            )
            
            # Create Excel file asynchronously
            excel_buffer = io.BytesIO()
            await loop.run_in_executor(
                pool,
                lambda: create_excel_file(excel_buffer, output, request.fileType)
            )
            excel_buffer.seek(0)
            
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid file type: {e}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing the ZIP file: {str(e)}")

    dt = datetime.now().strftime('%m%d%y.%H%M%S')
    filename = f"invoice_output_{dt}.xlsx"
    
    headers = {
        'Content-Disposition': f'attachment; filename="{filename}"',
        'Content-Type': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    }
    
    return StreamingResponse(
        excel_buffer,
        headers=headers,
        media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )

def create_excel_file(buffer, df, file_type):
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name=file_type.lower())
    






    