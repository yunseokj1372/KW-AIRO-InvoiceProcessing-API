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
async def single_type_output(request: SingleTypeInput):  # Add async here
    try:
        file_content = request.fileKey  # Fixed to use proper Pydantic model access

    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Error: {e}")
    
    # Return an error if not a ZIP file
    if not zipfile.is_zipfile(io.BytesIO(file_content)):
        raise HTTPException(status_code=401, detail=f"Error: The file '{request.fileKey}' is not a ZIP file")
        
        
    try: 
        output = utils.parser.zip_file_handler(file_content, request.fileType)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid file type: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing the ZIP file: {str(e)}")
    
    # In-memory buffer for Excel file
    logger.info('Writing file to Excel...')  # Changed print to logger
    try:
        excel_buffer = io.BytesIO()
    except Exception as e:
        logger.error(f"Error creating BytesIO buffer: {e}")  # Changed print to logger
        raise HTTPException(status_code=501, detail=f"Error creating file buffer: {str(e)}")

    try:
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            output.to_excel(writer, index=False, sheet_name=request.fileType.lower())  # Fixed to use proper Pydantic model access
        excel_buffer.seek(0)
    except Exception as e:
        logger.error(f"Error writing Excel file: {e}")  # Changed print to logger
        raise HTTPException(status_code=502, detail=f"Error creating Excel file: {str(e)}")

    dt = datetime.now().strftime('%m%d%y.%H%M%S')
    
    # Create filename for the Excel file
    filename = f"invoice_output_{dt}.xlsx"
    
    # Return the Excel file as a StreamingResponse
    headers = {
        'Content-Disposition': f'attachment; filename="{filename}"',
        'Content-Type': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    }
    
    return StreamingResponse(
        excel_buffer,
        headers=headers,
        media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )
    






    