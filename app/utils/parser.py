import json
import zipfile
import io
import re
import os
import string
from typing import Dict, List, Optional, Tuple
import pandas as pd
import pymupdf
from abc import ABC, abstractmethod


class PDFProcessor(ABC):
    """Base class for PDF processing with common utilities."""
    
    def __init__(self):
        self.pdf_doc = None
        
    def open_pdf(self, pdf_file_path: str) -> None:
        """Opens a PDF file using PyMuPDF."""
        try:
            self.pdf_doc = pymupdf.open(pdf_file_path)
            print("Successfully opened file in PyMuPDF")
        except Exception as e:
            print(f"Error opening PDF: {e}")
            raise
            
    def validate_list_lengths(self, lists: List[List], error_msg: str = 'Not all lists are the same length!') -> None:
        """Validates that all lists have the same length."""
        if len({len(lst) for lst in lists}) != 1:
            raise ValueError(error_msg)
            
    def extract_table_data(self, page: pymupdf.Page, table_index: int) -> pd.DataFrame:
        """Extracts table data from a page."""
        tables = page.find_tables()
        return tables[table_index].to_pandas()
        
    def get_textbox(self, page: pymupdf.Page, rect: pymupdf.Rect) -> str:
        """Gets text from a specific rectangle on the page."""
        return page.get_textbox(rect)
        
    def process_model_numbers(self, text: str, split_char: str = '\n') -> List[str]:
        """Processes model numbers from text."""
        return [i.split('.', 1)[0] for i in text.split(split_char) if '.' in i]
        
    def clean_text(self, text: str) -> str:
        """Removes punctuation and whitespace."""
        return text.translate(str.maketrans('', '', string.punctuation)).lower().replace(' ', '').replace('\n', '')
        
    @abstractmethod
    def process_pdf(self, pdf_file_path: str) -> pd.DataFrame:
        """Process the PDF file and return a DataFrame."""
        pass


def zip_file_handler(zip_file_content: bytes, file_type_option: str) -> pd.DataFrame:
    """
    Process all PDF files in a ZIP archive using the appropriate processor.

    Arguments:
        zip_file_content: The binary file content of the ZIP file read from S3
        file_type_option: String name of the file type selected from the Budibase app menu (ex. ASIS, DR, etc.)
    
    Returns:
        A dataframe with the extracted data from each file in the ZIP file
    """
    output_df = []

    # Get the appropriate processor for this file type
    processor = get_processor(file_type_option)

    # Open the zip file for processing
    with zipfile.ZipFile(io.BytesIO(zip_file_content)) as main_zip:
        for file_name in main_zip.namelist():
            # Skip __MACOSX files
            if file_name.startswith('__MACOSX/._'):
                print(f"Skipping unwanted file: {file_name}")
                continue

            print(f"Processing {file_name}...")

            # Process the PDF file
            if file_name.lower().endswith('.pdf'):
                print(f"Found PDF: {file_name}") 

                with main_zip.open(file_name) as pdf_file:
                    # Read the PDF file
                    pdf_binary_data = pdf_file.read()
                    print(f"Successfully read {len(pdf_binary_data)} bytes from {file_name}")

                    # Write the PDF data to a temporary file
                    try:
                        temp_file_path = f'/tmp/{file_name}'
                        with open(temp_file_path, 'wb') as temp_file:
                            temp_file.write(pdf_binary_data)
                        print(f"PDF written to {temp_file_path}")
                    except Exception as e:
                        print(f"Error writing file: {e}")
                        continue

                    try:
                        # Process the file using the processor
                        single_pdf_output = processor.process_pdf(temp_file_path)
                        output_df.append(single_pdf_output)
                    except Exception as e:
                        print(f"Error processing file {file_name}: {e}")
                    finally:
                        # Clear PDF file from memory after processing
                        try:
                            os.remove(temp_file_path)
                            print(f"Deleted temporary file: {temp_file_path}")
                        except Exception as e:
                            print(f"Error deleting temporary file: {e}")

    return pd.concat(output_df, ignore_index=True) if output_df else pd.DataFrame()



def get_processor(file_type_option: str) -> PDFProcessor:
    """
    Get the appropriate PDF processor for the given file type.

    Arguments:
        file_type_option: The file type of PDFs to be processed

    Returns:
        An instance of the appropriate PDFProcessor subclass

    Raises:
        ValueError: If the file type is not supported
    """
    file_type_option = file_type_option.upper()
    
    processors = {
        'ASIS': ASISProcessor(),
        'DR': DRProcessor(),
        'LGBGOOD': LGBGoodProcessor(),
        'LGPARTS': LGPartsProcessor(),
        'SGBGOOD': SGBGoodProcessor(),
        'SGPARTS': SGPartsProcessor(),
        'SRA': SRAProcessor()
    }
    
    if file_type_option not in processors:
        raise ValueError(f"Unsupported file type: {file_type_option}")
        
    return processors[file_type_option]



class ASISProcessor(PDFProcessor):
    """Processor for ASIS PDF files."""
    
    COLUMNS = [
        'Biz Type', 'WH Code', 'Type\nIN/OUT/INOUT/\nOUTRETURN/INRETURN', 
        'Vendor Code', 'Customer\nCode', 'Order Date', 'Receive Date', 
        'Issue Date', 'TO NO', 'Ref No **\n(Invoice No)', 'Ref No 2', 
        'Memo', 'Shipper Company', 'ShipFrom\nCompany', 'ShipFrom\nAddress', 
        'ShipFrom\nCity', 'ShipFrom\nState', 'ShipFrom\nZip Code', 'ShipFrom\nCountry Code', 
        'ShipTo\nCompany', 'ShipTo\nAddress', 'ShipTo\nCity', 'ShipTo\nState', 
        'ShipTo\nZip Code', 'ShipTo\nCountry Code', 'Model No **', 
        'Item Serial No **\n(Original Ref No)', 'Model Description', 'Qty', ''
    ]
    
    GLOBAL_VALUES = {
        'Biz Type': 'AI',
        'WH Code': 'WHZZ',
        'Type\nIN/OUT/INOUT/\nOUTRETURN/INRETURN': 'IN',
        'Vendor Code': '103855',
        'Memo': 'LG AS-IS',
        'ShipFrom\nCompany': 'LG Electronics USA INC.',
        'ShipFrom\nAddress': '910 Sylvan Avenue',
        'ShipFrom\nCity': 'Englewood Cliffs',
        'ShipFrom\nState': 'NJ',
        'ShipFrom\nZip Code': '07632',
        'ShipFrom\nCountry Code': 'USA'
    }
    
    def process_pdf(self, pdf_file_path: str) -> pd.DataFrame:
        """Process ASIS PDF file and return extracted data."""
        print('asis process starting')
        
        output = pd.DataFrame(columns=self.COLUMNS)
        self.open_pdf(pdf_file_path)
        
        for page in self.pdf_doc:
            table_df = self.extract_table_data(page, 5)
            
            # Extract invoice number and date
            inv_num = self.extract_table_data(page, 1).columns[1]
            date = self.extract_table_data(page, 2).columns[1]
            
            # Process model numbers
            modelno_list = table_df.iloc[7, 0].split('\n')
            modelno_list = [modelno_list[i] for i in range(len(modelno_list)) if i % 2 == 0]
            modelno_list = [i.split('.', 1)[0] for i in modelno_list]
            
            # Extract other data
            poref_num = table_df['P.O./ REF. NO.'][0]
            cp_num = table_df.iloc[3, 4]
            shipq_list = table_df.iloc[7, 11].split('\n')
            up_list = table_df.iloc[7, 13].split('\n')
            
            # Validate list lengths
            self.validate_list_lengths([modelno_list, shipq_list, up_list])
            
            # Populate output
            for i in range(len(modelno_list)):
                output = output._append({
                    'Order Date': date, 'Receive Date': date, 'TO NO': cp_num,
                    'Ref No **\n(Invoice No)': poref_num, 'Ref No 2': inv_num,
                    'Model No **': modelno_list[i], 'Item Serial No **\n(Original Ref No)': poref_num,
                    'Qty': shipq_list[i], '': up_list[i]
                }, ignore_index=True)
            print(f"Models added to output: {len(modelno_list)}")
        
        # Add global values
        for key, value in self.GLOBAL_VALUES.items():
            output[key] = value
            
        return output


def asis_pdf_to_excel(pdf_file_path: str) -> pd.DataFrame:
    """Wrapper function for backward compatibility."""
    processor = ASISProcessor()
    return processor.process_pdf(pdf_file_path)



class DRProcessor(PDFProcessor):
    """Processor for DR (Defective Return) PDF files."""
    
    COLUMNS = ['INVOICE', 'Date', 'Model', 'RA #', 'QTY', 'Unit Price']
    
    def process_pdf(self, pdf_file_path: str) -> pd.DataFrame:
        """Process DR PDF file and return extracted data."""
        print(f'Processing PDF: {pdf_file_path}')
        
        output = pd.DataFrame(columns=self.COLUMNS)
        self.open_pdf(pdf_file_path)
        
        for page in self.pdf_doc:
            table_df = self.extract_table_data(page, 5)
            
            # Extract invoice number and date
            inv_num = self.extract_table_data(page, 1).columns[1]
            date = self.extract_table_data(page, 2).columns[1]
            
            # Process RA numbers
            ra_num = table_df.iloc[7, 3].split('\n')
            ra_num = [x for x in ra_num if x.isnumeric()]
            
            # Process model numbers
            modelno_list = table_df.iloc[7, 0].split('\n')
            if len(modelno_list) > len(ra_num):
                modelno_list = [x for x in modelno_list if '.' in x]
            modelno_list = [i.split('.', 1)[0] for i in modelno_list]
            
            # Extract quantities and unit prices
            shipq_list = table_df.iloc[7, 11].split('\n')
            up_list = table_df.iloc[7, 13].split('\n')
            
            # Validate list lengths
            self.validate_list_lengths([modelno_list, ra_num, shipq_list, up_list])
            
            # Populate output
            for i in range(len(modelno_list)):
                output = output._append({
                    'INVOICE': inv_num, 'Date': date, 'Model': modelno_list[i],
                    'RA #': ra_num[i], 'QTY': shipq_list[i], 'Unit Price': '$' + up_list[i]
                }, ignore_index=True)
        
        print(f'Completed processing PDF: {pdf_file_path}')
        return output


def dr_pdf_to_excel(pdf_file_path: str) -> pd.DataFrame:
    """Wrapper function for backward compatibility."""
    processor = DRProcessor()
    return processor.process_pdf(pdf_file_path)



class LGBGoodProcessor(PDFProcessor):
    """Processor for LG B-Good PDF files."""
    
    COLUMNS = ['W/H', 'INVOICE', 'Date', 'Model', 'RA #', 'QTY', 'Unit Price']
    
    WH_CODES = {
        'NTR': 'TX', 'NIR': 'IL', 'NFR': 'FL', 
        'NMR': 'NJ', 'NCR': 'CA', 'NQR': 'WA', 'NGR': 'GA'
    }
    
    def __init__(self):
        super().__init__()
        # Cache for table extractions to avoid repeated work
        self._table_cache = {}
    
    def get_warehouse_state(self, code: str) -> str:
        """Get state from warehouse code."""
        return self.WH_CODES.get(code, 'Unknown')
    
    def process_pdf(self, pdf_file_path: str) -> pd.DataFrame:
        """Process LG B-Good PDF file and return extracted data."""
        print(f"Processing PDF: {pdf_file_path}")
        
        # Clear cache for fresh processing
        self._table_cache.clear()
        
        # Use list to collect rows then create DataFrame once (much faster than _append)
        rows = []
        self.open_pdf(pdf_file_path)
        
        for page_idx, page in enumerate(self.pdf_doc):
            # Cache key for this page
            cache_key = f"page_{page_idx}"
            
            # Extract all needed tables at once and cache them
            if cache_key not in self._table_cache:
                tables = list(page.find_tables())  # Convert to list
                if len(tables) >= 6:
                    self._table_cache[cache_key] = {
                        'main': tables[5].to_pandas(),
                        'invoice': tables[1].to_pandas(), 
                        'date': tables[2].to_pandas()
                    }
                else:
                    continue
            
            # Use cached tables
            table_df = self._table_cache[cache_key]['main']
            
            # Extract invoice number and date from column headers (tables have 0 rows)
            invoice_table = self._table_cache[cache_key]['invoice']
            date_table = self._table_cache[cache_key]['date']
            
            # Get invoice number and date from column headers
            inv_num = invoice_table.columns[1] if len(invoice_table.columns) > 1 else ""
            date = date_table.columns[1] if len(date_table.columns) > 1 else ""
            
            # Extract warehouse code (same logic)
            wh_code = table_df.iloc[3, 14].split('/')[0]
            wh = self.get_warehouse_state(wh_code)
            
            # Process model numbers (same logic)
            modelno_list = table_df.iloc[7, 0].split('\n')
            modelno_list = [i.split('.', 1)[0] for i in modelno_list if '.' in i]
            
            # Process RA numbers (same logic)
            rano_list = table_df.iloc[7, 3].split('\n')
            rano_list = [rano_list[i] for i in range(len(rano_list)) if i % 2 != 0]
            
            # Extract quantities and unit prices (same logic)
            shipq_list = table_df.iloc[7, 11].split('\n')
            up_list = table_df.iloc[7, 13].split('\n')
            
            # Validate list lengths (same logic)
            self.validate_list_lengths([modelno_list, shipq_list, up_list])
            
            # Collect rows instead of appending to DataFrame
            for i in range(len(modelno_list)):
                rows.append({
                    'W/H': wh, 'INVOICE': inv_num, 'Date': date, 'Model': modelno_list[i],
                    'RA #': rano_list[i], 'QTY': shipq_list[i], 'Unit Price': up_list[i]
                })
        
        # Create DataFrame once from all rows (much faster than repeated _append)
        return pd.DataFrame(rows, columns=self.COLUMNS)


def lgbgood_pdf_to_excel(pdf_file_path: str) -> pd.DataFrame:
    """Wrapper function for backward compatibility."""
    processor = LGBGoodProcessor()
    return processor.process_pdf(pdf_file_path)



def lgparts_pdf_to_excel(pdf_file_path):
    """
    Arguments:
        - pdf_file_path: A filepath to the PDF file stored in temporary memory\
        
    Returns:
        - output: A dataframe with the extracted data for a single PDF file
    """

    print('lgparts process starting')

    # Initialize the final output df
    output = pd.DataFrame(columns=[
        'Biz Type', 'WH Code', 'Type\nIN/OUT/INOUT/\nOUTRETURN/INRETURN', 'Vendor Code', 'Customer Code', 
        'Order Date', 'Receive Date', 'Issue Date', 'TO NO', 'Ref No **\n(Invoice No)', 'Ref No 2', 'Memo', 
        'Shipper Company', 'ShipFrom\nCompany', 'ShipFrom\nAddress', 'ShipFrom\nCity', 'ShipFrom\nState', 
        'ShipFrom\nZip Code', 'ShipFrom\nCountry Code', 'ShipTo\nCompany', 'ShipTo\nAddress', 'ShipTo\nCity', 
        'ShipTo\nState', 'ShipTo\nZip Code', 'ShipTo\nCountry Code', 'Model No **', 'Item Serial No **\n(Original Ref No)', 
        'Model Description', 'Qty', 'ITEM CODE (model, serial are not needed)', 'Unit Price'
    ])

    # Open the PDF using PyMuPDF
    try:
        pdf = pymupdf.open(pdf_file_path)
        print("Successfully opened file in PyMuPDF")
    except Exception as e:
        print(f"Error opening PDF: {e}")
        return pd.DataFrame()

    # Define the customer codes mapping
    cust_codes_df = pd.DataFrame({
        'Customer Code': [...],  # List of codes
        'Code': [...]  # Corresponding values
    }).set_index('Customer Code')

    # Define text extraction rectangles
    invnum_rect = pymupdf.Rect(500, 100, 580, 117)
    invdate_rect = pymupdf.Rect(500, 117.5, 580, 132)

    for page in pdf:
        tables = page.find_tables()
        table_df = tables[0].to_pandas()

        ref_num = table_df.iloc[6, 3]
        cc_idx = next((i for i, chr in enumerate(ref_num) if chr.isdigit()), len(ref_num))
        cc_char = ref_num[:cc_idx]
        cust_code = cust_codes_df.loc[cc_char, 'Code']

        st_calist = table_df.iloc[0, 11].split('\n')
        st_company, st_address = st_calist[0], st_calist[1]
        st_city = table_df.iloc[3, 11]
        st_state, st_zip = table_df.iloc[3, 17].split(' ')

        or_date = ref_num.replace(cc_char, '')[:6] if '-' not in ref_num else ref_num.split('-')[0]
        or_datef = pd.to_datetime(or_date[-6:], format='%m%d%y').strftime('%m/%d/%Y')

        inv_num = page.get_textbox(invnum_rect)
        inv_date = page.get_textbox(invdate_rect)

        modelno_list = re.sub(r'[^\w\s]', '', re.sub(r'\([^)]*\)', '', table_df.iloc[10, 0])).split('\n')
        shipq_list = table_df.iloc[10, 10].split('\n')
        up_list = table_df.iloc[10, 12].split('\n')

        if len({len(modelno_list), len(shipq_list), len(up_list)}) != 1:
            raise ValueError('Not all lists are the same length!')

        for i in range(len(modelno_list)):
            output = output._append({
                'Customer Code': cust_code, 'Order Date': or_datef, 'Receive Date': inv_date, 'Issue Date': inv_date, 
                'Ref No **\n(Invoice No)': ref_num, 'Ref No 2': inv_num, 'ShipTo\nCompany': st_company, 
                'ShipTo\nAddress': st_address, 'ShipTo\nCity': st_city, 'ShipTo\nState': st_state, 
                'ShipTo\nZip Code': st_zip, 'Model No **': modelno_list[i], 
                'Item Serial No **\n(Original Ref No)': ref_num, 'Qty': shipq_list[i], 'Unit Price': '$' + up_list[i]
            }, ignore_index=True)

    output['Biz Type'] = 'LPT'
    output['WH Code'] = 'WHZZ'
    output['Type\nIN/OUT/INOUT/\nOUTRETURN/INRETURN'] = 'INOUT'
    output['Vendor Code'] = '104256'
    output['Memo'] = 'LG PARTS'
    output['ShipFrom\nCompany'] = 'LG ELECTRONICS ALABAMA INC'
    output['ShipFrom\nAddress'] = '201 James Record Road - Bldg 3'
    output['ShipFrom\nCity'] = 'Huntsville'
    output['ShipFrom\nState'] = 'AL'
    output['ShipFrom\nZip Code'] = '35824'
    output['ShipFrom\nCountry Code'] = 'USA'

    return output



class LGPartsProcessor(PDFProcessor):
    """Processor for LG Parts PDF files."""
    
    COLUMNS = [
        'Biz Type', 'WH Code', 'Type\nIN/OUT/INOUT/\nOUTRETURN/INRETURN', 'Vendor Code', 'Customer Code', 
        'Order Date', 'Receive Date', 'Issue Date', 'TO NO', 'Ref No **\n(Invoice No)', 'Ref No 2', 'Memo', 
        'Shipper Company', 'ShipFrom\nCompany', 'ShipFrom\nAddress', 'ShipFrom\nCity', 'ShipFrom\nState', 
        'ShipFrom\nZip Code', 'ShipFrom\nCountry Code', 'ShipTo\nCompany', 'ShipTo\nAddress', 'ShipTo\nCity', 
        'ShipTo\nState', 'ShipTo\nZip Code', 'ShipTo\nCountry Code', 'Model No **', 'Item Serial No **\n(Original Ref No)', 
        'Model Description', 'Qty', 'ITEM CODE (model, serial are not needed)', 'Unit Price'
    ]
    
    GLOBAL_VALUES = {
        'Biz Type': 'LPT',
        'WH Code': 'WHZZ',
        'Type\nIN/OUT/INOUT/\nOUTRETURN/INRETURN': 'INOUT',
        'Vendor Code': '104256',
        'Memo': 'LG PARTS',
        'ShipFrom\nCompany': 'LG ELECTRONICS ALABAMA INC',
        'ShipFrom\nAddress': '201 James Record Road - Bldg 3',
        'ShipFrom\nCity': 'Huntsville',
        'ShipFrom\nState': 'AL',
        'ShipFrom\nZip Code': '35824',
        'ShipFrom\nCountry Code': 'USA'
    }
    
    CUSTOMER_CODES = pd.DataFrame({
        'Customer Code': [
            'NA', 'FL', 'STA', 'NW', 'WA', 'EA', 'LY', 'SA', 'AD', 'AMP', 'BB', 'IHG', 'OCY', 'FT', 'KG', 'ID', 'BI', 'YB', 
            'JJ', 'WEA', 'DLP', 'EL', 'TRI', 'RLP', 'ZQN', 'WLA', 'AMK', 'END', 'EN', 'WL', 'AWO', 'DSC', 'GRW', 'TD', 'KW'
        ],
        'Code': [
            '102299', '102300', '102301', '102302', '102303', '102304', '102305', '102073', '102311', '101865', '102180', 
            '102150', '101552', '101741', '102310', '102218', '101975', '100249', '101695', '102256', '102401', '101880', 
            '102500', '102490', '102543', '102579', '100165', '102604', '102604', '102579', '100933', '102025', '102687', '100254', '1400'
        ]
    }).set_index('Customer Code')
    
    TEXT_BOXES = {
        'invoice_number': pymupdf.Rect(500, 100, 580, 117),
        'invoice_date': pymupdf.Rect(500, 117.5, 580, 132)
    }
    
    def get_customer_code(self, ref_num: str) -> str:
        """Extract customer code from reference number."""
        cc_idx = next((i for i, chr in enumerate(ref_num) if chr.isdigit()), len(ref_num))
        cc_char = ref_num[:cc_idx]
        return self.CUSTOMER_CODES.loc[cc_char, 'Code']
    
    def format_date(self, ref_num: str, cc_char: str) -> str:
        """Format order date from reference number."""
        if '-' in ref_num:
            or_date = ref_num.split('-')[0]
        else:
            or_date = ref_num.replace(cc_char, '')[:6]
        
        or_date = or_date[len(or_date) - 6:]
        or_date = re.findall(r'\d+', or_date)
        return pd.to_datetime(or_date, format='%m%d%y').strftime('%m/%d/%Y')
    
    def process_pdf(self, pdf_file_path: str) -> pd.DataFrame:
        """Process LG Parts PDF file and return extracted data."""
        print('lgparts process starting')
        
        output = pd.DataFrame(columns=self.COLUMNS)
        self.open_pdf(pdf_file_path)
        
        for page in self.pdf_doc:
            table_df = self.extract_table_data(page, 0)
            
            # Extract reference number and customer code
            ref_num = table_df.iloc[6, 3]
            cc_idx = next((i for i, chr in enumerate(ref_num) if chr.isdigit()), len(ref_num))
            cc_char = ref_num[:cc_idx]
            cust_code = self.get_customer_code(ref_num)
            
            # Extract shipping information
            st_calist = table_df.iloc[0, 11].split('\n')
            st_company, st_address = st_calist[0], st_calist[1]
            st_city = table_df.iloc[3, 11]
            st_state, st_zip = table_df.iloc[3, 17].split(' ')
            
            # Process dates
            or_datef = self.format_date(ref_num, cc_char)
            inv_num = self.get_textbox(page, self.TEXT_BOXES['invoice_number'])
            inv_date = self.get_textbox(page, self.TEXT_BOXES['invoice_date'])
            
            # Process model numbers and quantities
            modelno_list = re.sub(r'[^\w\s]', '', re.sub(r'\([^)]*\)', '', table_df.iloc[10, 0])).split('\n')
            shipq_list = table_df.iloc[10, 10].split('\n')
            up_list = table_df.iloc[10, 12].split('\n')
            
            # Validate list lengths
            self.validate_list_lengths([modelno_list, shipq_list, up_list])
            
            # Populate output
            for i in range(len(modelno_list)):
                output = output._append({
                    'Customer Code': cust_code, 'Order Date': or_datef, 'Receive Date': inv_date,
                    'Issue Date': inv_date, 'Ref No **\n(Invoice No)': ref_num, 'Ref No 2': inv_num,
                    'ShipTo\nCompany': st_company, 'ShipTo\nAddress': st_address, 'ShipTo\nCity': st_city,
                    'ShipTo\nState': st_state, 'ShipTo\nZip Code': st_zip, 'Model No **': modelno_list[i],
                    'Item Serial No **\n(Original Ref No)': ref_num, 'Qty': shipq_list[i],
                    'Unit Price': '$' + up_list[i]
                }, ignore_index=True)
        
        # Add global values
        for key, value in self.GLOBAL_VALUES.items():
            output[key] = value
            
        return output


def lgparts_pdf_to_excel(pdf_file_path: str) -> pd.DataFrame:
    """Wrapper function for backward compatibility."""
    processor = LGPartsProcessor()
    return processor.process_pdf(pdf_file_path)



class SGBGoodProcessor(PDFProcessor):
    """Processor for Samsung B-Good PDF files."""
    
    COLUMNS = [
        'Biz Type', 'WH Code', 'Type\nIN/OUT/INOUT', 'Vendor Code', 'Customer Code', 'Order Date', 
        'Receive Date', 'Sales Date', 'TO NO', 'Ref No **\n(Invoice No)', 'Ref No 2', 'Memo', 
        'Shipper Company', 'ShipFrom\nCompany', 'ShipFrom\nAddress', 'ShipFrom\nCity', 'ShipFrom\nState', 
        'ShipFrom\nZip Code', 'ShipFrom\nCountry Code', 'ShipTo\nCompany', 'ShipTo\nAddress', 
        'ShipTo\nCity', 'ShipTo\nState', 'ShipTo\nZip Code', 'ShipTo\nCountry Code', 'Model No', 
        'Item Serial No **\n(Original Ref No)', 'Model Description', 'Qty', ''
    ]
    
    GLOBAL_VALUES = {
        'Biz Type': 'SBG',
        'WH Code': 'WHZZ',
        'Type\nIN/OUT/INOUT': 'IN',
        'Vendor Code': '101317',
        'Memo': 'SAMSUNG B GOODS',
        'ShipFrom\nCompany': 'SAMSUNG ELECTRONIC AMERICA, INC',
        'ShipFrom\nAddress': '13034 Collections Center Drive',
        'ShipFrom\nCity': 'CHICAGO',
        'ShipFrom\nState': 'IL',
        'ShipFrom\nZip Code': '60693',
        'ShipFrom\nCountry Code': 'USA'
    }
    
    def process_pdf(self, pdf_file_path: str) -> pd.DataFrame:
        """Process Samsung B-Good PDF file and return extracted data."""
        print('sgbgood process starting')
        
        output = pd.DataFrame(columns=self.COLUMNS)
        self.open_pdf(pdf_file_path)
        
        for page in self.pdf_doc:
            table_df = self.extract_table_data(page, 0)
            
            # Extract dates and invoice numbers
            or_date = table_df['INVOICE DATE'][0]
            ref_num = table_df.iloc[11, 0].split(': ')[1].replace(' ', '')
            inv_num = table_df['INVOICE NUMBER'][0]
            
            # Process model numbers and quantities
            modelno_list = table_df.iloc[10, 0].split('\n')
            shipq_list = table_df.iloc[10, 2].split('\n')
            nup_list = table_df.iloc[10, 12].split('\n')
            
            # Validate list lengths
            self.validate_list_lengths([modelno_list, shipq_list, nup_list])
            
            # Populate output
            for i in range(len(modelno_list)):
                output = output._append({
                    'Order Date': or_date, 'Receive Date': or_date,
                    'Ref No **\n(Invoice No)': ref_num, 'Ref No 2': inv_num,
                    'Model No': modelno_list[i], 'Item Serial No **\n(Original Ref No)': ref_num,
                    'Qty': shipq_list[i], '': nup_list[i]
                }, ignore_index=True)
        
        # Add global values
        for key, value in self.GLOBAL_VALUES.items():
            output[key] = value
            
        return output


def sgbgood_pdf_to_excel(pdf_file_path: str) -> pd.DataFrame:
    """Wrapper function for backward compatibility."""
    processor = SGBGoodProcessor()
    return processor.process_pdf(pdf_file_path)



class SGPartsProcessor(PDFProcessor):
    """Processor for Samsung Parts PDF files."""
    
    COLUMNS = [
        'Biz Type', 'WH Code', 'Type\nIN/OUT/INOUT/\nOUTRETURN/INRETURN', 'Vendor Code', 'Customer Code', 
        'Order Date', 'Receive Date', 'Issue Date', 'TO NO', 'Ref No **\n(Invoice No)', 
        'Ref No 2', 'Memo', 'Shipper Company', 'ShipFrom\nCompany', 'ShipFrom\nAddress', 
        'ShipFrom\nCity', 'ShipFrom\nState', 'ShipFrom\nZip Code', 'ShipFrom\nCountry Code', 
        'ShipTo\nCompany', 'ShipTo\nAddress', 'ShipTo\nCity', 'ShipTo\nState', 'ShipTo\nZip Code', 
        'ShipTo\nCountry Code', 'Model No **', 'Item Serial No **\n(Original Ref No)', 
        'Model Description', 'Qty', 'Unit Price'
    ]
    
    GLOBAL_VALUES = {
        'Biz Type': 'SPT',
        'WH Code': 'WHZZ',
        'Type\nIN/OUT/INOUT/\nOUTRETURN/INRETURN': 'IN',
        'Vendor Code': '100484',
        'Memo': 'SAMSUNG PART',
        'ShipFrom\nCompany': 'Global Parts Center America',
        'ShipFrom\nAddress': '18600 Broadwick Street',
        'ShipFrom\nCity': 'Rancho Dominguez',
        'ShipFrom\nState': 'CA',
        'ShipFrom\nZip Code': '90220',
        'ShipFrom\nCountry Code': 'USA'
    }
    
    CUSTOMER_CODES = {
        'northernappliances': '102299',
        'flhappyappliances': '102300',
        'southernappliances': '102301',
        'northwesternappliances': '102302',
        'westernappliances': '102303',
        'easternappliances': '102304',
        'lyappliances': '102305',
        'bbappliancesinc': '102180',
        'frontier': '101741',
        'ada': '102311',
        'jeff': '101865',
        'ihg': '102150',
        'ocyinc': '101552',
        'superiorappliance': '102073',
        'kagemushaexpresscorp': '102310',
        'interglobaldistributorsinc': '102218',
        'b612inc': '101975',
        'ybs': '100249',
        'jjappliancesllc': '101695',
        'wena': '102256',
        'youngchoidbaendlesstradinginc': '102343',
        'dlp': '102401',
        'elecdepot': '101880',
        'reallowprice': '102490',
        'amkotroninc': '100165',
        'tristarsinc': '102500',
        'zqian': '102543',
        'awo': '100933',
        'wonderfullifeappliances': '102579'
    }
    
    TEXT_BOXES = {
        'invoice_date': pymupdf.Rect(466, 91, 529, 104),
        'invoice_number': pymupdf.Rect(370, 91.5, 440, 104),
        'reference_number': pymupdf.Rect(32, 275, 111, 623),
        'model_number_base': pymupdf.Rect(120, 283.853, 193, 285.907),
        'ship_quantity': pymupdf.Rect(390, 275, 430, 623),
        'unit_price': pymupdf.Rect(431, 275, 476, 623),
        'customer_code': pymupdf.Rect(345, 160, 530, 172)
    }
    
    def get_customer_code(self, page: pymupdf.Page) -> str:
        """Extract and map customer code from page."""
        customer = self.get_textbox(page, self.TEXT_BOXES['customer_code'])
        customer = self.clean_text(customer)
        return self.CUSTOMER_CODES.get(customer)
    
    def get_model_number_rect(self, base_rect: pymupdf.Rect, index: int) -> pymupdf.Rect:
        """Calculate model number rectangle for given index."""
        y_offset = 42.571 * index
        return pymupdf.Rect(
            base_rect.x0,
            base_rect.y0 + y_offset,
            base_rect.x1,
            base_rect.y0 + y_offset + 2.054
        )
    
    def process_pdf(self, pdf_file_path: str) -> pd.DataFrame:
        """Process Samsung Parts PDF file and return extracted data."""
        print('sgparts process starting')
        
        output = pd.DataFrame(columns=self.COLUMNS)
        self.open_pdf(pdf_file_path)
        
        for page in self.pdf_doc:
            # Extract invoice information
            inv_date = self.get_textbox(page, self.TEXT_BOXES['invoice_date'])
            inv_datef = pd.to_datetime(inv_date).strftime('%m/%d/%Y')
            inv_no = self.get_textbox(page, self.TEXT_BOXES['invoice_number'])
            
            # Get customer code
            cust_code = self.get_customer_code(page)
            
            # Extract reference numbers
            refno_list = self.get_textbox(page, self.TEXT_BOXES['reference_number']).split('\n')
            refno_list = [x for x in refno_list if x.strip()]
            refno_list = [refno_list[i] for i in range(len(refno_list)) if i % 2 == 0]
            
            # Extract quantities and prices
            shipq_list = [x for x in self.get_textbox(page, self.TEXT_BOXES['ship_quantity']).split('\n') if x.strip()]
            up_list = [x for x in self.get_textbox(page, self.TEXT_BOXES['unit_price']).split('\n') if x.strip()]
            up_list = [up_list[i] for i in range(len(up_list)) if i % 2 == 0]
            
            # Validate list lengths
            self.validate_list_lengths([refno_list, shipq_list, up_list])
            
            # Populate output
            for i in range(len(shipq_list)):
                model_rect = self.get_model_number_rect(self.TEXT_BOXES['model_number_base'], i)
                modelno = self.get_textbox(page, model_rect).strip()
                
                output = output._append({
                    'Customer Code': cust_code, 'Order Date': inv_datef, 'Receive Date': inv_datef,
                    'Ref No **\n(Invoice No)': refno_list[i], 'Ref No 2': inv_no,
                    'Model No **': modelno, 'Item Serial No **\n(Original Ref No)': refno_list[i],
                    'Qty': shipq_list[i], 'Unit Price': up_list[i]
                }, ignore_index=True)
        
        # Add global values
        for key, value in self.GLOBAL_VALUES.items():
            output[key] = value
            
        return output


def sgparts_pdf_to_excel(pdf_file_path: str) -> pd.DataFrame:
    """Wrapper function for backward compatibility."""
    processor = SGPartsProcessor()
    return processor.process_pdf(pdf_file_path)



class SRAProcessor(PDFProcessor):
    """Processor for SRA PDF files."""
    
    COLUMNS = [
        'Business type', 'PO#', 'Invoice NO', 'INV DATE', 'MODEL#', 
        'SHIPPED QUANTITY', 'NET UNIT PRICE', 'NET AMOUNT USD'
    ]
    
    GLOBAL_VALUES = {
        'Business type': 'SRA'
    }
    
    def process_pdf(self, pdf_file_path: str) -> pd.DataFrame:
        """Process SRA PDF file and return extracted data."""
        print('sra process starting')
        
        output = pd.DataFrame(columns=self.COLUMNS)
        self.open_pdf(pdf_file_path)
        
        for page in self.pdf_doc:
            table_df = self.extract_table_data(page, 0)
            
            # Extract invoice information
            po_num = table_df.iloc[2, 1]
            inv_num = table_df['INVOICE NUMBER'][0]
            inv_date = table_df['INVOICE DATE'][0]
            
            # Process model numbers and quantities
            modelno_list = table_df.iloc[10, 0].split('\n')
            shipq_list = table_df.iloc[10, 2].split('\n')
            netup_list = table_df.iloc[10, 12].split('\n')
            nausd_list = table_df.iloc[10, 14].split('\n')
            
            # Validate list lengths
            self.validate_list_lengths([modelno_list, shipq_list, netup_list, nausd_list])
            
            # Populate output
            for i in range(len(modelno_list)):
                output = output._append({
                    'PO#': po_num, 'Invoice NO': inv_num, 'INV DATE': inv_date,
                    'MODEL#': modelno_list[i], 'SHIPPED QUANTITY': shipq_list[i],
                    'NET UNIT PRICE': netup_list[i], 'NET AMOUNT USD': nausd_list[i]
                }, ignore_index=True)
        
        # Add global values
        for key, value in self.GLOBAL_VALUES.items():
            output[key] = value
            
        return output


def sra_pdf_to_excel(pdf_file_path: str) -> pd.DataFrame:
    """Wrapper function for backward compatibility."""
    processor = SRAProcessor()
    return processor.process_pdf(pdf_file_path)