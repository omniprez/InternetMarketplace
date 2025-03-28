import cv2
import numpy as np
import pytesseract
import re
import logging
from PIL import Image
import os

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def preprocess_image(image_path):
    """
    Preprocess the image to enhance OCR accuracy
    """
    # Read the image
    img = cv2.imread(image_path)
    
    if img is None:
        raise ValueError(f"Could not read image from {image_path}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Apply dilation and erosion to remove noise
    kernel = np.ones((1, 1), np.uint8)
    img_dilation = cv2.dilate(thresh, kernel, iterations=1)
    img_erosion = cv2.erode(img_dilation, kernel, iterations=1)
    
    # Apply Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(img_erosion, (5, 5), 0)
    
    return blur

def extract_text(preprocessed_image):
    """
    Extract text from the preprocessed image using pytesseract
    """
    try:
        # Use pytesseract to extract text
        text = pytesseract.image_to_string(preprocessed_image)
        return text
    except Exception as e:
        logger.error(f"Error in OCR text extraction: {str(e)}")
        raise

def extract_table_data(image_path):
    """
    Extract table data from the image using pytesseract
    """
    # Read image and preprocess
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Use pytesseract's image_to_data function to get bounding box info
    custom_config = r'--oem 3 --psm 6'
    data = pytesseract.image_to_data(gray, config=custom_config, output_type=pytesseract.Output.DICT)
    
    # Convert image to pill format for more options
    pill_img = Image.open(image_path)
    
    # Use pytesseract to get table data
    table_data = pytesseract.image_to_data(pill_img)
    
    # Convert to list of dictionaries for easier processing
    table_data_list = []
    lines = table_data.split('\n')
    headers = lines[0].split('\t')
    
    for line in lines[1:]:
        cells = line.split('\t')
        if len(cells) == len(headers):
            row_dict = dict(zip(headers, cells))
            table_data_list.append(row_dict)
    
    return table_data_list

def parse_invoice_header(text):
    """
    Extract header information from the invoice text
    """
    header = {}
    
    # Extract invoice number
    invoice_match = re.search(r'Invoice #?:?\s*([A-Za-z0-9]+)', text) or re.search(r'Invoice No\.?:?\s*([A-Za-z0-9]+)', text)
    if invoice_match:
        header['invoice_number'] = invoice_match.group(1).strip()
    else:
        # Try to find by nearby text patterns
        invoice_match = re.search(r'IN(\d+)EDL(\d+)', text)
        if invoice_match:
            header['invoice_number'] = f"IN{invoice_match.group(1)}EDL{invoice_match.group(2)}"
        else:
            header['invoice_number'] = ""
    
    # Extract date
    date_match = re.search(r'Date:?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})', text) or re.search(r'Date:?\s*(\d{1,2}\s+[A-Za-z]+\s+\d{2,4})', text)
    if date_match:
        header['date'] = date_match.group(1).strip()
    else:
        header['date'] = ""
    
    # Extract customer reference
    cust_ref_match = re.search(r'Customer Ref\.?:?\s*([A-Za-z0-9]+)', text)
    if cust_ref_match:
        header['customer_ref'] = cust_ref_match.group(1).strip()
    else:
        header['customer_ref'] = ""
    
    # Extract customer name
    cust_name_match = re.search(r'Customer Name:?\s*(.+?)(?=\n|Address)', text, re.DOTALL)
    if cust_name_match:
        header['customer_name'] = cust_name_match.group(1).strip()
    else:
        header['customer_name'] = ""
    
    # Extract VAT registration number
    vat_reg_match = re.search(r'VAT Reg No\.?:?\s*([A-Za-z0-9]+)', text)
    if vat_reg_match:
        header['vat_reg_no'] = vat_reg_match.group(1).strip()
    else:
        header['vat_reg_no'] = ""
    
    # Extract business registration number
    business_reg_match = re.search(r'Business Reg No\.?:?\s*([A-Za-z0-9]+)', text)
    if business_reg_match:
        header['business_reg_no'] = business_reg_match.group(1).strip()
    else:
        header['business_reg_no'] = ""
    
    return header

def parse_line_items(text):
    """
    Extract line items from the invoice text
    """
    # Find the table section
    table_section_match = re.search(r'Product\s+Code.*?(?=Total \(Rs\)|GRAND TOTAL)', text, re.DOTALL | re.IGNORECASE)
    
    if not table_section_match:
        return []
    
    table_text = table_section_match.group(0)
    
    # Define pattern for line items
    # Looking for: Product Code, Quantity, Description, Unit Price, Discount, Total
    line_pattern = r'(\w+)\s+(\d+(?:\.\d+)?)\s+([A-Za-z0-9\s]+?)\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)'
    
    line_items = []
    for match in re.finditer(line_pattern, table_text):
        item = {
            'product_code': match.group(1).strip(),
            'quantity': match.group(2).strip(),
            'description': match.group(3).strip(),
            'unit_price': match.group(4).strip(),
            'discount': match.group(5).strip(),
            'total': match.group(6).strip()
        }
        line_items.append(item)
    
    # If the pattern didn't work well, fallback to line-by-line processing
    if not line_items:
        lines = table_text.strip().split('\n')
        for line in lines[1:]:  # Skip header row
            parts = line.strip().split()
            if len(parts) >= 6:
                try:
                    item = {
                        'product_code': parts[0],
                        'quantity': parts[1],
                        'description': ' '.join(parts[2:-3]),
                        'unit_price': parts[-3],
                        'discount': parts[-2],
                        'total': parts[-1]
                    }
                    line_items.append(item)
                except Exception as e:
                    logger.warning(f"Couldn't parse line item: {line} - {str(e)}")
    
    return line_items

def parse_totals(text):
    """
    Extract total values from the invoice text
    """
    totals = {}
    
    # Extract subtotal
    subtotal_match = re.search(r'Total \(Rs\)\s*(\d+(?:,\d+)*(?:\.\d+)?)', text)
    if subtotal_match:
        subtotal = subtotal_match.group(1).replace(',', '')
        totals['subtotal'] = subtotal
    else:
        totals['subtotal'] = ""
    
    # Extract VAT amount
    vat_match = re.search(r'VAT\s*@\s*\d+%\s*(\d+(?:,\d+)*(?:\.\d+)?)', text) or re.search(r'VAT\s*(\d+(?:,\d+)*(?:\.\d+)?)', text)
    if vat_match:
        vat = vat_match.group(1).replace(',', '')
        totals['vat'] = vat
    else:
        totals['vat'] = ""
    
    # Extract grand total
    grand_total_match = re.search(r'GRAND TOTAL\s*(\d+(?:,\d+)*(?:\.\d+)?)', text)
    if grand_total_match:
        grand_total = grand_total_match.group(1).replace(',', '')
        totals['grand_total'] = grand_total
    else:
        totals['grand_total'] = ""
    
    return totals

def apply_specific_rules(invoice_data, text):
    """
    Apply specific rules for known invoice formats
    """
    # Check if it's a Edendale Distributors invoice
    if 'Edendale Distributors' in text:
        # Special processing for Edendale invoices
        # Based on the sample invoice in the image
        
        # Extract invoice number format IN1234EDL5678
        invoice_match = re.search(r'IN(\d+)EDL(\d+)', text)
        if invoice_match:
            invoice_data['header']['invoice_number'] = f"IN{invoice_match.group(1)}EDL{invoice_match.group(2)}"
        
        # Extract line items - using known product code formats
        product_code_lines = re.findall(r'(5\d{7}|8\d{7}|9\d{7})\s+(\d+(?:\.\d+)?)\s+([A-Za-z0-9\s&]+?)\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)', text)
        
        if product_code_lines:
            invoice_data['line_items'] = []
            for line in product_code_lines:
                item = {
                    'product_code': line[0].strip(),
                    'quantity': line[1].strip(),
                    'description': line[2].strip(),
                    'unit_price': line[3].strip(),
                    'discount': line[4].strip(),
                    'total': line[5].strip()
                }
                invoice_data['line_items'].append(item)
                
        # Check for special VAT format
        if not invoice_data['totals']['vat']:
            vat_match = re.search(r'VAT\s*@\s*(\d+(?:,\d+)*(?:\.\d+)?)', text)
            if vat_match:
                invoice_data['totals']['vat'] = vat_match.group(1).replace(',', '')

    return invoice_data

def process_image(image_path):
    """
    Main function to process the invoice image and extract all required data
    """
    logger.info(f"Starting processing of image: {image_path}")
    
    try:
        # Check if file exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Preprocess the image
        preprocessed_image = preprocess_image(image_path)
        
        # Extract text using OCR
        text = extract_text(preprocessed_image)
        logger.debug(f"Extracted text: {text[:200]}...")  # Log first 200 chars of extracted text
        
        # Parse different sections of the invoice
        header = parse_invoice_header(text)
        line_items = parse_line_items(text)
        totals = parse_totals(text)
        
        # Combine all data
        invoice_data = {
            'header': header,
            'line_items': line_items,
            'totals': totals
        }
        
        # Apply specific rules for known invoice formats
        invoice_data = apply_specific_rules(invoice_data, text)
        
        logger.info(f"Successfully processed image and extracted data")
        
        return invoice_data
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise
