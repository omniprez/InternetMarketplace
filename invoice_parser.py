import cv2
import numpy as np
import pytesseract
import re
import logging
from PIL import Image
import os
import math

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def preprocess_image(image_path):
    """
    Enhanced preprocessing to improve OCR accuracy
    """
    # Read the image
    img = cv2.imread(image_path)
    
    if img is None:
        raise ValueError(f"Could not read image from {image_path}")
    
    # Get image dimensions
    height, width = img.shape[:2]
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Check if the image needs to be deskewed
    # Calculate skew angle
    coords = np.column_stack(np.where(gray > 0))
    angle = cv2.minAreaRect(coords)[-1]
    
    # Adjust angle
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
        
    # Only deskew if the angle is significant
    if abs(angle) > 0.5:
        # Rotate the image to deskew it
        (h, w) = gray.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        gray = rotated
    
    # Apply adaptive thresholding - better for varying lighting conditions
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    # Apply morphological operations more aggressively for table structure
    # First remove small noise
    kernel = np.ones((1, 1), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # Then apply dilation to make text more pronounced
    dilation_kernel = np.ones((2, 2), np.uint8)
    dilation = cv2.dilate(opening, dilation_kernel, iterations=1)
    
    # Contrast stretching to enhance visibility
    stretched = cv2.normalize(dilation, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    
    return stretched

def extract_text(preprocessed_image):
    """
    Enhanced text extraction using Tesseract with optimized configuration
    """
    try:
        # Use multiple PSM modes to improve text extraction
        # PSM 4 - Assume a single column of text of variable sizes
        # PSM 6 - Assume a single uniform block of text
        # PSM 11 - Sparse text. Find as much text as possible in no particular order
        # PSM 12 - Sparse text with OSD
        
        # Configure Tesseract for optimal invoice recognition
        custom_config = r'--oem 3 --psm 6 -l eng --dpi 300'
        text_psm6 = pytesseract.image_to_string(preprocessed_image, config=custom_config)
        
        custom_config = r'--oem 3 --psm 11 -l eng --dpi 300'
        text_psm11 = pytesseract.image_to_string(preprocessed_image, config=custom_config)
        
        # Combine results, with preference to PSM 6 for structured text
        combined_text = text_psm6
        
        # If PSM11 contains numbers or important patterns that PSM6 might miss, add them
        invoice_patterns = ['invoice', 'total', 'vat', 'quantity', 'product']
        for pattern in invoice_patterns:
            if pattern in text_psm11.lower() and pattern not in combined_text.lower():
                combined_text += "\n" + text_psm11
                break
                
        return combined_text
    except Exception as e:
        logger.error(f"Error in OCR text extraction: {str(e)}")
        raise

def extract_table_data(image_path):
    """
    Enhanced table extraction using computer vision techniques
    """
    # Read the original image
    img = cv2.imread(image_path)
    original_img = img.copy()
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply preprocessing specifically for table structure detection
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Apply morphological operations to identify table structure
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    
    # Find contours of table-like structures
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours by area (largest first) to identify table regions
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    table_data_list = []
    
    # Try to identify the main table region
    table_region = None
    min_table_area = 0.1 * img.shape[0] * img.shape[1]  # Minimum 10% of image area
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_table_area:
            x, y, w, h = cv2.boundingRect(contour)
            # Check if this is likely to be a table (width > height / 3)
            if w > h / 3:
                table_region = (x, y, w, h)
                break
    
    # If we found a table region, extract it and process
    if table_region:
        x, y, w, h = table_region
        table_img = original_img[y:y+h, x:x+w]
        
        # Apply OCR specifically to the table region
        custom_config = r'--oem 3 --psm 6 -c preserve_interword_spaces=1'
        table_text = pytesseract.image_to_string(table_img, config=custom_config)
        
        # Process the table text
        lines = table_text.strip().split('\n')
        
        # Try to identify header row
        header_row = None
        for i, line in enumerate(lines):
            if 'Product' in line and 'Code' in line and ('Quantity' in line or 'Description' in line):
                header_row = i
                break
        
        # Process table rows
        if header_row is not None:
            # Get column positions from header
            header = lines[header_row]
            
            # Extract data from rows below header
            for i in range(header_row + 1, len(lines)):
                line = lines[i].strip()
                if line and any(char.isdigit() for char in line):  # Ensure line has at least one digit
                    row_dict = {'text': line}
                    table_data_list.append(row_dict)
    
    # If table detection failed, fallback to processing the whole image
    if not table_data_list:
        # Use Tesseract's built-in table detection
        custom_config = r'--oem 3 --psm 6 -c preserve_interword_spaces=1'
        data = pytesseract.image_to_data(gray, config=custom_config, output_type=pytesseract.Output.DICT)
        
        for i in range(len(data['text'])):
            if int(data['conf'][i]) > 60:  # Only include confident detections
                text = data['text'][i].strip()
                if text and any(char.isdigit() for char in text):  # Line has text and at least one digit
                    row_dict = {
                        'text': text,
                        'left': data['left'][i],
                        'top': data['top'][i],
                        'width': data['width'][i],
                        'height': data['height'][i]
                    }
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
    logger.debug("Parsing line items from text...")
    
    # Find the table section using multiple patterns
    table_patterns = [
        # Pattern 1: Look for standard table headers to end of table
        r'(?:Product\s+Code|Item\s+Code).*?(?=Total\s+\(Rs\)|GRAND\s+TOTAL|Sub\s+Total)',
        # Pattern 2: Look for numeric product codes at the start of lines
        r'(?:\d{8,10}\s+\d+(?:\.\d+)?.*?\n)+',
        # Pattern 3: Look for common product identifier patterns
        r'(?:[A-Z0-9]{6,}\s+\d+(?:\.\d+)?.*?\n)+'
    ]
    
    table_text = ""
    for pattern in table_patterns:
        matches = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if matches:
            table_text = matches.group(0)
            logger.debug(f"Found table text using pattern: {pattern[:30]}...")
            break
    
    if not table_text:
        logger.warning("No table section found in the invoice text")
        return []
    
    # Multiple patterns for line items based on different invoice formats
    line_item_patterns = [
        # Pattern 1: Standard format with 6 columns
        r'(\d+|[A-Z0-9]{6,})\s+(\d+(?:\.\d+)?)\s+([A-Za-z0-9\s&\-\.,]+?)\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s+(\d+(?:,\d+)*(?:\.\d+)?)',
        # Pattern 2: Format with merged discount column
        r'(\d+|[A-Z0-9]{6,})\s+(\d+(?:\.\d+)?)\s+([A-Za-z0-9\s&\-\.,]+?)\s+(\d+(?:\.\d+)?)\s+(\d+(?:,\d+)*(?:\.\d+)?)',
        # Pattern 3: Format specifically for Edendale invoices
        r'(5\d{7}|8\d{7}|9\d{7})\s+(\d+(?:\.\d+)?)\s+([A-Za-z0-9\s&\-\.,]+?)\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s+(\d+(?:,\d+)*(?:\.\d+)?)'
    ]
    
    line_items = []
    
    # Try each pattern
    for pattern in line_item_patterns:
        matches = re.finditer(pattern, table_text)
        for match in matches:
            groups = match.groups()
            
            # Initialize item variable
            item = None
            
            # Handle different formats based on number of groups
            if len(groups) == 6:  # Full format with discount
                item = {
                    'product_code': groups[0].strip(),
                    'quantity': groups[1].strip(),
                    'description': groups[2].strip(),
                    'unit_price': groups[3].strip(),
                    'discount': groups[4].strip(),
                    'total': groups[5].strip().replace(',', '')
                }
            elif len(groups) == 5:  # Format without separate discount
                item = {
                    'product_code': groups[0].strip(),
                    'quantity': groups[1].strip(),
                    'description': groups[2].strip(),
                    'unit_price': groups[3].strip(),
                    'discount': '0',  # Default to zero discount
                    'total': groups[4].strip().replace(',', '')
                }
            
            # Only add valid items (with numeric product code or properly formatted code)
            if item and (re.match(r'^\d+$', item['product_code']) or re.match(r'^[A-Z0-9]{6,}$', item['product_code'])):
                line_items.append(item)
    
    # If patterns didn't work, try line-by-line approach with smarter parsing
    if not line_items:
        logger.debug("Regular expression patterns didn't match, using line-by-line approach")
        lines = table_text.strip().split('\n')
        
        # Skip header row and empty lines
        data_lines = [line.strip() for line in lines if line.strip() and not re.search(r'Product\s+Code|Item\s+Code', line, re.IGNORECASE)]
        
        for line in data_lines:
            # Replace multiple spaces with a single space for easier splitting
            clean_line = re.sub(r'\s+', ' ', line).strip()
            parts = clean_line.split(' ')
            
            # Only process if we have enough parts for a valid line item
            if len(parts) >= 5:
                try:
                    # Check if first part looks like a product code
                    if re.match(r'^\d+$', parts[0]) or re.match(r'^[A-Z0-9]{6,}$', parts[0]):
                        # Determine the structure based on the parts
                        # - Product code is always the first element
                        # - Quantity is always the second element if it's a number
                        # - Total is always the last element
                        # - Description is the middle section
                        # - Unit price is usually the element before discount
                        # - Discount is usually the element before total
                        
                        product_code = parts[0]
                        quantity = parts[1] if re.match(r'^\d+(?:\.\d+)?$', parts[1]) else '1'
                        
                        # Last part is always total
                        total = parts[-1].replace(',', '')
                        
                        # If we have at least 6 parts, assume discount and unit price
                        if len(parts) >= 6:
                            unit_price = parts[-3]
                            discount = parts[-2]
                            # Description is everything between quantity and unit price
                            description = ' '.join(parts[2:-3])
                        else:
                            # With fewer parts, assume no discount
                            unit_price = parts[-2]
                            discount = '0'
                            # Description is everything between quantity and unit price
                            description = ' '.join(parts[2:-2])
                        
                        item = {
                            'product_code': product_code,
                            'quantity': quantity,
                            'description': description,
                            'unit_price': unit_price,
                            'discount': discount,
                            'total': total
                        }
                        
                        line_items.append(item)
                except Exception as e:
                    logger.warning(f"Couldn't parse line item: {line} - {str(e)}")
    
    logger.debug(f"Extracted {len(line_items)} line items from invoice")
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
        logger.info("Preprocessing the image for improved OCR accuracy")
        preprocessed_image = preprocess_image(image_path)
        
        # Extract text using OCR with improved settings
        logger.info("Extracting text from preprocessed image")
        text = extract_text(preprocessed_image)
        logger.debug(f"Extracted text: {text[:200]}...")  # Log first 200 chars of extracted text
        
        # First try to detect the invoice type for better specialized extraction
        invoice_type = 'unknown'
        if 'Edendale Distributors' in text:
            invoice_type = 'edendale'
            logger.info("Detected Edendale Distributors invoice format")
        # Add more invoice type detections as needed
        
        # Parse different sections of the invoice
        logger.info("Parsing invoice header information")
        header = parse_invoice_header(text)
        
        # Get table data using computer vision techniques for better line item detection
        logger.info("Extracting table data using specialized table detection")
        table_data = extract_table_data(image_path)
        logger.debug(f"Extracted {len(table_data)} table elements")
        
        # Use table data to improve line item extraction when available
        # We'll combine results from both approaches for better accuracy
        
        # Parse line items from text
        logger.info("Parsing line items from extracted text")
        line_items = parse_line_items(text)
        
        # If table data extraction provided useful information but text extraction didn't find line items
        if table_data and (not line_items or len(line_items) < 2):
            logger.info("Using table data to improve line item detection")
            
            # Process each row from table data
            table_items = []
            for row in table_data:
                if 'text' in row and row['text'].strip():
                    # Clean up the text
                    clean_text = re.sub(r'\s+', ' ', row['text']).strip()
                    
                    # Try to detect if this is a line item using patterns
                    if (re.match(r'^\d+', clean_text) or 
                        re.match(r'^[A-Z0-9]{6,}', clean_text)):
                        
                        # Split by spaces and try to extract components
                        parts = clean_text.split(' ')
                        if len(parts) >= 4:  # Need at least 4 parts for a valid item
                            try:
                                # Extract using position-based approach
                                product_code = parts[0]
                                
                                # Check if second part is a number (quantity)
                                if re.match(r'^\d+(?:\.\d+)?$', parts[1]):
                                    quantity = parts[1]
                                    desc_start = 2
                                else:
                                    quantity = '1'  # Default quantity
                                    desc_start = 1
                                
                                # Last part is usually total
                                total = parts[-1].replace(',', '')
                                
                                # Try to find unit price and discount, typically near end
                                if len(parts) >= desc_start + 3:
                                    unit_price = parts[-3]
                                    discount = parts[-2]
                                    desc_end = -3
                                else:
                                    unit_price = parts[-2] if len(parts) >= desc_start + 2 else '0'
                                    discount = '0'
                                    desc_end = -2 if len(parts) >= desc_start + 2 else -1
                                
                                # Description is everything between
                                description = ' '.join(parts[desc_start:desc_end])
                                
                                table_items.append({
                                    'product_code': product_code,
                                    'quantity': quantity,
                                    'description': description,
                                    'unit_price': unit_price,
                                    'discount': discount,
                                    'total': total
                                })
                            except Exception as e:
                                logger.warning(f"Error processing table row: {clean_text} - {str(e)}")
            
            # If we found items from table data and they look better than text extraction
            if table_items and (not line_items or len(table_items) > len(line_items)):
                line_items = table_items
                logger.info(f"Using {len(table_items)} items extracted from table data")
        
        # Parse totals from the text
        logger.info("Extracting invoice total information")
        totals = parse_totals(text)
        
        # Combine all data
        invoice_data = {
            'header': header,
            'line_items': line_items,
            'totals': totals,
            'invoice_type': invoice_type
        }
        
        # Apply specific rules for known invoice formats for final cleanup
        logger.info(f"Applying specialized rules for invoice type: {invoice_type}")
        invoice_data = apply_specific_rules(invoice_data, text)
        
        # Post-processing validation and cleanup
        # Ensure all numeric fields are properly formatted
        for item in invoice_data['line_items']:
            for field in ['quantity', 'unit_price', 'discount', 'total']:
                # Clean any non-numeric characters except decimal point
                if item[field]:
                    item[field] = re.sub(r'[^\d.]', '', item[field])
        
        logger.info(f"Successfully processed image and extracted {len(invoice_data['line_items'])} line items")
        
        return invoice_data
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise
