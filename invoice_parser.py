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
    Enhanced preprocessing to improve OCR accuracy with multiple techniques
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
    
    # Create multiple processed versions of the image for better OCR
    preprocessed_images = []
    
    # Version 1: Basic adaptive thresholding (good for clean documents)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    preprocessed_images.append(binary)
    
    # Version 2: Stronger adaptive thresholding (good for low contrast)
    binary_strong = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 3)
    preprocessed_images.append(binary_strong)
    
    # Version 3: Otsu's thresholding (good for bimodal images)
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    preprocessed_images.append(otsu)
    
    # Version 4: Noise reduction and edge enhancement
    # First denoise
    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    # Then threshold
    _, denoise_thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    preprocessed_images.append(denoise_thresh)
    
    # Version 5: Special table structure processing for line items
    # Apply erosion followed by dilation to enhance table structures
    kernel = np.ones((2, 2), np.uint8)
    eroded = cv2.erode(binary, kernel, iterations=1)
    dilated = cv2.dilate(eroded, kernel, iterations=2)
    preprocessed_images.append(dilated)
    
    # Apply contrast stretching to all versions
    enhanced_images = []
    for img in preprocessed_images:
        # Normalize to enhance contrast
        enhanced = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        enhanced_images.append(enhanced)
    
    # Return the most processed version for line item extraction and number recognition
    # This is the version with table structure enhancement
    return enhanced_images[4]

def extract_text(preprocessed_image):
    """
    Enhanced text extraction using Tesseract with multiple configurations
    and intelligent text combination for better accuracy
    """
    try:
        # Try multiple OCR approaches and configurations for better results
        ocr_results = []
        
        # Configuration 1: Standard with layout analysis (good for structured documents)
        config_standard = r'--oem 3 --psm 6 -l eng --dpi 300'
        text_standard = pytesseract.image_to_string(preprocessed_image, config=config_standard)
        ocr_results.append(text_standard)
        
        # Configuration 2: Single column variable height (better for mixed content)
        config_single_column = r'--oem 3 --psm 4 -l eng --dpi 300'
        text_single_column = pytesseract.image_to_string(preprocessed_image, config=config_single_column)
        ocr_results.append(text_single_column)
        
        # Configuration 3: Sparse text (find text anywhere, good for tables and headers)
        config_sparse = r'--oem 3 --psm 11 -l eng --dpi 300'
        text_sparse = pytesseract.image_to_string(preprocessed_image, config=config_sparse)
        ocr_results.append(text_sparse)
        
        # Configuration 4: Special for numbers (better for detecting prices and quantities)
        config_digits = r'--oem 3 --psm 7 -l eng --dpi 300 -c tessedit_char_whitelist="0123456789.,ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz "'
        text_digits = pytesseract.image_to_string(preprocessed_image, config=config_digits)
        ocr_results.append(text_digits)
        
        # Combine results intelligently (more sophisticated than just concatenation)
        # Start with the standard text as our base
        combined_text = text_standard
        
        # Define key patterns to look for in invoices
        key_patterns = {
            'invoice_number': [r'invoice\s*(?:#|number|no[.:]?)\s*([A-Za-z0-9-]+)', 
                              r'IN(\d+)EDL(\d+)'],
            'date': [r'date\s*:\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})', 
                    r'date\s*:\s*(\d{1,2}\s+[A-Za-z]+\s+\d{2,4})'],
            'customer': [r'customer\s*(?:name|ref)[.:]?\s*(.+?)(?=\n|address)', 
                        r'customer[.:]?\s*([A-Za-z0-9\s&\-,.]+)'],
            'vat': [r'vat\s*(?:reg|registration)\s*(?:no|number)[.:]?\s*([A-Za-z0-9]+)',
                   r'vat\s*@\s*\d+%\s*(\d+(?:,\d+)*(?:\.\d+)?)'],
            'product_codes': [r'(?:\d{8}\s+\d+\.\d+)', r'(?:[A-Z0-9]{6,}\s+\d+\.\d+)'],
            'totals': [r'total\s*\((?:rs|inr)\)\s*(\d+(?:,\d+)*(?:\.\d+)?)', 
                      r'grand\s*total\s*(\d+(?:,\d+)*(?:\.\d+)?)']
        }
        
        # For each pattern, look in all OCR results and enhance the combined text
        for pattern_type, patterns in key_patterns.items():
            pattern_found = False
            
            # Check if the pattern already exists in the combined text
            for pattern in patterns:
                if re.search(pattern, combined_text, re.IGNORECASE):
                    pattern_found = True
                    break
            
            # If not found, look in other OCR results
            if not pattern_found:
                for result in ocr_results:
                    for pattern in patterns:
                        if re.search(pattern, result, re.IGNORECASE):
                            # Extract the relevant section and add it to combined text
                            match = re.search(pattern, result, re.IGNORECASE)
                            if match:
                                # Extract context around the match (2 lines before and after)
                                lines = result.split('\n')
                                for i, line in enumerate(lines):
                                    if pattern in line.lower() or pattern_type in line.lower():
                                        start_idx = max(0, i - 2)
                                        end_idx = min(len(lines), i + 3)
                                        context = '\n'.join(lines[start_idx:end_idx])
                                        if context not in combined_text:
                                            combined_text += '\n' + context
                                        pattern_found = True
                                        break
                    if pattern_found:
                        break
        
        # Final clean-up: remove duplicate lines and excess whitespace
        lines = combined_text.split('\n')
        unique_lines = []
        for line in lines:
            clean_line = line.strip()
            if clean_line and clean_line not in unique_lines:
                unique_lines.append(clean_line)
        
        combined_text = '\n'.join(unique_lines)
        
        return combined_text
    except Exception as e:
        logger.error(f"Error in OCR text extraction: {str(e)}")
        raise

def extract_table_data(image_path):
    """
    Enhanced table extraction using computer vision techniques with more robust error handling
    """
    try:
        # Read the original image
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"Could not read image from {image_path}")
            return []
            
        original_img = img.copy()
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        table_data_list = []
        
        # Method 1: Try to use line-based extraction from direct OCR
        try:
            # Process the full image as text first - simpler and less prone to errors
            # Use a more reliable configuration
            simple_config = r'--oem 3 --psm 4'  # Assume single column of text
            full_text = pytesseract.image_to_string(gray, config=simple_config)
            
            # Extract lines that look like table rows
            lines = full_text.strip().split('\n')
            for line in lines:
                clean_line = line.strip()
                # Only consider lines that have numbers and enough content to be a table row
                if clean_line and len(clean_line) > 10 and any(char.isdigit() for char in clean_line):
                    # Check if line starts with what looks like a product code
                    if re.match(r'^\d{5,}', clean_line) or re.match(r'^[A-Z0-9]{6,}', clean_line):
                        row_dict = {'text': clean_line}
                        table_data_list.append(row_dict)
        except Exception as e:
            logger.warning(f"Simple OCR extraction failed: {str(e)}")
        
        # If we already have some data, return it
        if len(table_data_list) >= 3:  # If we found at least 3 rows, consider it successful
            logger.info(f"Found {len(table_data_list)} table rows using simple line extraction")
            return table_data_list
            
        # Method 2: Try more complex table extraction only if the simple method didn't work well
        try:
            # Apply preprocessing specifically for table structure detection
            _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Apply morphological operations to identify table structure
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            dilated = cv2.dilate(thresh, kernel, iterations=1)
            
            # Find contours of table-like structures
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter and sort contours by area
            valid_contours = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 1000:  # Minimum area threshold
                    valid_contours.append(contour)
            
            valid_contours = sorted(valid_contours, key=cv2.contourArea, reverse=True)
            
            # Try to identify the main table region
            table_region = None
            for contour in valid_contours[:5]:  # Check top 5 contours only
                x, y, w, h = cv2.boundingRect(contour)
                # Check if this is likely to be a table (reasonable width and height)
                if w > 300 and h > 100:
                    table_region = (x, y, w, h)
                    break
            
            # If we found a table region, extract it and process
            if table_region:
                x, y, w, h = table_region
                table_img = original_img[y:y+h, x:x+w]
                
                # Apply OCR specifically to the table region
                custom_config = r'--oem 3 --psm 6'
                table_text = pytesseract.image_to_string(table_img, config=custom_config)
                
                # Process the table text
                lines = table_text.strip().split('\n')
                
                # Extract rows that look like table data
                for line in lines:
                    clean_line = line.strip()
                    if clean_line and len(clean_line) > 10 and any(char.isdigit() for char in clean_line):
                        parts = clean_line.split()
                        # Check if first part looks like a product code
                        if len(parts) >= 4 and (re.match(r'^\d+', parts[0]) or re.match(r'^[A-Z0-9]{5,}', parts[0])):
                            row_dict = {'text': clean_line}
                            table_data_list.append(row_dict)
        except Exception as e:
            logger.warning(f"Complex table extraction failed: {str(e)}")
            
        # Method 3: Fallback to direct structure detection (only if we still don't have data)
        if not table_data_list:
            try:
                # Split the image into horizontal sections and process each
                height, width = gray.shape
                num_sections = 5
                section_height = height // num_sections
                
                for i in range(num_sections):
                    # Define section boundaries
                    y_start = i * section_height
                    y_end = (i + 1) * section_height if i < num_sections - 1 else height
                    
                    # Extract section
                    section = gray[y_start:y_end, 0:width]
                    
                    # OCR the section
                    section_config = r'--oem 3 --psm 6'
                    section_text = pytesseract.image_to_string(section, config=section_config)
                    
                    # Process lines from this section
                    section_lines = section_text.strip().split('\n')
                    for line in section_lines:
                        clean_line = line.strip()
                        if clean_line and len(clean_line) > 10 and any(char.isdigit() for char in clean_line):
                            parts = re.split(r'\s+', clean_line)
                            if len(parts) >= 4:
                                row_dict = {'text': clean_line}
                                table_data_list.append(row_dict)
            except Exception as e:
                logger.warning(f"Sectional extraction failed: {str(e)}")
        
        return table_data_list
    
    except Exception as e:
        logger.error(f"Error in table extraction: {str(e)}")
        # Return empty list on error, the system will fall back to other extraction methods
        return []

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
    if 'Edendale' in text or 'EDENDALE' in text:
        logger.info("Applying specialized Edendale invoice processing")
        # Special processing for Edendale invoices
        # Based on the sample invoice in the image
        
        # Extract invoice number format IN1234EDL5678
        invoice_number_patterns = [
            r'IN(\d+)EDL(\d+)',
            r'Customer Code:\s*(\w+)',
            r'Invoice No:\s*(\w+)'
        ]
        
        for pattern in invoice_number_patterns:
            invoice_match = re.search(pattern, text)
            if invoice_match:
                if 'IN' in pattern:
                    invoice_data['header']['invoice_number'] = f"IN{invoice_match.group(1)}EDL{invoice_match.group(2)}"
                else:
                    invoice_data['header']['invoice_number'] = invoice_match.group(1).strip()
                break
        
        # Extract VAT and Business registration numbers with more specific patterns
        vat_reg_patterns = [
            r'VAT Reg No\.?:?\s*([A-Za-z0-9]+)',
            r'VAT Reg No\.?:?\s*(\d+)'
        ]
        
        for pattern in vat_reg_patterns:
            vat_match = re.search(pattern, text)
            if vat_match:
                invoice_data['header']['vat_reg_no'] = vat_match.group(1).strip()
                break
        
        # Extract business registration number
        business_reg_patterns = [
            r'Business Reg No\.?:?\s*([A-Za-z0-9]+)',
            r'Business Reg No\.?:?\s*(\d+)'
        ]
        
        for pattern in business_reg_patterns:
            business_match = re.search(pattern, text)
            if business_match:
                invoice_data['header']['business_reg_no'] = business_match.group(1).strip()
                break
        
        # Extract line items - using known product code formats for Edendale invoices
        # Multiple patterns for different possible OCR interpretations
        product_code_patterns = [
            # Pattern for correct recognition
            r'(5\d{7}|8\d{7}|9\d{7})\s+(\d+(?:\.\d+)?)\s+([A-Za-z0-9\s&\-.,]+?)\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s+(\d+(?:,\d+)*(?:\.\d+)?)',
            # Pattern for when OCR might merge numbers
            r'(5\d{7}|8\d{7}|9\d{7})\s+(\d+)\s+([A-Za-z0-9\s&\-.,]+?)\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s+(\d+)',
            # Pattern for possible OCR errors in product codes
            r'(5\d{5,7}|8\d{5,7}|9\d{5,7})\s+(\d+(?:\.\d+)?)\s+([A-Za-z0-9\s&\-.,]+?)\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s+(\d+(?:,\d+)*(?:\.\d+)?)'
        ]
        
        edendale_items = []
        
        for pattern in product_code_patterns:
            product_code_lines = re.findall(pattern, text)
            if product_code_lines:
                for line in product_code_lines:
                    item = {
                        'product_code': line[0].strip(),
                        'quantity': line[1].strip(),
                        'description': line[2].strip(),
                        'unit_price': line[3].strip(),
                        'discount': line[4].strip(),
                        'total': line[5].strip().replace(',', '')
                    }
                    edendale_items.append(item)
        
        # If we found items using the specialized patterns, use them instead
        if edendale_items:
            logger.info(f"Found {len(edendale_items)} line items using Edendale-specific patterns")
            invoice_data['line_items'] = edendale_items
                
        # Enhanced extraction for totals in Edendale format
        # Extract Total
        total_patterns = [
            r'Total \(Rs\)\s*(\d+(?:,\d+)*(?:\.\d+)?)',
            r'Total\s*\(Rs\)\s*(\d+(?:,\d+)*(?:\.\d+)?)',
            r'Total\s*(\d+(?:,\d+)*(?:\.\d+)?)'
        ]
        
        for pattern in total_patterns:
            total_match = re.search(pattern, text)
            if total_match:
                invoice_data['totals']['subtotal'] = total_match.group(1).replace(',', '')
                break
        
        # Extract VAT
        vat_patterns = [
            r'VAT\s*@\s*\d+%\s*(\d+(?:,\d+)*(?:\.\d+)?)',
            r'VAT\s*@\s*(\d+(?:,\d+)*(?:\.\d+)?)',
            r'VAT\s*(\d+(?:,\d+)*(?:\.\d+)?)'
        ]
        
        for pattern in vat_patterns:
            vat_match = re.search(pattern, text)
            if vat_match:
                invoice_data['totals']['vat'] = vat_match.group(1).replace(',', '')
                break
        
        # Extract Grand Total
        grand_total_patterns = [
            r'GRAND TOTAL\s*(\d+(?:,\d+)*(?:\.\d+)?)',
            r'Grand Total\s*(\d+(?:,\d+)*(?:\.\d+)?)',
            r'GRAND\s*TOTAL\s*(\d+(?:,\d+)*(?:\.\d+)?)'
        ]
        
        for pattern in grand_total_patterns:
            grand_total_match = re.search(pattern, text)
            if grand_total_match:
                invoice_data['totals']['grand_total'] = grand_total_match.group(1).replace(',', '')
                break

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
        # Check for Edendale invoice with multiple patterns
        edendale_patterns = [
            'Edendale Distributors',
            'Edendale',
            'EDENDALE',
            # Product code patterns specific to Edendale
            r'5\d{7}|8\d{7}|9\d{7}',
            # Check for their format of VAT registration
            'VAT Reg No.: VAT20362266',
            # Check for typical text in their invoices
            'CREDIT SALES',
            # Check for receipt statement
            'Credit Sales not settled within 1 month from invoice date will bear interest at 2%'
        ]
        
        for pattern in edendale_patterns:
            if pattern in text or re.search(pattern, text):
                invoice_type = 'edendale'
                logger.info(f"Detected Edendale Distributors invoice format using pattern: {pattern[:20]}")
                break
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
