import re
import logging
from PIL import Image
import os
import math
import sys
import signal
import time

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Import OpenCV with fallback
CV2_AVAILABLE = False
try:
    import cv2
    CV2_AVAILABLE = True
    logger.info("OpenCV is available for invoice processing")
except ImportError:
    logger.warning("OpenCV (cv2) not available. Using basic image processing.")
except Exception as e:
    logger.error(f"Error importing OpenCV: {str(e)}")

# Import NumPy with fallback
NUMPY_AVAILABLE = False
try:
    import numpy as np
    NUMPY_AVAILABLE = True
    logger.info("NumPy is available for invoice processing")
except ImportError:
    logger.warning("NumPy not available. Using basic numerical operations.")
except Exception as e:
    logger.error(f"Error importing NumPy: {str(e)}")

# Import pytesseract with fallback
TESSERACT_AVAILABLE = False
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
    logger.info("Pytesseract is available for OCR")
except ImportError:
    logger.warning("Pytesseract not available. OCR functionality will be limited.")
except Exception as e:
    logger.error(f"Error importing Pytesseract: {str(e)}")
    
# Create fallback function for when critical libraries are missing
def create_fallback_invoice_data(image_path):
    """Create basic invoice data structure when libraries are unavailable"""
    logger.warning(f"Using fallback processing for {image_path}")
    
    # Try to extract basic information using PIL if available
    filename = os.path.basename(image_path)
    file_without_extension, _ = os.path.splitext(filename)
    
    # Create basic invoice structure with filename as invoice number
    invoice_data = {
        'header': {
            'invoice_number': file_without_extension[:15],
            'date': '',
            'customer_name': '',
            'customer_ref': '',
            'vat_reg_no': '',
            'business_reg_no': ''
        },
        'line_items': [],
        'totals': {
            'subtotal': '0',
            'vat': '0',
            'grand_total': '0'
        },
        'invoice_type': 'unknown',
        'processing_note': 'Limited processing due to missing libraries'
    }
    
    # Try to use PIL to at least extract image dimensions
    try:
        from PIL import Image
        with Image.open(image_path) as img:
            width, height = img.size
            logger.info(f"Image dimensions: {width}x{height}")
            # Add a single placeholder line item
            invoice_data['line_items'].append({
                'product_code': 'FALLBACK',
                'description': f'Image processed with limited capabilities ({width}x{height})',
                'quantity': '1',
                'unit_price': '0',
                'discount': '0',
                'total': '0'
            })
    except Exception as e:
        logger.error(f"Basic image processing failed: {str(e)}")
        invoice_data['line_items'].append({
            'product_code': 'FALLBACK',
            'description': 'Image could not be processed',
            'quantity': '1',
            'unit_price': '0',
            'discount': '0',
            'total': '0'
        })
    
    return invoice_data

def preprocess_image(image_path):
    """
    Optimized preprocessing to improve OCR accuracy while maintaining performance
    """
    # Read the image
    img = cv2.imread(image_path)
    
    if img is None:
        raise ValueError(f"Could not read image from {image_path}")
    
    # Resize large images to improve performance while maintaining quality
    max_dimension = 2000  # Maximum width or height
    height, width = img.shape[:2]
    if height > max_dimension or width > max_dimension:
        scale = max_dimension / max(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        logger.info(f"Resized image from {width}x{height} to {new_width}x{new_height}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Simple deskew if needed - using faster approach
    try:
        # Check if the image needs to be deskewed
        # Calculate skew angle - only on a sample of points to improve performance
        sample_step = 10  # Only use every 10th point
        coords = np.column_stack(np.where(gray[::sample_step, ::sample_step] > 0))
        if len(coords) > 100:  # Only if we have enough points
            angle = cv2.minAreaRect(coords)[-1]
            
            # Adjust angle
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
                
            # Only deskew if the angle is significant
            if abs(angle) > 1.0:  # Only correct more significant angles
                # Rotate the image to deskew it
                (h, w) = gray.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                gray = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)
    except Exception as e:
        logger.warning(f"Skipping deskew due to error: {str(e)}")
    
    # Use simpler but effective preprocessing approach - focus on performance
    
    # Basic adaptive thresholding
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    # Apply simple morphological operations to clean up noise and enhance text
    kernel = np.ones((2, 2), np.uint8)
    processed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # Return processed image - optimized for balance of quality and speed
    return processed

def extract_text(preprocessed_image):
    """
    Optimized text extraction using Tesseract with efficient processing
    """
    try:
        # Use only 2 most effective OCR configurations instead of 4 to improve performance
        # while maintaining quality
        
        # Use a timeout mechanism to prevent hanging on complex images
        import signal
        
        class TimeoutException(Exception):
            pass
            
        def timeout_handler(signum, frame):
            raise TimeoutException("OCR processing timed out")
            
        # Set a timeout for OCR operations
        signal.signal(signal.SIGALRM, timeout_handler)
        
        # Configuration 1: Optimized configuration for invoice processing
        # Use fast mode (--oem 1) instead of more advanced neural net mode for better performance
        config_optimized = r'--oem 1 --psm 6 -l eng --dpi 300'
        
        # Initialize with empty string to avoid unbound reference
        text_primary = ""
        
        logger.info("Starting primary OCR extraction")
        signal.alarm(20)  # Set 20 second timeout for primary extraction
        try:
            text_primary = pytesseract.image_to_string(preprocessed_image, config=config_optimized)
            signal.alarm(0)  # Cancel alarm if successful
        except TimeoutException:
            logger.warning("Primary OCR timed out, using simplified configuration")
            # If timeout, try with even faster config
            signal.alarm(10)
            text_primary = pytesseract.image_to_string(preprocessed_image, config=r'--oem 0 --psm 6')
            signal.alarm(0)
        
        # Most invoice data should be found in the primary extraction
        # But run a second extraction method only for specific areas that might need it
        
        # Configuration 2: Add a specialized configuration for tables and numbers
        config_specialized = r'--oem 1 --psm 4 -l eng'
        
        # Initialize text_secondary with default empty string
        text_secondary = ""
        
        # Only process with second configuration if first one gave reasonable results
        if len(text_primary) > 100:  # If first extraction gave decent amount of text
            signal.alarm(10)  # Set 10 second timeout for secondary extraction
            try:
                text_secondary = pytesseract.image_to_string(preprocessed_image, config=config_specialized)
                signal.alarm(0)  # Cancel alarm if successful
            except TimeoutException:
                logger.warning("Secondary OCR timed out, using only primary results")
                text_secondary = ""
                signal.alarm(0)
        else:
            # If primary extraction failed, try a different approach
            signal.alarm(15)
            try:
                text_secondary = pytesseract.image_to_string(preprocessed_image, config=r'--oem 0 --psm 11')
                signal.alarm(0)
            except TimeoutException:
                logger.warning("Fallback OCR timed out, using only partial results")
                text_secondary = ""
                signal.alarm(0)
        
        # Simple combination strategy - focus on performance
        combined_text = text_primary
        
        # Add any unique lines from secondary text
        primary_lines = set(line.strip() for line in text_primary.split('\n') if line.strip())
        for line in text_secondary.split('\n'):
            line = line.strip()
            if line and line not in primary_lines:
                combined_text += '\n' + line
                primary_lines.add(line)
        
        # Basic cleanup without complex processing
        combined_text = re.sub(r'\n{3,}', '\n\n', combined_text)  # Remove excess newlines
        
        return combined_text
    except Exception as e:
        logger.error(f"Error in OCR text extraction: {str(e)}")
        # Return empty string as fallback
        return ""

def extract_table_data(image_path):
    """
    Optimized table extraction with performance improvements and timeouts
    """
    try:
        # Read the original image
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"Could not read image from {image_path}")
            return []
            
        # Resize large images for better performance
        max_dimension = 2000
        height, width = img.shape[:2]
        if height > max_dimension or width > max_dimension:
            scale = max_dimension / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
            logger.info(f"Resized image for table extraction from {width}x{height} to {new_width}x{new_height}")
        
        # Set up timeout handling
        import signal
        
        class TimeoutException(Exception):
            pass
            
        def timeout_handler(signum, frame):
            raise TimeoutException("Table extraction processing timed out")
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        table_data_list = []
        
        # Method 1: Optimized line-based extraction - the simplest and fastest approach
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(15)  # Set 15 second timeout
        
        try:
            # Use a faster OCR engine configuration
            simple_config = r'--oem 1 --psm 6'
            full_text = pytesseract.image_to_string(gray, config=simple_config)
            signal.alarm(0)  # Cancel alarm if successful
            
            # Extract lines that look like table rows - using efficient processing
            lines = full_text.strip().split('\n')
            for line in lines:
                clean_line = line.strip()
                # Only consider lines that have numbers and enough content to be a table row
                if clean_line and len(clean_line) > 10 and any(char.isdigit() for char in clean_line):
                    # Check if line starts with what looks like a product code
                    # More efficient regex with limited backtracking
                    if (re.match(r'^\d{5,}(?!\d)', clean_line) or 
                        re.match(r'^[A-Z0-9]{6,}(?![A-Z0-9])', clean_line)):
                        row_dict = {'text': clean_line}
                        table_data_list.append(row_dict)
        except TimeoutException:
            logger.warning("Simple OCR timed out, trying fallback approach")
            signal.alarm(0)  # Reset alarm
        except Exception as e:
            logger.warning(f"Simple OCR extraction failed: {str(e)}")
            signal.alarm(0)  # Reset alarm
        
        # If we already have some data, return it
        if len(table_data_list) >= 3:  # If we found at least 3 rows, consider it successful
            logger.info(f"Found {len(table_data_list)} table rows using simple line extraction")
            return table_data_list
        
        # Only continue if we have few or no results - try a simpler approach for table detection
        # This is a compromise between effectiveness and performance
        table_region = None
        
        try:
            # Try to find the table by searching for table-like structures in the middle of the image
            # Most invoices have tables in the middle section
            height, width = gray.shape
            
            # Focus on middle 60% of the image where tables are typically located
            start_y = int(height * 0.2)
            end_y = int(height * 0.8)
            middle_section = gray[start_y:end_y, :]
            
            # Simple binary thresholding - faster than adaptive thresholding
            _, thresh = cv2.threshold(middle_section, 150, 255, cv2.THRESH_BINARY_INV)
            
            # Find potential table areas using horizontal line detection
            # This is much faster than contour analysis for tables
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
            horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel)
            
            # Find the region with the most horizontal lines - likely to be a table
            line_density = np.sum(horizontal_lines, axis=1)
            if np.max(line_density) > 0:
                dense_line_region = np.argmax(line_density)
                
                # Extract a region around the densest line area
                region_start = max(0, dense_line_region - 100)
                region_end = min(end_y - start_y, dense_line_region + 200)
                
                # Define table region in original image coordinates
                table_region = (0, start_y + region_start, width, region_end - region_start)
        except Exception as e:
            logger.warning(f"Table region detection failed: {str(e)}")
        
        # If we found a table region, process it
        if table_region:
            try:
                x, y, w, h = table_region
                table_img = gray[y:y+h, x:x+w]
                
                # Set a timeout for this OCR operation
                signal.alarm(10)
                
                # Use a faster OCR configuration
                custom_config = r'--oem 1 --psm 6'
                table_text = pytesseract.image_to_string(table_img, config=custom_config)
                signal.alarm(0)  # Cancel alarm if successful
                
                # Process the table text efficiently
                lines = table_text.strip().split('\n')
                for line in lines:
                    clean_line = line.strip()
                    if clean_line and len(clean_line) > 10 and any(char.isdigit() for char in clean_line):
                        parts = clean_line.split()
                        # Check if first part looks like a product code with an efficient check
                        if len(parts) >= 4 and re.match(r'^(\d{5,}|[A-Z0-9]{5,})$', parts[0]):
                            row_dict = {'text': clean_line}
                            table_data_list.append(row_dict)
            except TimeoutException:
                logger.warning("Table region OCR timed out")
                signal.alarm(0)  # Reset alarm
            except Exception as e:
                logger.warning(f"Table region processing failed: {str(e)}")
                signal.alarm(0)  # Reset alarm
        
        # Simplified fallback - only if we still have no data and if time allows
        if not table_data_list:
            try:
                # Process just the middle section of the image where table content is likely
                height, width = gray.shape
                middle_y = height // 2
                section = gray[middle_y-150:middle_y+150, :]  # Just process middle 300 pixels
                
                # Quick timeout
                signal.alarm(8)
                
                # Fast OCR configuration
                section_config = r'--oem 0 --psm 6'  # Fastest engine
                section_text = pytesseract.image_to_string(section, config=section_config)
                signal.alarm(0)  # Cancel alarm
                
                # Process lines from this section
                section_lines = section_text.strip().split('\n')
                for line in section_lines:
                    clean_line = line.strip()
                    if clean_line and len(clean_line) > 10 and any(char.isdigit() for char in clean_line):
                        # Quick check for product code pattern
                        if re.search(r'\d{5,}|\b[A-Z0-9]{6,}\b', clean_line):
                            row_dict = {'text': clean_line}
                            table_data_list.append(row_dict)
            except Exception as e:
                logger.warning(f"Fallback extraction failed: {str(e)}")
                signal.alarm(0)  # Reset alarm just in case
        
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
    # Apply rules based on the detected invoice type
    if invoice_data['invoice_type'] == 'edendale':
        logger.info("Applying specialized Edendale invoice processing")
        # Special processing for Edendale invoices based on sample invoice
        
        # Extract invoice number - Enhanced for Edendale format based on sample
        invoice_number_patterns = [
            r'Invoice\s*No\.?:?\s*(\d+EDL\d+)',
            r'IN(\d+)EDL(\d+)',
            r'Customer Code:\s*(\w+)',
            r'Invoice No:\s*(\d+EDL\d+)'
        ]
        
        for pattern in invoice_number_patterns:
            invoice_match = re.search(pattern, text)
            if invoice_match:
                if 'IN(' in pattern:
                    invoice_data['header']['invoice_number'] = f"IN{invoice_match.group(1)}EDL{invoice_match.group(2)}"
                else:
                    invoice_data['header']['invoice_number'] = invoice_match.group(1).strip()
                break
        
        # Extract date - Enhanced for Edendale format
        date_patterns = [
            r'Date:?\s*(\d{1,2}/\d{1,2}/\d{2,4})',
            r'Date:?\s*(\d{1,2}-\d{1,2}-\d{2,4})',
            r'Date:?\s*(\d{1,2}\.\d{1,2}\.\d{2,4})'
        ]
        
        for pattern in date_patterns:
            date_match = re.search(pattern, text)
            if date_match:
                invoice_data['header']['date'] = date_match.group(1).strip()
                break
        
        # Extract VAT and Business registration numbers with more specific patterns
        vat_reg_patterns = [
            r'VAT Reg No\.?:?\s*([A-Za-z0-9]+)',
            r'Vat Reg No\.?:?\s*([A-Za-z0-9]+)',
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
            r'Business Reg No\.?:?\s*([C0-9]+)',
            r'Business Reg No\.?:?\s*(\d+)'
        ]
        
        for pattern in business_reg_patterns:
            business_match = re.search(pattern, text)
            if business_match:
                invoice_data['header']['business_reg_no'] = business_match.group(1).strip()
                break
        
        # Extract line items - using known product code formats from sample Edendale invoice
        # Multiple patterns for different possible OCR interpretations of the 9-digit product codes
        product_code_patterns = [
            # Standard pattern from sample invoice (9-digit product code starting with 5 or 8)
            r'(5\d{8}|8\d{8}|9\d{8})\s+(\d+(?:\.\d+)?(?:\([^)]+\))?)\s+([A-Za-z0-9\s&\-.,]+?)\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s+(\d+(?:,\d+)*(?:\.\d+)?)',
            
            # For OCR that might misinterpret some digits
            r'(5\d{7,8}|8\d{7,8}|9\d{7,8})\s+(\d+(?:\.\d+)?(?:\([^)]+\))?)\s+([A-Za-z0-9\s&\-.,]+?)\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s+(\d+(?:,\d+)*(?:\.\d+)?)',
            
            # For cases where quantity might contain additional text like "(C48P)"
            r'(5\d{7,8}|8\d{7,8}|9\d{7,8})\s+(\d+[^\s]*)\s+([A-Za-z0-9\s&\-.,]+?)\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s+(\d+(?:,\d+)*(?:\.\d+)?)'
        ]
        
        edendale_items = []
        
        for pattern in product_code_patterns:
            product_code_lines = re.findall(pattern, text)
            if product_code_lines:
                for line in product_code_lines:
                    # Extract just numeric part from quantity - handle cases like "24.00(C24P)"
                    quantity_match = re.search(r'(\d+(?:\.\d+)?)', line[1])
                    quantity = quantity_match.group(1) if quantity_match else "1"
                    
                    # Clean unit price and discount of non-numeric characters except decimal
                    unit_price = re.sub(r'[^\d.]', '', line[3]) 
                    discount = re.sub(r'[^\d.]', '', line[4])
                    
                    # Clean total of any non-numeric characters except decimal
                    total = re.sub(r'[^\d.]', '', line[5])
                    
                    item = {
                        'product_code': line[0].strip(),
                        'quantity': quantity,
                        'description': line[2].strip(),
                        'unit_price': unit_price,
                        'discount': discount,
                        'total': total
                    }
                    edendale_items.append(item)
        
        # If we found items using the specialized patterns, use them instead
        if edendale_items:
            logger.info(f"Found {len(edendale_items)} line items using Edendale-specific patterns")
            invoice_data['line_items'] = edendale_items
                
        # Enhanced extraction for totals in Edendale format based on sample invoice
        # Extract Total from the specific patterns on the Edendale invoice
        total_patterns = [
            r'Total\s*\(Rs\)\s*(?:Rs\.?)?\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)',
            r'Total\s*(?:\(Rs\))?\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)',
            r'Total[\s:]*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)',
            r'Total\s*Amount\s*(?:\(Rs\))?\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)'
        ]
        
        for pattern in total_patterns:
            total_match = re.search(pattern, text)
            if total_match:
                # Clean value of any non-numeric chars except decimal point
                total_value = re.sub(r'[^\d.]', '', total_match.group(1))
                invoice_data['totals']['subtotal'] = total_value
                break
        
        # Extract VAT from the specific format seen in the sample invoice
        vat_patterns = [
            r'VAT\s*@\s*\d+%\s*(?:Rs\.?)?\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)',
            r'VAT\s*(?:@\s*\d+%)?\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)',
            r'VAT[\s:]*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)'
        ]
        
        for pattern in vat_patterns:
            vat_match = re.search(pattern, text)
            if vat_match:
                # Clean value of any non-numeric chars except decimal point
                vat_value = re.sub(r'[^\d.]', '', vat_match.group(1))
                invoice_data['totals']['vat'] = vat_value
                break
        
        # Extract Grand Total from the specific format seen in the sample invoice
        grand_total_patterns = [
            r'GRAND\s*TOTAL\s*(?:Rs\.?)?\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)',
            r'(?:GRAND|Grand)\s*Total\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)',
            r'GRAND\s*TOTAL[\s:]*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)'
        ]
        
        for pattern in grand_total_patterns:
            grand_total_match = re.search(pattern, text)
            if grand_total_match:
                # Clean value of any non-numeric chars except decimal point
                grand_total_value = re.sub(r'[^\d.]', '', grand_total_match.group(1))
                invoice_data['totals']['grand_total'] = grand_total_value
                break

    return invoice_data

def process_image(image_path):
    """
    Main function to process the invoice image and extract all required data
    Includes optimizations for performance and timeout handling
    """
    logger.info(f"Starting processing of image: {image_path}")
    
    # Check if required libraries are available
    if not CV2_AVAILABLE or not NUMPY_AVAILABLE or not TESSERACT_AVAILABLE:
        logger.warning("Critical libraries unavailable. Using basic fallback processing.")
        # Return basic fallback structure with empty fields
        return create_fallback_invoice_data(image_path)
        
    # Initialize with empty/default values
    default_invoice_data = {
        'header': {
            'invoice_number': '',
            'date': '',
            'customer_name': '',
            'customer_ref': '',
            'vat_reg_no': '',
            'business_reg_no': ''
        },
        'line_items': [],
        'totals': {
            'subtotal': '0',
            'vat': '0',
            'grand_total': '0'
        },
        'invoice_type': 'unknown'
    }
    
    # Setup timeout handling
    import signal
    
    class TimeoutException(Exception):
        pass
        
    def timeout_handler(signum, frame):
        raise TimeoutException("Image processing timed out")
    
    # Set signal handler
    signal.signal(signal.SIGALRM, timeout_handler)
    
    try:
        # Check if file exists
        if not os.path.exists(image_path):
            logger.error(f"Image file not found: {image_path}")
            return default_invoice_data
        
        # Set 30 second timeout for entire extraction process
        signal.alarm(30)
        
        try:
            # Preprocess the image - more efficient version
            logger.info("Preprocessing the image for improved OCR accuracy")
            preprocessed_image = preprocess_image(image_path)
            
            # Extract text using OCR with optimized settings
            logger.info("Extracting text from preprocessed image")
            text = extract_text(preprocessed_image)
            logger.debug(f"Extracted text length: {len(text)} characters")
            
            # Turn off alarm after potentially long operations
            signal.alarm(0)
            
            # Quick check for empty text
            if not text or len(text.strip()) < 20:
                logger.warning("Extracted text is too short - OCR may have failed")
                return default_invoice_data
                
            # First try to detect the invoice type for better specialized extraction
            invoice_type = 'unknown'
            
            # Check for Edendale invoice with targeted patterns (performance optimized)
            for pattern in ['Edendale', 'EDENDALE', 'CREDIT SALES', r'5\d{7}|8\d{7}|9\d{7}']:
                if pattern in text or re.search(pattern, text):
                    invoice_type = 'edendale'
                    logger.info(f"Detected Edendale invoice format using pattern: {pattern[:20]}")
                    break
            
            # Parse header (quick operation)
            logger.info("Parsing invoice header information")
            header = parse_invoice_header(text)
            
            # Get table data with optimized table detection (with internal timeouts)
            logger.info("Extracting table data using optimized table detection")
            table_data = extract_table_data(image_path)
            logger.debug(f"Extracted {len(table_data)} table elements")
            
            # Parse line items with timeout protection
            signal.alarm(10)  # Set shorter timeout for potentially intensive regex operations
            try:
                logger.info("Parsing line items from extracted text")
                line_items = parse_line_items(text)
                signal.alarm(0)  # Cancel alarm
            except TimeoutException:
                logger.warning("Line item parsing timed out")
                line_items = []  # Default to empty if timed out
                signal.alarm(0)  # Make sure alarm is canceled
            
            # Process table data for line items if text extraction didn't produce results
            # Optimized to be more efficient
            if table_data and (not line_items or len(line_items) < 2):
                logger.info("Using table data to improve line item detection")
                
                table_items = []
                for row in table_data:
                    if 'text' in row and row['text'].strip():
                        try:
                            # Clean up the text
                            clean_text = re.sub(r'\s+', ' ', row['text']).strip()
                            
                            # Quick check if this is a line item
                            if (re.match(r'^\d+', clean_text) or re.match(r'^[A-Z0-9]{5,}', clean_text)):
                                parts = clean_text.split(' ')
                                if len(parts) >= 4:
                                    # Extract components (simplified for performance)
                                    product_code = parts[0]
                                    quantity = parts[1] if re.match(r'^\d+(?:\.\d+)?$', parts[1]) else '1'
                                    total = parts[-1].replace(',', '')
                                    
                                    if len(parts) >= 5:
                                        unit_price = parts[-2]
                                        description = ' '.join(parts[2:-2])
                                        discount = '0'  # Default to zero discount for simplicity
                                    else:
                                        unit_price = '0'
                                        description = ' '.join(parts[2:-1])
                                        discount = '0'
                                    
                                    table_items.append({
                                        'product_code': product_code,
                                        'quantity': quantity,
                                        'description': description,
                                        'unit_price': unit_price,
                                        'discount': discount,
                                        'total': total
                                    })
                        except Exception as e:
                            logger.warning(f"Error processing table row: {str(e)}")
                            continue  # Skip problematic rows and continue
                
                # Use table items if they're better than text extraction
                if table_items and (not line_items or len(table_items) > len(line_items)):
                    line_items = table_items
                    logger.info(f"Using {len(table_items)} items from table data")
            
            # Parse totals (quick operation)
            logger.info("Extracting invoice total information")
            totals = parse_totals(text)
            
            # Combine all data
            invoice_data = {
                'header': header,
                'line_items': line_items,
                'totals': totals,
                'invoice_type': invoice_type
            }
            
            # Apply specific rules based on invoice type
            invoice_data = apply_specific_rules(invoice_data, text)
            
            # Post-processing - sanitize all numeric fields
            for item in invoice_data['line_items']:
                for field in ['quantity', 'unit_price', 'discount', 'total']:
                    if item[field]:
                        item[field] = re.sub(r'[^\d.]', '', item[field])
                        # Provide defaults for empty values after cleaning
                        if not item[field]:
                            item[field] = '0'
            
            # Also clean totals to ensure they're valid
            for key in invoice_data['totals']:
                if not invoice_data['totals'][key] or not re.match(r'^\d+\.?\d*$', invoice_data['totals'][key]):
                    invoice_data['totals'][key] = '0'
            
            logger.info(f"Successfully processed image and extracted {len(invoice_data['line_items'])} line items")
            return invoice_data
            
        except TimeoutException:
            logger.error("Processing timed out, returning default data")
            signal.alarm(0)  # Reset the alarm
            return default_invoice_data
            
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        # Return default data instead of raising exception to avoid server errors
        return default_invoice_data
