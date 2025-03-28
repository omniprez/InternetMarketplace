import os
import logging
import uuid
import time
import json
import sys
import tempfile
import shutil
import base64

# Get log directory from environment variable or create a default
logs_dir = os.environ.get('LOG_DIR', os.path.join(tempfile.gettempdir(), 'logs'))
os.makedirs(logs_dir, exist_ok=True)
log_file = os.path.join(logs_dir, 'flask-debug.log')

# Setup logging first with file handler for detailed debugging
# Using just stream handler to avoid file permission issues
logging.basicConfig(level=logging.DEBUG,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                   handlers=[
                       logging.StreamHandler()
                   ])
logger = logging.getLogger(__name__)

# Import Flask and related dependencies with error handling
try:
    from flask import Flask, render_template, request, redirect, url_for, flash, session, send_file, jsonify
    logger.info("Successfully imported Flask")
except ImportError as e:
    logger.error(f"Failed to import Flask: {e}")
    sys.exit(1)

try:
    import pandas as pd
    logger.info("Successfully imported pandas")
except ImportError as e:
    logger.error(f"Failed to import pandas: {e}")
    sys.exit(1)

try:
    from werkzeug.utils import secure_filename
    logger.info("Successfully imported werkzeug")
except ImportError as e:
    logger.error(f"Failed to import werkzeug: {e}")
    sys.exit(1)

# Import our application modules with error handling
try:
    from invoice_parser import process_image
    logger.info("Successfully imported invoice_parser")
except ImportError as e:
    logger.error(f"Failed to import invoice_parser: {e}")
    sys.exit(1)

try:
    from excel_generator import create_excel_file
    logger.info("Successfully imported excel_generator")
except ImportError as e:
    logger.error(f"Failed to import excel_generator: {e}")
    sys.exit(1)

# Set up enhanced OCR capabilities with available libraries
logger.info("Setting up enhanced OCR capabilities with OpenCV and scikit-learn")
enhanced_ocr_available = False

try:
    import cv2
    import numpy as np
    enhanced_ocr_available = True
    logger.info("Enhanced OCR capabilities available with OpenCV")
except ImportError as e:
    logger.warning(f"Enhanced OCR with OpenCV not available: {e}")
    
try:
    from sklearn import feature_extraction, metrics
    enhanced_ocr_available = enhanced_ocr_available and True
    logger.info("Text analysis capabilities available with scikit-learn")
except ImportError as e:
    logger.warning(f"Text analysis with scikit-learn not available: {e}")
    
# No need for Anthropic/Claude integration - using purely open-source methods

# Try to import the AI field recognizer with graceful fallback
field_recognizer = None
try:
    from ai_field_recognition import AIFieldRecognizer
    field_recognizer = AIFieldRecognizer()
    logger.info("Successfully initialized AI field recognizer")
except Exception as e:
    logger.error(f"Failed to initialize AI field recognizer: {e}")
    logger.warning("Continuing without AI field recognition capabilities")
    
    # Create enhanced fallback recognizer with open-source libraries for improved OCR accuracy
    class FallbackFieldRecognizer:
        def __init__(self):
            # Use flags to determine which enhanced capabilities are available
            self.enhanced_ocr = enhanced_ocr_available
            
            # Additional thresholds for extraction confidence
            self.table_detection_threshold = 0.6
            self.field_confidence_threshold = 0.7
            
            # No API connections used - pure open-source methods
            
            if self.enhanced_ocr:
                logger.info("Enhanced OCR field recognition enabled with OpenCV and scikit-learn")
            else:
                logger.info("Enhanced OCR not available - using basic OCR only")
                
        def process_invoice(self, image_path):
            """Process the invoice with enhanced OCR if available, otherwise fallback to basic"""
            logger.info(f"Processing invoice with FallbackFieldRecognizer: {image_path}")
            
            # Start with standard OCR processing
            from invoice_parser import process_image
            result = process_image(image_path)
            
            # Create basic extraction result
            extraction_result = {
                'fields': {k: v for k, v in result.get('header', {}).items()},
                'line_items': result.get('line_items', []),
                'visualization_data': None,
                'enhanced': False
            }
            
            # Apply enhanced processing if libraries are available
            if self.enhanced_ocr and os.path.exists(image_path):
                try:
                    logger.info("Applying enhanced image processing with OpenCV")
                    
                    # Load image with OpenCV for enhancement
                    image = cv2.imread(image_path)
                    if image is None:
                        logger.warning(f"Could not read image at {image_path} with OpenCV")
                        return extraction_result
                    
                    # Apply preprocessing to improve OCR quality
                    # Convert to grayscale
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    
                    # Apply adaptive thresholding for better text extraction
                    binary = cv2.adaptiveThreshold(
                        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                        cv2.THRESH_BINARY, 11, 2
                    )
                    
                    # Remove noise with morphological operations
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
                    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
                    
                    # Use enhanced binary image for improved OCR
                    # First, save the enhanced image to a temporary file
                    enhanced_path = image_path + "_enhanced.jpg"
                    cv2.imwrite(enhanced_path, cleaned)
                    
                    # Run improved OCR on the enhanced image
                    import pytesseract
                    ocr_text = pytesseract.image_to_string(enhanced_path)
                    
                    # Use scikit-learn enhanced text analysis if available
                    try:
                        # Use TF-IDF to identify important keywords in invoice
                        from sklearn.feature_extraction.text import TfidfVectorizer
                        
                        # Break text into individual lines for analysis
                        lines = [line for line in ocr_text.split('\n') if line.strip()]
                        
                        # Create a TF-IDF vectorizer focused on invoice terminology
                        vectorizer = TfidfVectorizer(
                            min_df=1, stop_words='english', 
                            ngram_range=(1, 2)  # Use unigrams and bigrams
                        )
                        
                        # Transform the text data
                        tfidf_matrix = vectorizer.fit_transform(lines)
                        
                        # Get feature names (terms)
                        feature_names = vectorizer.get_feature_names_out()
                        
                        # Extract important invoice terms with their scores
                        # For each line, find the most important terms
                        important_terms = []
                        for i, line in enumerate(lines):
                            feature_index = tfidf_matrix[i, :].nonzero()[1]
                            tfidf_scores = zip(
                                [feature_names[x] for x in feature_index],
                                [tfidf_matrix[i, x] for x in feature_index]
                            )
                            # Sort by score
                            sorted_terms = sorted(tfidf_scores, key=lambda x: x[1], reverse=True)
                            important_terms.append((line, sorted_terms[:3]))  # Top 3 terms
                        
                        # Improve field recognition based on important terms
                        improved_fields = {}
                        for line, terms in important_terms:
                            line_lower = line.lower()
                            
                            # Look for invoice number with better pattern matching
                            if any(t[0] in ['invoice', 'inv', 'invoice number', 'inv no'] for t in terms):
                                # Extract invoice number with smarter pattern matching
                                import re
                                inv_match = re.search(r'(?:invoice|inv)(?:\s+number|\s+no)?[:#\s]*([A-Z0-9\-]+)', line_lower)
                                if inv_match:
                                    improved_fields['invoice_number'] = inv_match.group(1).strip()
                                    
                            # Look for dates with better pattern recognition
                            if any(t[0] in ['date', 'issued', 'invoice date'] for t in terms):
                                # Extract date with comprehensive pattern matching
                                date_patterns = [
                                    r'(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4})',  # DD/MM/YYYY, MM/DD/YYYY
                                    r'(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4})',  # DD Mon YYYY
                                    r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{2,4})'  # Mon DD, YYYY
                                ]
                                
                                for pattern in date_patterns:
                                    date_match = re.search(pattern, line, re.IGNORECASE)
                                    if date_match:
                                        improved_fields['date'] = date_match.group(1).strip()
                                        break
                                        
                            # Look for vendor information
                            if any(t[0] in ['from', 'vendor', 'supplier', 'company'] for t in terms):
                                # Look for vendor name using patterns
                                if 'vendor' not in improved_fields and len(line) > 5:
                                    if not any(w in line_lower for w in ['invoice', 'total', 'subtotal', 'date']):
                                        improved_fields['vendor'] = line.strip()
                            
                            # Look for totals with better pattern matching
                            if any(t[0] in ['total', 'amount', 'due', 'balance', 'amount due'] for t in terms):
                                # Extract amount with comprehensive pattern
                                total_match = re.search(r'(?:total|amount|due|balance)[:\s]*[$€£]?\s*(\d+[,\d]*\.\d+|\d+[,\d]*)', line_lower)
                                if total_match:
                                    # Clean up the total value
                                    total_str = total_match.group(1).replace(',', '')
                                    try:
                                        improved_fields['total'] = total_str
                                    except:
                                        # Just use the string as is if conversion fails
                                        improved_fields['total'] = total_match.group(1).strip()
                                    
                        # Update extraction results with improved fields
                        for field, value in improved_fields.items():
                            if value:  # Only update if we have a value
                                extraction_result['fields'][field] = value
                        
                        # Try to improve line item detection
                        line_items = self._detect_line_items_from_text(ocr_text)
                        if line_items and len(line_items) > len(extraction_result['line_items']):
                            extraction_result['line_items'] = line_items
                            
                        # Attempt to detect and parse table structures if OpenCV is available
                        try:
                            table_regions = self._detect_tables(image)
                            if table_regions:
                                # Process each detected table region
                                for i, (x, y, w, h) in enumerate(table_regions):
                                    table_img = image[y:y+h, x:x+w]
                                    table_items = self._extract_table_data(table_img)
                                    if table_items and len(table_items) > len(extraction_result['line_items']):
                                        extraction_result['line_items'] = table_items
                        except Exception as table_err:
                            logger.warning(f"Table detection failed: {table_err}")
                        
                        # Mark that enhanced processing was applied
                        extraction_result['enhanced'] = True
                        logger.info("Successfully applied enhanced OCR and text analysis")
                        
                    except Exception as sklearn_err:
                        logger.warning(f"Enhanced text analysis failed: {sklearn_err}")
                    
                    # Clean up temporary file
                    try:
                        os.remove(enhanced_path)
                    except:
                        pass
                        
                except Exception as e:
                    logger.error(f"Enhanced image processing failed: {e}")
            
            return extraction_result
        
        def _detect_line_items_from_text(self, text):
            """Enhanced line item detection from text"""
            line_items = []
            lines = text.split('\n')
            
            # Look for blocks of text that appear to be line items
            in_line_items_section = False
            item_candidates = []
            
            # Keywords that indicate start of line items section
            item_section_indicators = ['item', 'description', 'product', 'service', 'quantity', 'qty', 'price', 'amount']
            
            # Try to extract line items
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                # Check if this line could be a header for line items
                if not in_line_items_section:
                    line_lower = line.lower()
                    # Check if this looks like a table header
                    if sum(1 for indicator in item_section_indicators if indicator in line_lower) >= 2:
                        in_line_items_section = True
                        continue
                
                # If we're in the line items section, collect potential items
                if in_line_items_section:
                    # Skip lines that are likely to be section headers
                    if any(header in line.lower() for header in ['subtotal', 'total', 'tax', 'shipping', 'discount']):
                        continue
                        
                    # Skip very short lines
                    if len(line) < 5:
                        continue
                        
                    # If line contains numbers and text, it's a candidate for a line item
                    if any(c.isdigit() for c in line) and any(c.isalpha() for c in line):
                        item_candidates.append(line)
            
            # Process item candidates into structured data
            for item in item_candidates:
                # Try to extract components - this is simplified and would need
                # more sophisticated parsing in a real implementation
                parts = item.split()
                
                # Skip if too few parts
                if len(parts) < 2:
                    continue
                
                # Try to identify description, quantity, unit price and total
                line_item = {'description': '', 'quantity': '', 'unit_price': '', 'total': ''}
                
                # Simple heuristic - assume first parts are description, last numeric part is total
                # and if enough parts, try to extract quantity and unit price
                
                # Extract description (take first part of the line, up to 70% of words)
                desc_end = max(1, int(len(parts) * 0.7))
                line_item['description'] = ' '.join(parts[:desc_end])
                
                # Look for numeric values in the remaining parts
                numeric_parts = []
                for i in range(desc_end, len(parts)):
                    # Clean up the part to extract numeric value
                    clean_part = parts[i].strip('$€£,.;:()-')
                    if any(c.isdigit() for c in clean_part):
                        numeric_parts.append(parts[i])
                
                # If we have numeric parts, try to assign them
                if len(numeric_parts) >= 1:
                    line_item['total'] = numeric_parts[-1]  # Last numeric value is likely total
                
                if len(numeric_parts) >= 2:
                    line_item['unit_price'] = numeric_parts[-2]  # Second to last is likely unit price
                
                if len(numeric_parts) >= 3:
                    line_item['quantity'] = numeric_parts[-3]  # Third to last is likely quantity
                
                # Add the item to our results
                line_items.append(line_item)
            
            return line_items
        
        def _detect_tables(self, image):
            """Detect table regions in an image using OpenCV"""
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Threshold 
            _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
            
            # Dilate to connect nearby elements
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            dilated = cv2.dilate(thresh, kernel, iterations=2)
            
            # Find contours
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Look for rectangular contours that might be tables
            table_regions = []
            for contour in contours:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Tables typically have certain aspect ratios and sizes
                aspect_ratio = w / float(h)
                if 1.0 < aspect_ratio < 10.0 and w > image.shape[1] * 0.3 and h > 50:
                    table_regions.append((x, y, w, h))
            
            return table_regions
            
        def _extract_table_data(self, table_image):
            """Extract structured data from a table image"""
            # This is a simplified implementation - in a real app, would use a more
            # sophisticated method to extract tabular data from images
            
            # Convert to grayscale
            gray = cv2.cvtColor(table_image, cv2.COLOR_BGR2GRAY)
            
            # Get text using pytesseract
            import pytesseract
            data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
            
            # Process the OCR data to extract lines
            # Group text by line
            line_items = []
            
            # Group text data by line
            line_texts = {}
            for i in range(len(data['text'])):
                if data['text'][i].strip():
                    line_num = data['line_num'][i]
                    if line_num not in line_texts:
                        line_texts[line_num] = []
                    line_texts[line_num].append((data['left'][i], data['text'][i]))
            
            # Sort each line by horizontal position and join
            lines = []
            for line_num in sorted(line_texts.keys()):
                sorted_line = sorted(line_texts[line_num])
                text = ' '.join(text for _, text in sorted_line)
                lines.append(text)
                
            # Skip first line which is often a header
            if len(lines) > 1:
                lines = lines[1:]
                
            # Process each line into a line item
            # This is a simplistic approach - real implementation would be more sophisticated
            for line in lines:
                parts = line.split()
                if len(parts) < 2:
                    continue
                    
                # Create a line item
                line_item = {'description': '', 'quantity': '', 'unit_price': '', 'total': ''}
                
                # Simple heuristic parsing as in _detect_line_items_from_text
                desc_end = max(1, int(len(parts) * 0.7))
                line_item['description'] = ' '.join(parts[:desc_end])
                
                numeric_parts = []
                for i in range(desc_end, len(parts)):
                    clean_part = parts[i].strip('$€£,.;:()-')
                    if any(c.isdigit() for c in clean_part):
                        numeric_parts.append(parts[i])
                
                if len(numeric_parts) >= 1:
                    line_item['total'] = numeric_parts[-1]
                if len(numeric_parts) >= 2:
                    line_item['unit_price'] = numeric_parts[-2]
                if len(numeric_parts) >= 3:
                    line_item['quantity'] = numeric_parts[-3]
                
                line_items.append(line_item)
                
            return line_items
            
        def generate_visualization(self, extraction_result):
            """Generate visualization HTML for the extraction results"""
            # Create visualization based on whether enhanced processing was applied
            is_enhanced = extraction_result.get('enhanced', False)
            
            if is_enhanced:
                html = f"""
                <div class="ai-enhanced-preview">
                    <h4>Enhanced Extraction Preview</h4>
                    <div class="alert alert-info">
                        <strong>Success:</strong> Open-source AI enhancement with OpenCV and scikit-learn applied.
                    </div>
                    
                    <div class="card mb-3">
                        <div class="card-header bg-primary text-white">
                            <strong>Extracted Fields</strong>
                        </div>
                        <div class="card-body">
                            <div class="row">
                """
                
                # Add each field
                for field, value in extraction_result.get('fields', {}).items():
                    html += f"""
                                <div class="col-md-6 mb-2">
                                    <div class="input-group">
                                        <span class="input-group-text">{field.replace('_', ' ').title()}</span>
                                        <input type="text" class="form-control" value="{value}" readonly>
                                    </div>
                                </div>
                    """
                
                html += """
                            </div>
                        </div>
                    </div>
                    
                    <div class="card">
                        <div class="card-header bg-primary text-white">
                            <strong>Line Items</strong>
                        </div>
                        <div class="card-body">
                            <div class="table-responsive">
                                <table class="table table-bordered table-striped">
                                    <thead>
                                        <tr>
                                            <th>Description</th>
                                            <th>Quantity</th>
                                            <th>Unit Price</th>
                                            <th>Total</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                """
                
                # Add each line item
                for item in extraction_result.get('line_items', []):
                    html += f"""
                                        <tr>
                                            <td>{item.get('description', '')}</td>
                                            <td>{item.get('quantity', '')}</td>
                                            <td>{item.get('unit_price', '')}</td>
                                            <td>{item.get('total', '')}</td>
                                        </tr>
                    """
                
                html += """
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    </div>
                </div>
                """
                return html
            else:
                # Basic visualization
                return """
                <div class="basic-preview">
                    <h4>Basic Extraction Preview</h4>
                    <div class="alert alert-info">
                        <strong>Note:</strong> Using standard OCR technology for invoice data extraction.
                    </div>
                    <p>The extraction process has identified invoice fields and line items using OCR technology.</p>
                    <p>Review the extracted data below and make any necessary corrections before proceeding.</p>
                </div>
                """
    
    # Set the fallback recognizer
    field_recognizer = FallbackFieldRecognizer()

# Initialize Flask application
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "invoice-ocr-secret-key")

# Configure upload folder
UPLOAD_FOLDER = tempfile.mkdtemp()
if not os.path.exists(os.path.join(UPLOAD_FOLDER, 'uploads')):
    os.makedirs(os.path.join(UPLOAD_FOLDER, 'uploads'))

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pdf', 'tiff'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Field recognizer is already initialized above

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/batch')
def batch():
    return render_template('batch.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Upload and process a single invoice file"""
    # Check if the post request has the file part
    if 'invoice' not in request.files:
        flash('No file part', 'danger')
        return redirect(request.url)
    
    file = request.files['invoice']
    
    # If user does not select file, browser submits an empty file
    if file.filename == '':
        flash('No selected file', 'danger')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        try:
            # Create a unique filename
            filename = str(uuid.uuid4()) + '_' + secure_filename(file.filename)
            
            # Save to uploads folder for web access
            upload_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'uploads')
            if not os.path.exists(upload_dir):
                os.makedirs(upload_dir)
                
            filepath = os.path.join(upload_dir, filename)
            file.save(filepath)
            
            # Save a copy for processing
            process_filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            shutil.copy2(filepath, process_filepath)
            
            # Configure extraction settings based on user options
            ai_enhanced = request.form.get('ai_enhanced', 'true') == 'true'
            enhancement_level = request.form.get('enhancement_level', 'medium')
            recognition_mode = request.form.get('recognition_mode', 'auto')
            
            # Check if enhanced mode is specifically requested
            if recognition_mode == 'enhanced':
                ai_enhanced = True  # Force AI enhancement if enhanced mode is selected
                logger.info("Enhanced OpenCV+ML processing requested")
            
            # Store settings in session (with serializable values only)
            session['extraction_settings'] = {
                'ai_enhanced': ai_enhanced,
                'enhancement_level': enhancement_level, 
                'recognition_mode': recognition_mode
            }
            
            # Measure processing time
            start_time = time.time()
            
            # Initialize with default values
            invoice_data = {
                'header': {},
                'line_items': [],
                'totals': {},
                'invoice_type': 'unknown'
            }
            extraction_preview = "<p>Processing error occurred</p>"
            
            # Process with AI enhancement if available and enabled
            if ai_enhanced and field_recognizer is not None:
                try:
                    # Use AI-enhanced field recognition
                    logger.info(f"Processing file with AI enhancement: {process_filepath}")
                    extraction_result = field_recognizer.process_invoice(process_filepath)
                    
                    # Convert AI result format to the expected invoice_data format
                    invoice_data = {
                        'header': {},
                        'line_items': [],
                        'totals': {},
                        'invoice_type': 'unknown'
                    }
                    
                    # Handle line_items safely - ensure they're serializable
                    line_items_raw = extraction_result.get('line_items', [])
                    serializable_line_items = []
                    
                    # Process line items into a guaranteed serializable format
                    for item in line_items_raw:
                        if isinstance(item, dict):
                            # Dictionary items can be used directly if all values are strings
                            try:
                                # Ensure all values are strings
                                clean_item = {}
                                for k, v in item.items():
                                    clean_item[k] = str(v) if v is not None else ""
                                serializable_line_items.append(clean_item)
                            except Exception as e:
                                logger.error(f"Error processing line item dict: {e}")
                                serializable_line_items.append({"description": "Error processing item"})
                        elif isinstance(item, str):
                            # Process string items into structured format
                            parts = item.split('-')
                            if len(parts) >= 2:
                                serializable_line_items.append({
                                    "product_code": parts[0].strip(),
                                    "description": parts[1].strip(),
                                    "quantity": "1",
                                    "unit_price": "0",
                                    "total": "0"
                                })
                            else:
                                serializable_line_items.append({
                                    "description": item.strip(),
                                    "product_code": "",
                                    "quantity": "1",
                                    "unit_price": "0",
                                    "total": "0"
                                })
                        else:
                            # Handle other types
                            try:
                                serializable_line_items.append({
                                    "description": str(item),
                                    "product_code": "",
                                    "quantity": "1",
                                    "unit_price": "0",
                                    "total": "0"
                                })
                            except Exception as e:
                                logger.error(f"Failed to convert line item to string: {e}")
                    
                    # Use the serializable line items
                    invoice_data['line_items'] = serializable_line_items
                    
                    # Map AI-extracted fields to invoice header format - with type safety
                    fields = extraction_result.get('fields', {})
                    for field, value in fields.items():
                        try:
                            field_str = str(field)
                            value_str = str(value) if value is not None else ""
                            
                            if field_str == 'invoice_number':
                                invoice_data['header']['invoice_number'] = value_str
                            elif field_str == 'date':
                                invoice_data['header']['date'] = value_str
                            elif field_str == 'total':
                                invoice_data['totals']['grand_total'] = value_str
                            else:
                                # Add other fields to header for future use
                                invoice_data['header'][field_str] = value_str
                        except Exception as field_err:
                            logger.error(f"Error processing field {field}: {field_err}")
                    
                    # Add the visualization HTML - with error handling
                    try:
                        extraction_preview = field_recognizer.generate_visualization(extraction_result)
                    except Exception as vis_err:
                        logger.error(f"Error generating visualization: {vis_err}")
                        extraction_preview = "<p>AI-enhanced visualization failed, but data extraction completed</p>"
                except Exception as e:
                    logger.error(f"Error in AI enhancement, falling back to traditional OCR: {e}")
                    # Fall back to traditional OCR
                    try:
                        invoice_data = process_image(process_filepath)
                        # Ensure all values are strings for JSON serialization
                        for section in ['header', 'totals']:
                            for k, v in invoice_data.get(section, {}).items():
                                invoice_data[section][k] = str(v) if v is not None else ""
                                
                        # Ensure line items are properly serializable
                        safe_line_items = []
                        for item in invoice_data.get('line_items', []):
                            if isinstance(item, dict):
                                safe_item = {}
                                for k, v in item.items():
                                    safe_item[k] = str(v) if v is not None else ""
                                safe_line_items.append(safe_item)
                            else:
                                safe_line_items.append({"description": str(item)})
                        invoice_data['line_items'] = safe_line_items
                        
                        extraction_preview = "<p>AI-enhanced visualization failed: using traditional OCR results</p>"
                    except Exception as ocr_err:
                        logger.error(f"Traditional OCR also failed: {ocr_err}")
                        # Create minimum viable data
                        invoice_data = {
                            'header': {'invoice_number': os.path.basename(process_filepath)},
                            'line_items': [{'description': 'OCR processing failed', 'quantity': '0', 'total': '0'}],
                            'totals': {'grand_total': '0'},
                            'invoice_type': 'unknown'
                        }
                        extraction_preview = "<p>Invoice processing failed. Please try another image.</p>"
            else:
                # Use traditional OCR processing with enhanced error handling
                ai_reason = "disabled by user" if not ai_enhanced else "not available"
                logger.info(f"Processing file with traditional OCR ({ai_reason}): {process_filepath}")
                try:
                    invoice_data = process_image(process_filepath)
                    
                    # Ensure all values are strings for JSON serialization
                    for section in ['header', 'totals']:
                        for k, v in invoice_data.get(section, {}).items():
                            invoice_data[section][k] = str(v) if v is not None else ""
                            
                    # Ensure line items are properly serializable
                    safe_line_items = []
                    for item in invoice_data.get('line_items', []):
                        if isinstance(item, dict):
                            safe_item = {}
                            for k, v in item.items():
                                safe_item[k] = str(v) if v is not None else ""
                            safe_line_items.append(safe_item)
                        else:
                            safe_line_items.append({"description": str(item)})
                    invoice_data['line_items'] = safe_line_items
                    
                    extraction_preview = f"<p>AI-enhanced visualization {ai_reason}</p>"
                except Exception as ocr_err:
                    logger.error(f"Traditional OCR failed: {ocr_err}")
                    # Create minimum viable data
                    invoice_data = {
                        'header': {'invoice_number': os.path.basename(process_filepath)},
                        'line_items': [{'description': 'OCR processing failed', 'quantity': '0', 'total': '0'}],
                        'totals': {'grand_total': '0'},
                        'invoice_type': 'unknown'
                    }
                    extraction_preview = "<p>Invoice processing failed. Please try another image.</p>"
            
            # Calculate processing time
            processing_time = round(time.time() - start_time, 2)
            
            # Calculate confidence based on the amount of data extracted
            fields_detected = len(invoice_data.get('header', {}))
            line_items_count = len(invoice_data.get('line_items', []))
            
            # Simple confidence calculation based on data extracted
            if fields_detected > 4 and line_items_count > 5:
                confidence = 85
            elif fields_detected > 2 and line_items_count > 2:
                confidence = 65
            else:
                confidence = 40
            
            # Store stats for the preview page
            stats = {
                'fields_detected': fields_detected,
                'line_items': line_items_count,
                'confidence': confidence,
                'processing_time': processing_time
            }
            
            # Store data in session for further processing - using json to ensure it's serializable
            session['invoice_data'] = invoice_data
            session['file_path'] = process_filepath
            session['filename'] = filename
            session['extraction_preview'] = extraction_preview
            session['stats'] = stats
            
            # Save session immediately to catch any serialization issues
            session.modified = True
            
            logger.info("Successfully processed invoice. Redirecting to preview page.")
            # Redirect to preview or review page based on setting
            return redirect(url_for('preview_extraction'))
            
        except Exception as e:
            logger.error(f"Error processing invoice: {str(e)}")
            flash(f'Error processing invoice: {str(e)}', 'danger')
            return redirect(url_for('index'))
    else:
        flash('File type not allowed. Please upload a PNG, JPG, JPEG, PDF, or TIFF file.', 'danger')
        return redirect(url_for('index'))

@app.route('/batch_upload', methods=['POST'])
def batch_upload():
    # Check if the post request has files
    if 'invoices' not in request.files:
        flash('No files selected', 'danger')
        return redirect(url_for('batch'))
    
    files = request.files.getlist('invoices')
    
    # Check if any files were selected
    if len(files) == 0 or files[0].filename == '':
        flash('No selected files', 'danger')
        return redirect(url_for('batch'))
    
    processed_files = []
    failed_files = []
    all_invoice_data = []
    
    for file in files:
        if file and allowed_file(file.filename):
            try:
                # Create a unique filename
                filename = str(uuid.uuid4()) + '_' + secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                # Process the invoice image with enhanced OCR
                logger.info(f"Processing file: {filepath}")
                invoice_data = process_image(filepath)
                
                # Log the invoice type detected and number of line items found for debugging
                invoice_type = invoice_data.get('invoice_type', 'unknown')
                line_items_count = len(invoice_data.get('line_items', []))
                logger.info(f"Processed {file.filename}: Type={invoice_type}, Items={line_items_count}")
                
                all_invoice_data.append(invoice_data)
                processed_files.append(file.filename)
                
            except Exception as e:
                logger.error(f"Error processing {file.filename}: {str(e)}")
                failed_files.append((file.filename, str(e)))
        else:
            failed_files.append((file.filename, "Invalid file format"))
    
    # Store all invoice data in session
    if all_invoice_data:
        session['batch_invoice_data'] = all_invoice_data
        
        # Generate a combined Excel file
        excel_file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'combined_invoices.xlsx')
        create_excel_file(all_invoice_data, excel_file_path, batch_mode=True)
        session['excel_file_path'] = excel_file_path
        
        logger.info(f"Successfully processed {len(processed_files)} invoices into Excel")
    else:
        logger.warning("No invoices were successfully processed")
    
    # Prepare summary data for template
    summary = {
        'processed': processed_files,
        'failed': failed_files,
        'total_processed': len(processed_files),
        'total_failed': len(failed_files)
    }
    
    return render_template('batch.html', summary=summary, has_results=True)

@app.route('/review')
def review_data():
    # Get invoice data from session
    invoice_data = session.get('invoice_data')
    
    if not invoice_data:
        flash('No invoice data to review. Please upload an invoice first.', 'warning')
        return redirect(url_for('index'))
    
    # Transform the data for easier display in the template
    header_data = invoice_data.get('header', {})
    line_items = invoice_data.get('line_items', [])
    totals = invoice_data.get('totals', {})
    invoice_type = invoice_data.get('invoice_type', 'unknown')
    
    # Log for debugging
    logger.debug(f"Reviewing invoice of type '{invoice_type}' with {len(line_items)} line items")
    
    return render_template('results.html', 
                          header=header_data, 
                          line_items=line_items, 
                          totals=totals,
                          invoice_type=invoice_type)

@app.route('/update_data', methods=['POST'])
def update_data():
    invoice_data = session.get('invoice_data', {})
    
    # Update header information
    header = {}
    for key in ['invoice_number', 'date', 'customer_name', 'customer_ref', 'vat_reg_no', 'business_reg_no']:
        header[key] = request.form.get(f'header_{key}', '')
    
    # Update line items
    line_items = []
    item_count = int(request.form.get('item_count', 0))
    
    for i in range(item_count):
        # Sanitize numeric values to handle OCR errors
        quantity = request.form.get(f'item_{i}_quantity', '')
        unit_price = request.form.get(f'item_{i}_unit_price', '')
        discount = request.form.get(f'item_{i}_discount', '')
        total = request.form.get(f'item_{i}_total', '')
        
        # Clean numeric values
        try:
            quantity = clean_numeric_value(quantity)
            unit_price = clean_numeric_value(unit_price)
            discount = clean_numeric_value(discount)
            total = clean_numeric_value(total)
        except Exception as e:
            logger.warning(f"Error cleaning numeric values: {str(e)}")
        
        item = {
            'product_code': request.form.get(f'item_{i}_product_code', ''),
            'quantity': quantity,
            'description': request.form.get(f'item_{i}_description', ''),
            'unit_price': unit_price,
            'discount': discount,
            'total': total
        }
        line_items.append(item)
    
    # Update totals
    totals = {}
    for key in ['subtotal', 'vat', 'grand_total']:
        totals[key] = clean_numeric_value(request.form.get(f'totals_{key}', ''))
    
    # Preserve the invoice type if it exists in the original data
    invoice_type = invoice_data.get('invoice_type', 'unknown')
    
    # Update the invoice data in session
    invoice_data['header'] = header
    invoice_data['line_items'] = line_items
    invoice_data['totals'] = totals
    invoice_data['invoice_type'] = invoice_type  # Keep the invoice type
    session['invoice_data'] = invoice_data
    
    try:
        # Generate Excel file
        file_path = session.get('file_path')
        if file_path:
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            excel_file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{base_name}.xlsx")
            create_excel_file(invoice_data, excel_file_path)
            session['excel_file_path'] = excel_file_path
            
            flash('Data updated successfully and Excel file generated!', 'success')
        else:
            flash('Data updated but could not generate Excel file.', 'warning')
    except Exception as e:
        logger.error(f"Error generating Excel file: {str(e)}")
        flash(f'Error generating Excel file: {str(e)}. Please check the data for any invalid values.', 'danger')
    
    return redirect(url_for('review_data'))

# Helper function to clean numeric values from OCR
def clean_numeric_value(value):
    """Clean numeric values that might contain OCR errors"""
    if not value:
        return '0'
        
    # Remove any non-numeric characters except decimal point and comma
    value = ''.join(c for c in value if c.isdigit() or c in '.,')
    
    # Replace comma with dot for decimal
    value = value.replace(',', '.')
    
    # If empty after cleaning, return zero
    if not value:
        return '0'
        
    try:
        # Try to convert to float to validate
        float_val = float(value)
        return str(float_val)
    except:
        # If conversion fails, return as is
        return value

@app.route('/download_excel')
def download_excel():
    excel_file_path = session.get('excel_file_path')
    
    if not excel_file_path or not os.path.exists(excel_file_path):
        flash('Excel file not found. Please process your invoice data first.', 'danger')
        return redirect(url_for('index'))
    
    filename = os.path.basename(excel_file_path)
    return send_file(excel_file_path, 
                    mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                    as_attachment=True,
                    download_name=filename)

@app.route('/download_batch_excel')
def download_batch_excel():
    excel_file_path = session.get('excel_file_path')
    
    if not excel_file_path or not os.path.exists(excel_file_path):
        flash('Excel file not found. Please process your invoices first.', 'danger')
        return redirect(url_for('batch'))
    
    return send_file(excel_file_path, 
                    mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                    as_attachment=True,
                    download_name='combined_invoices.xlsx')

@app.route('/preview')
def preview_extraction():
    """
    Preview the extraction results with interactive visualization
    """
    try:
        # Get data from session with safe defaults
        invoice_data = session.get('invoice_data', {})
        extraction_preview = session.get('extraction_preview', '<p>No preview available</p>')
        filename = session.get('filename', '')
        stats = session.get('stats', {
            'fields_detected': 0,
            'line_items': 0,
            'confidence': 0,
            'processing_time': 0
        })
        
        if not invoice_data:
            flash('No extraction data to preview. Please upload an invoice first.', 'warning')
            return redirect(url_for('index'))
        
        # Prepare data for template with safe defaults and type checks
        if not isinstance(invoice_data, dict):
            logger.error(f"Invalid invoice_data type: {type(invoice_data)}")
            invoice_data = {
                'header': {'error': 'Invalid data format'},
                'line_items': [{'description': 'Data processing error'}],
                'totals': {'grand_total': '0'}
            }
            
        header_data = invoice_data.get('header', {})
        if not isinstance(header_data, dict):
            header_data = {'error': 'Invalid header format'}
            
        line_items = invoice_data.get('line_items', [])
        if not isinstance(line_items, list):
            line_items = [{'description': 'Invalid line items format'}]
            
        totals = invoice_data.get('totals', {})
        if not isinstance(totals, dict):
            totals = {'grand_total': '0'}
        
        # Get original image path for display with safety checks
        uploaded_image_path = None
        if filename and isinstance(filename, str):
            # Sanitize filename for URL
            safe_filename = filename.replace('..', '').replace('/', '_')
            uploaded_image_path = url_for('static', filename=f'uploads/{safe_filename}')
        
        # Ensure extraction_preview is a string
        if not isinstance(extraction_preview, str):
            try:
                extraction_preview = str(extraction_preview)
            except:
                extraction_preview = "<p>Preview generation error</p>"
        
        logger.info("Rendering preview template with extracted data")
        return render_template('preview.html',
                            header=header_data,
                            line_items=line_items,
                            totals=totals,
                            extraction_preview=extraction_preview,
                            uploaded_image=uploaded_image_path,
                            stats=stats)
                            
    except Exception as e:
        logger.error(f"Error in preview_extraction: {e}")
        flash(f"An error occurred while preparing the preview: {str(e)}", "danger")
        return redirect(url_for('index'))

@app.route('/accept_extraction', methods=['POST'])
def accept_extraction():
    """
    Accept the extraction results and proceed to review
    """
    return redirect(url_for('review_data'))

@app.route('/reset')
def reset():
    # Clear session data
    session.pop('invoice_data', None)
    session.pop('file_path', None)
    session.pop('excel_file_path', None)
    session.pop('batch_invoice_data', None)
    session.pop('extraction_preview', None)
    session.pop('filename', None)
    session.pop('stats', None)
    session.pop('extraction_settings', None)
    
    flash('Data has been reset. You can upload a new invoice.', 'info')
    return redirect(url_for('index'))

@app.route('/image/<filename>')
def serve_image(filename):
    """Serve invoice images for visualization"""
    # Get the file path from session
    file_path = session.get('file_path', '')
    
    # Check if this is the image we're looking for
    if file_path and os.path.basename(file_path) == filename:
        try:
            return send_file(file_path, mimetype='image/jpeg')
        except Exception as e:
            logger.error(f"Error serving image file: {e}")
            return "Image not found", 404
    
    # Try to locate the file in the temp directory
    temp_dir = '/tmp'
    for root, dirs, files in os.walk(temp_dir):
        if filename in files:
            full_path = os.path.join(root, filename)
            try:
                return send_file(full_path, mimetype='image/jpeg')
            except Exception as e:
                logger.error(f"Error serving image file: {e}")
                break
    
    return "Image not found", 404

@app.errorhandler(413)
def request_entity_too_large(error):
    flash('File too large. Maximum file size is 16MB.', 'danger')
    return redirect(url_for('index')), 413

@app.teardown_appcontext
def cleanup(exception):
    # Clean up temporary files periodically
    # In a production app, you'd want a more sophisticated approach
    pass
