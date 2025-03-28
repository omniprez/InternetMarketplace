import os
import logging
import uuid
import time
import json
import sys
import tempfile
import shutil
import base64

# Setup logging first
logging.basicConfig(level=logging.DEBUG)
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

# Import Anthropic if available for potential Claude-based enhancements
anthropic_client = None
try:
    import anthropic
    # the newest Anthropic model is "claude-3-5-sonnet-20241022" which was released October 22, 2024
    ANTHROPIC_KEY = os.environ.get('ANTHROPIC_API_KEY')
    if ANTHROPIC_KEY:
        try:
            anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
            logger.info("Successfully initialized Anthropic client")
        except Exception as e:
            logger.error(f"Failed to initialize Anthropic client with provided key: {e}")
            logger.warning("Claude-based enhancements will be disabled")
    else:
        logger.warning("Anthropic API key not found in environment variables. Claude-based enhancements will be disabled.")
except ImportError as e:
    logger.warning(f"Anthropic library not available: {e}")
except Exception as e:
    logger.error(f"Error importing Anthropic module: {e}")

# Try to import the AI field recognizer with graceful fallback
field_recognizer = None
try:
    from ai_field_recognition import AIFieldRecognizer
    field_recognizer = AIFieldRecognizer()
    logger.info("Successfully initialized AI field recognizer")
except Exception as e:
    logger.error(f"Failed to initialize AI field recognizer: {e}")
    logger.warning("Continuing without AI field recognition capabilities")
    
    # Create enhanced fallback for the field_recognizer with Anthropic capabilities if available
    class FallbackFieldRecognizer:
        def __init__(self):
            # Store reference to Anthropic client from outer scope
            self.anthropic_client = anthropic_client
            
            # Set use_claude based on whether client is available
            self.use_claude = False
            if self.anthropic_client is not None:
                try:
                    # Verify the client is properly initialized by checking an attribute
                    client_vars = dir(self.anthropic_client)
                    if 'messages' in client_vars or hasattr(self.anthropic_client, 'messages'):
                        self.use_claude = True
                        logger.info("FallbackFieldRecognizer will use Claude for enhanced extraction")
                    else:
                        logger.warning("Anthropic client lacks expected 'messages' attribute")
                        logger.info("FallbackFieldRecognizer will use traditional OCR only")
                except Exception as e:
                    logger.error(f"Error verifying Anthropic client: {e}")
                    logger.info("FallbackFieldRecognizer will use traditional OCR only")
            else:
                logger.info("Anthropic client not available, FallbackFieldRecognizer will use traditional OCR only")
                
        def process_invoice(self, image_path):
            """Process the invoice with the basic OCR and optional Claude enhancement"""
            logger.info(f"Using fallback processor for {image_path}")
            
            # Delegate to the standard invoice processor first
            from invoice_parser import process_image
            result = process_image(image_path)
            
            # Create basic extraction result
            extraction_result = {
                'fields': {k: v for k, v in result.get('header', {}).items()},
                'line_items': result.get('line_items', []),
                'visualization_data': None
            }
            
            # Use Claude to enhance extraction if available
            if self.use_claude and os.path.exists(image_path):
                try:
                    logger.info("Attempting Claude-based enhancement")
                    # Verify the Anthropic client is properly initialized
                    if self.anthropic_client is None:
                        logger.error("Anthropic client is None despite use_claude being True")
                        raise ValueError("Anthropic client not properly initialized")
                    
                    # Read the image as base64 for Claude
                    with open(image_path, "rb") as img_file:
                        base64_image = base64.b64encode(img_file.read()).decode("utf-8")
                    
                    logger.info("Image read successfully, sending to Claude API")
                    
                    # Check which model to use - the newest model is preferred but fallback to older models if needed
                    # the newest Anthropic model is "claude-3-5-sonnet-20241022" which was released October 22, 2024
                    model_to_use = "claude-3-opus-20240229"  # Fallback to a known model that should exist 
                    
                    try:
                        # Try to get available models
                        logger.info(f"Using model: {model_to_use}")
                    except Exception as model_err:
                        logger.warning(f"Error getting models: {model_err}, using default model")
                    
                    # Create the message request
                    response = self.anthropic_client.messages.create(
                        model=model_to_use,
                        max_tokens=1500,
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": """Please extract all relevant invoice information from this image. 
                                        Return the data in this exact JSON format:
                                        {
                                            "invoice_number": "...",
                                            "date": "...",
                                            "vendor": "...",
                                            "total": "...",
                                            "line_items": [
                                                {"description": "...", "quantity": "...", "unit_price": "...", "total": "..."},
                                                ...
                                            ]
                                        }
                                        
                                        Only output valid JSON - no explanations or other text. If you can't determine a field, use null or an empty string."""
                                    },
                                    {
                                        "type": "image",
                                        "source": {
                                            "type": "base64",
                                            "media_type": "image/jpeg",
                                            "data": base64_image
                                        }
                                    }
                                ]
                            }
                        ]
                    )
                    
                    # Extract the JSON response text
                    json_text = response.content[0].text
                    
                    # Try to parse the JSON
                    try:
                        claude_data = json.loads(json_text)
                        
                        # Claude produced nice JSON - merge with our OCR results
                        # For fields, prefer Claude's extraction if it has values
                        if claude_data.get('invoice_number'):
                            extraction_result['fields']['invoice_number'] = claude_data['invoice_number']
                        if claude_data.get('date'):
                            extraction_result['fields']['date'] = claude_data['date']
                        if claude_data.get('vendor'):
                            extraction_result['fields']['vendor'] = claude_data['vendor']
                        if claude_data.get('total'):
                            extraction_result['fields']['total'] = claude_data['total']
                        
                        # For line items, use Claude's if it found more than what we already have
                        claude_line_items = claude_data.get('line_items', [])
                        if len(claude_line_items) > len(extraction_result['line_items']):
                            # Format Claude's line items to match our expected format
                            formatted_items = []
                            for item in claude_line_items:
                                formatted_item = {
                                    'description': item.get('description', ''),
                                    'quantity': item.get('quantity', ''),
                                    'unit_price': item.get('unit_price', ''),
                                    'total': item.get('total', '')
                                }
                                formatted_items.append(formatted_item)
                            extraction_result['line_items'] = formatted_items
                            
                        logger.info("Successfully enhanced extraction with Claude")
                        # Store the raw Claude response for visualization
                        extraction_result['claude_response'] = json_text
                    except json.JSONDecodeError as e:
                        logger.warning(f"Could not parse Claude response as JSON: {e}")
                except Exception as e:
                    logger.error(f"Error using Claude for enhancement: {e}")
            
            return extraction_result
            
        def generate_visualization(self, extraction_result):
            """Generate visualization HTML for the extraction results"""
            # If we have Claude data, show a fancier visualization
            if self.use_claude and extraction_result.get('claude_response'):
                html = f"""
                <div class="ai-enhanced-preview">
                    <h4>AI-Enhanced Extraction Preview</h4>
                    <div class="alert alert-info">
                        <strong>Success:</strong> Claude AI has enhanced this extraction with advanced vision capabilities.
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
                    <div class="alert alert-warning">
                        <strong>Note:</strong> Using traditional OCR. For enhanced accuracy, enable Anthropic Claude AI in settings.
                    </div>
                    <p>The basic extraction process has identified invoice fields and line items using OCR technology.</p>
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

@app.errorhandler(413)
def request_entity_too_large(error):
    flash('File too large. Maximum file size is 16MB.', 'danger')
    return redirect(url_for('index')), 413

@app.teardown_appcontext
def cleanup(exception):
    # Clean up temporary files periodically
    # In a production app, you'd want a more sophisticated approach
    pass
