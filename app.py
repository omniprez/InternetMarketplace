import os
import logging
import uuid
import time
import json
import sys
import tempfile
import shutil

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

# Try to import the AI field recognizer with graceful fallback
field_recognizer = None
try:
    from ai_field_recognition import AIFieldRecognizer
    field_recognizer = AIFieldRecognizer()
    logger.info("Successfully initialized AI field recognizer")
except Exception as e:
    logger.error(f"Failed to initialize AI field recognizer: {e}")
    logger.warning("Continuing without AI field recognition capabilities")
    
    # Create minimal fallback for the field_recognizer
    class FallbackFieldRecognizer:
        def process_invoice(self, image_path):
            logger.warning(f"Using fallback processor for {image_path}")
            # Delegate to the standard invoice processor
            from invoice_parser import process_image
            result = process_image(image_path)
            # Convert to AI field recognizer format
            return {
                'fields': {k: v for k, v in result.get('header', {}).items()},
                'line_items': result.get('line_items', []),
                'visualization_data': None
            }
            
        def generate_visualization(self, extraction_result):
            return "<p>AI visualization not available - using fallback processor</p>"
    
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
        
        try:
            # Configure extraction settings based on user options
            ai_enhanced = request.form.get('ai_enhanced', 'true') == 'true'
            enhancement_level = request.form.get('enhancement_level', 'medium')
            recognition_mode = request.form.get('recognition_mode', 'auto')
            
            # Store settings in session
            session['extraction_settings'] = {
                'ai_enhanced': ai_enhanced,
                'enhancement_level': enhancement_level,
                'recognition_mode': recognition_mode
            }
            
            # Measure processing time
            start_time = time.time()
            
            if ai_enhanced and field_recognizer is not None:
                try:
                    # Use AI-enhanced field recognition
                    logger.info(f"Processing file with AI enhancement: {process_filepath}")
                    extraction_result = field_recognizer.process_invoice(process_filepath)
                    
                    # Convert AI result format to the expected invoice_data format
                    invoice_data = {
                        'header': {},
                        'line_items': extraction_result.get('line_items', []),
                        'totals': {},
                        'invoice_type': 'unknown'
                    }
                    
                    # Map AI-extracted fields to invoice header format
                    fields = extraction_result.get('fields', {})
                    for field, value in fields.items():
                        if field == 'invoice_number':
                            invoice_data['header']['invoice_number'] = value
                        elif field == 'date':
                            invoice_data['header']['date'] = value
                        elif field == 'total':
                            invoice_data['totals']['grand_total'] = value
                    
                    # Add the visualization HTML
                    extraction_preview = field_recognizer.generate_visualization(extraction_result)
                except Exception as e:
                    logger.error(f"Error in AI enhancement, falling back to traditional OCR: {e}")
                    # Fall back to traditional OCR
                    invoice_data = process_image(process_filepath)
                    extraction_preview = "<p>AI-enhanced visualization failed: falling back to traditional OCR</p>"
            else:
                # Use traditional OCR processing
                ai_reason = "disabled by user" if not ai_enhanced else "not available"
                logger.info(f"Processing file with traditional OCR ({ai_reason}): {process_filepath}")
                invoice_data = process_image(process_filepath)
                extraction_preview = f"<p>AI-enhanced visualization {ai_reason}</p>"
            
            # Calculate processing time and confidence
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
            
            # Store data in session for further processing
            session['invoice_data'] = invoice_data
            session['file_path'] = process_filepath
            session['filename'] = filename
            session['extraction_preview'] = extraction_preview
            session['stats'] = stats
            
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
    # Get data from session
    invoice_data = session.get('invoice_data')
    extraction_preview = session.get('extraction_preview')
    filename = session.get('filename')
    stats = session.get('stats')
    
    if not invoice_data or not extraction_preview:
        flash('No extraction data to preview. Please upload an invoice first.', 'warning')
        return redirect(url_for('index'))
    
    # Prepare data for template
    header_data = invoice_data.get('header', {})
    line_items = invoice_data.get('line_items', [])
    totals = invoice_data.get('totals', {})
    
    # Get original image path for display
    uploaded_image_path = None
    if filename:
        uploaded_image_path = url_for('static', filename=f'uploads/{filename}')
    
    return render_template('preview.html',
                         header=header_data,
                         line_items=line_items,
                         totals=totals,
                         extraction_preview=extraction_preview,
                         uploaded_image=uploaded_image_path,
                         stats=stats)

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
