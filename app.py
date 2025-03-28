import os
import logging
import uuid
from flask import Flask, render_template, request, redirect, url_for, flash, session, send_file
import pandas as pd
from werkzeug.utils import secure_filename
import tempfile
import shutil

from invoice_parser import process_image
from excel_generator import create_excel_file

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize Flask application
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "invoice-ocr-secret-key")

# Configure upload folder
UPLOAD_FOLDER = tempfile.mkdtemp()
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pdf', 'tiff'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

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
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Process the invoice image
            logger.info(f"Processing file: {filepath}")
            invoice_data = process_image(filepath)
            
            # Store invoice data in session for review
            session['invoice_data'] = invoice_data
            session['file_path'] = filepath
            
            return redirect(url_for('review_data'))
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
                
                # Process the invoice image
                invoice_data = process_image(filepath)
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
    
    return render_template('results.html', 
                          header=header_data, 
                          line_items=line_items, 
                          totals=totals)

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
        item = {
            'product_code': request.form.get(f'item_{i}_product_code', ''),
            'quantity': request.form.get(f'item_{i}_quantity', ''),
            'description': request.form.get(f'item_{i}_description', ''),
            'unit_price': request.form.get(f'item_{i}_unit_price', ''),
            'discount': request.form.get(f'item_{i}_discount', ''),
            'total': request.form.get(f'item_{i}_total', '')
        }
        line_items.append(item)
    
    # Update totals
    totals = {}
    for key in ['subtotal', 'vat', 'grand_total']:
        totals[key] = request.form.get(f'totals_{key}', '')
    
    # Update the invoice data in session
    invoice_data['header'] = header
    invoice_data['line_items'] = line_items
    invoice_data['totals'] = totals
    session['invoice_data'] = invoice_data
    
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
    
    return redirect(url_for('review_data'))

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

@app.route('/reset')
def reset():
    # Clear session data
    session.pop('invoice_data', None)
    session.pop('file_path', None)
    session.pop('excel_file_path', None)
    session.pop('batch_invoice_data', None)
    
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
