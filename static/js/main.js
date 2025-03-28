// Main JavaScript for Invoice OCR Application

// Wait for the DOM to be fully loaded
document.addEventListener('DOMContentLoaded', function() {
    // File upload handling
    setupFileUpload();
    
    // Batch upload handling
    setupBatchUpload();
    
    // Preview image functionality
    setupImagePreview();
    
    // Form submission with loading animation
    setupFormSubmission();
});

/**
 * Set up file upload functionality with drag & drop
 */
function setupFileUpload() {
    const uploadArea = document.getElementById('upload-area');
    const fileInput = document.getElementById('invoice-file');
    
    if (!uploadArea || !fileInput) return;
    
    // Handle click on upload area
    uploadArea.addEventListener('click', function() {
        fileInput.click();
    });
    
    // Handle file selection
    fileInput.addEventListener('change', function() {
        handleFileUpload(this.files);
    });
    
    // Handle drag & drop events
    uploadArea.addEventListener('dragover', function(e) {
        e.preventDefault();
        e.stopPropagation();
        this.classList.add('drag-over');
    });
    
    uploadArea.addEventListener('dragleave', function(e) {
        e.preventDefault();
        e.stopPropagation();
        this.classList.remove('drag-over');
    });
    
    uploadArea.addEventListener('drop', function(e) {
        e.preventDefault();
        e.stopPropagation();
        this.classList.remove('drag-over');
        
        const files = e.dataTransfer.files;
        handleFileUpload(files);
    });
}

/**
 * Set up batch file upload functionality
 */
function setupBatchUpload() {
    const batchUploadArea = document.getElementById('batch-upload-area');
    const batchFileInput = document.getElementById('batch-invoice-files');
    
    if (!batchUploadArea || !batchFileInput) return;
    
    // Handle click on upload area
    batchUploadArea.addEventListener('click', function() {
        batchFileInput.click();
    });
    
    // Handle file selection
    batchFileInput.addEventListener('change', function() {
        handleBatchFileUpload(this.files);
    });
    
    // Handle drag & drop events
    batchUploadArea.addEventListener('dragover', function(e) {
        e.preventDefault();
        e.stopPropagation();
        this.classList.add('drag-over');
    });
    
    batchUploadArea.addEventListener('dragleave', function(e) {
        e.preventDefault();
        e.stopPropagation();
        this.classList.remove('drag-over');
    });
    
    batchUploadArea.addEventListener('drop', function(e) {
        e.preventDefault();
        e.stopPropagation();
        this.classList.remove('drag-over');
        
        const files = e.dataTransfer.files;
        handleBatchFileUpload(files);
    });
}

/**
 * Handle file upload for single file
 * @param {FileList} files - The files to upload
 */
function handleFileUpload(files) {
    if (files.length === 0) return;
    
    const file = files[0];
    const fileNameElement = document.getElementById('file-name');
    const previewImage = document.getElementById('preview-image');
    const uploadForm = document.getElementById('upload-form');
    
    // Check file type
    const validTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/tiff', 'application/pdf'];
    if (!validTypes.includes(file.type)) {
        alert('Please upload a valid file type (JPG, PNG, TIFF, or PDF)');
        return;
    }
    
    // Update UI to show selected file
    if (fileNameElement) {
        fileNameElement.textContent = file.name;
    }
    
    // Show preview if it's an image
    if (previewImage && file.type.startsWith('image/')) {
        const reader = new FileReader();
        reader.onload = function(e) {
            previewImage.src = e.target.result;
            previewImage.style.display = 'block';
        };
        reader.readAsDataURL(file);
    }
    
    // Enable submit button
    const submitBtn = document.querySelector('#upload-form button[type="submit"]');
    if (submitBtn) {
        submitBtn.disabled = false;
    }
}

/**
 * Handle batch file upload
 * @param {FileList} files - The files to upload
 */
function handleBatchFileUpload(files) {
    if (files.length === 0) return;
    
    const fileCountElement = document.getElementById('file-count');
    const fileListElement = document.getElementById('file-list');
    
    // Check file types
    const validTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/tiff', 'application/pdf'];
    let allValid = true;
    
    for (let i = 0; i < files.length; i++) {
        if (!validTypes.includes(files[i].type)) {
            allValid = false;
            break;
        }
    }
    
    if (!allValid) {
        alert('Please ensure all files are valid types (JPG, PNG, TIFF, or PDF)');
        return;
    }
    
    // Update UI to show selected files
    if (fileCountElement) {
        fileCountElement.textContent = files.length;
    }
    
    // Display file list
    if (fileListElement) {
        fileListElement.innerHTML = '';
        for (let i = 0; i < files.length; i++) {
            const listItem = document.createElement('li');
            listItem.className = 'list-group-item';
            listItem.textContent = files[i].name;
            fileListElement.appendChild(listItem);
        }
        fileListElement.style.display = 'block';
    }
    
    // Enable submit button
    const submitBtn = document.querySelector('#batch-upload-form button[type="submit"]');
    if (submitBtn) {
        submitBtn.disabled = false;
    }
}

/**
 * Set up image preview functionality
 */
function setupImagePreview() {
    const previewImage = document.getElementById('preview-image');
    if (!previewImage) return;
    
    // Set up click to enlarge
    previewImage.addEventListener('click', function() {
        // Create modal for enlarged view
        const modal = document.createElement('div');
        modal.className = 'modal fade';
        modal.id = 'imageModal';
        modal.tabIndex = '-1';
        modal.setAttribute('aria-hidden', 'true');
        
        const modalDialog = document.createElement('div');
        modalDialog.className = 'modal-dialog modal-lg';
        
        const modalContent = document.createElement('div');
        modalContent.className = 'modal-content';
        
        const modalHeader = document.createElement('div');
        modalHeader.className = 'modal-header';
        
        const modalTitle = document.createElement('h5');
        modalTitle.className = 'modal-title';
        modalTitle.textContent = 'Invoice Preview';
        
        const closeButton = document.createElement('button');
        closeButton.type = 'button';
        closeButton.className = 'btn-close';
        closeButton.setAttribute('data-bs-dismiss', 'modal');
        closeButton.setAttribute('aria-label', 'Close');
        
        const modalBody = document.createElement('div');
        modalBody.className = 'modal-body';
        
        const enlargedImage = document.createElement('img');
        enlargedImage.src = previewImage.src;
        enlargedImage.className = 'img-fluid';
        enlargedImage.alt = 'Invoice Preview';
        
        // Assemble the modal
        modalHeader.appendChild(modalTitle);
        modalHeader.appendChild(closeButton);
        modalBody.appendChild(enlargedImage);
        modalContent.appendChild(modalHeader);
        modalContent.appendChild(modalBody);
        modalDialog.appendChild(modalContent);
        modal.appendChild(modalDialog);
        
        // Add to document and show
        document.body.appendChild(modal);
        const bootstrapModal = new bootstrap.Modal(modal);
        bootstrapModal.show();
        
        // Remove from DOM when hidden
        modal.addEventListener('hidden.bs.modal', function() {
            document.body.removeChild(modal);
        });
    });
}

/**
 * Set up form submission with loading animation
 */
function setupFormSubmission() {
    const uploadForm = document.getElementById('upload-form');
    const batchUploadForm = document.getElementById('batch-upload-form');
    const loadingOverlay = document.getElementById('loading-overlay');
    
    if (uploadForm && loadingOverlay) {
        uploadForm.addEventListener('submit', function() {
            loadingOverlay.style.display = 'flex';
        });
    }
    
    if (batchUploadForm && loadingOverlay) {
        batchUploadForm.addEventListener('submit', function() {
            loadingOverlay.style.display = 'flex';
        });
    }
}

/**
 * Validate and highlight form fields
 * @param {HTMLFormElement} form - The form to validate
 * @returns {boolean} - Whether the form is valid
 */
function validateForm(form) {
    let isValid = true;
    
    // Reset previous validation
    const invalidFields = form.querySelectorAll('.is-invalid');
    invalidFields.forEach(field => field.classList.remove('is-invalid'));
    
    // Check required fields
    const requiredFields = form.querySelectorAll('[required]');
    requiredFields.forEach(field => {
        if (!field.value.trim()) {
            field.classList.add('is-invalid');
            isValid = false;
        }
    });
    
    // Check numeric fields
    const numericFields = form.querySelectorAll('[data-type="numeric"]');
    numericFields.forEach(field => {
        if (field.value && isNaN(parseFloat(field.value))) {
            field.classList.add('is-invalid');
            isValid = false;
        }
    });
    
    return isValid;
}

/**
 * Show a toast notification
 * @param {string} message - The message to display
 * @param {string} type - The type of toast (success, error, warning, info)
 */
function showToast(message, type = 'info') {
    // Create toast container if it doesn't exist
    let toastContainer = document.querySelector('.toast-container');
    if (!toastContainer) {
        toastContainer = document.createElement('div');
        toastContainer.className = 'toast-container position-fixed bottom-0 end-0 p-3';
        document.body.appendChild(toastContainer);
    }
    
    // Create toast element
    const toastEl = document.createElement('div');
    toastEl.className = `toast align-items-center text-white bg-${type}`;
    toastEl.setAttribute('role', 'alert');
    toastEl.setAttribute('aria-live', 'assertive');
    toastEl.setAttribute('aria-atomic', 'true');
    
    // Create toast content
    const toastBody = document.createElement('div');
    toastBody.className = 'd-flex';
    
    const messageDiv = document.createElement('div');
    messageDiv.className = 'toast-body';
    messageDiv.textContent = message;
    
    const closeButton = document.createElement('button');
    closeButton.type = 'button';
    closeButton.className = 'btn-close btn-close-white me-2 m-auto';
    closeButton.setAttribute('data-bs-dismiss', 'toast');
    closeButton.setAttribute('aria-label', 'Close');
    
    // Assemble the toast
    toastBody.appendChild(messageDiv);
    toastBody.appendChild(closeButton);
    toastEl.appendChild(toastBody);
    
    // Add to container
    toastContainer.appendChild(toastEl);
    
    // Initialize and show toast
    const toast = new bootstrap.Toast(toastEl);
    toast.show();
    
    // Remove from DOM when hidden
    toastEl.addEventListener('hidden.bs.toast', function() {
        toastContainer.removeChild(toastEl);
    });
}
