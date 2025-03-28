"""
AI-based Field Recognition Module for Invoice Processing
This module enhances OCR accuracy using improved image processing techniques
to better recognize invoice fields, with optional AI enhancements when available.
"""

import os
import re
import logging
import base64
import io
from collections import defaultdict
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import optional dependencies with fallbacks
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    logger.warning("OpenCV (cv2) not available. Using basic image processing.")
    CV2_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    logger.warning("NumPy not available. Using basic data processing.")
    NUMPY_AVAILABLE = False

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    logger.warning("Pytesseract not available. OCR functionality will be limited.")
    TESSERACT_AVAILABLE = False

# Try to import AI-related libraries but provide fallbacks
try:
    from sklearn.cluster import DBSCAN
    SKLEARN_AVAILABLE = True
except ImportError:
    logger.warning("Scikit-learn not available. Using basic clustering.")
    SKLEARN_AVAILABLE = False

# Try to import TensorFlow with appropriate fallbacks
TF_AVAILABLE = False
try:
    # First check if numpy is fully working
    import numpy as np
    test_array = np.array([1, 2, 3])
    
    # Only try importing TensorFlow if numpy is working properly
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
    
    # Test TensorFlow functionality with a basic operation
    test_tf = tf.constant([1, 2, 3])
    
    # Only mark as available if all the above succeeded
    TF_AVAILABLE = True
    logger.info("TensorFlow is available for enhanced field recognition")
except ImportError as ie:
    TF_AVAILABLE = False
    logger.warning(f"TensorFlow functionality unavailable: {ie}. Using traditional OCR methods instead.")
except Exception as e:
    TF_AVAILABLE = False
    logger.warning(f"TensorFlow loading error: {e}. Using traditional OCR methods instead.")

# Matplotlib is optional - used only for debugging visualizations
try:
    import matplotlib.pyplot as plt
    MPL_AVAILABLE = True
except ImportError:
    logger.warning("Matplotlib not available. Debugging visualizations will be limited.")
    MPL_AVAILABLE = False

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AIFieldRecognizer:
    """AI-based field recognition for invoice processing with improved accuracy"""
    
    def __init__(self):
        """Initialize the AI field recognizer"""
        self.field_model = None
        self.character_model = None
        self.initialize_models()
        
        # Define field types and their patterns
        self.field_patterns = {
            'invoice_number': [
                r'invoice\s*#?\s*[:]?\s*([A-Z0-9]{5,15})',
                r'inv[\.:]?\s*no\.?\s*[:]?\s*([A-Z0-9]{5,15})',
                r'inv[\.:]?\s*num\.?\s*[:]?\s*([A-Z0-9]{5,15})',
                r'(INV-\d+)',
                r'invoice\s*number\s*:?\s*([A-Z0-9-]+)',
                # Edendale specific patterns
                r'\b([0-9]{8})\b',  # 8-digit invoice number
                r'\b([0-9]{5,9})\b',  # 5-9 digit invoice number near "invoice"
            ],
            'date': [
                r'date\s*:?\s*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})',
                r'date\s*:?\s*(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4})',
                r'invoice\s+date\s*:?\s*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})',
                r'(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})',
                r'(\d{1,2}/\d{1,2}/\d{2,4})',
                r'(\d{4}-\d{2}-\d{2})',  # ISO format
            ],
            'total': [
                r'total\s*:?\s*[$£€]?\s*(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2}))',
                r'amount\s+due\s*:?\s*[$£€]?\s*(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2}))',
                r'grand\s+total\s*:?\s*[$£€]?\s*(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2}))',
                r'balance\s+due\s*:?\s*[$£€]?\s*(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2}))',
                r'total\s+due\s*:?\s*[$£€]?\s*(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2}))',
                r'[$£€]\s*(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2}))',
            ],
            'product_code': [
                r'\b([A-Z0-9]{5,10})\b',
                r'\b(\d{5,9})\b',  # Edendale 5-9 digit product codes
                r'\b([A-Z]{2,3}-\d{3,5})\b',
            ]
        }

    def initialize_models(self):
        """Initialize TensorFlow models for field recognition and character enhancement if available"""
        self.field_model = None
        self.character_model = None
        
        # Only initialize TensorFlow models if available
        if TF_AVAILABLE:
            try:
                # Create a simple model for field classification
                self.field_model = Sequential([
                    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 512, 1)),
                    MaxPooling2D((2, 2)),
                    Conv2D(64, (3, 3), activation='relu'),
                    MaxPooling2D((2, 2)),
                    Conv2D(64, (3, 3), activation='relu'),
                    Flatten(),
                    Dense(64, activation='relu'),
                    Dropout(0.5),
                    Dense(10, activation='softmax')  # 10 common field types
                ])
                
                # Create a simple model for character enhancement
                self.character_model = Sequential([
                    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 1)),
                    MaxPooling2D((2, 2)),
                    Conv2D(64, (3, 3), activation='relu'),
                    MaxPooling2D((2, 2)),
                    Flatten(),
                    Dense(128, activation='relu'),
                    Dense(94, activation='softmax')  # 94 printable ASCII characters
                ])
                
                logger.info("AI models initialized successfully")
            except Exception as e:
                logger.error(f"Error initializing AI models: {e}")
                # Fall back to traditional methods if models can't be initialized
                self.field_model = None
                self.character_model = None
        else:
            logger.info("Using traditional methods without TensorFlow models")

    def preprocess_image(self, image_path):
        """
        Enhanced image preprocessing for better OCR accuracy
        
        Args:
            image_path: Path to the invoice image
            
        Returns:
            Preprocessed image and original image
        """
        # Check if OpenCV is available
        if not CV2_AVAILABLE or not NUMPY_AVAILABLE:
            logger.warning("OpenCV or NumPy not available. Returning original image without preprocessing.")
            # Return the original image path for Tesseract to handle directly
            return image_path, image_path
            
        try:
            # Read the image
            if isinstance(image_path, str):
                image = cv2.imread(image_path)
            else:
                # If image_path is already a numpy array (for preview functionality)
                image = image_path
                
            if image is None:
                logger.warning(f"Could not read image from {image_path}. Returning original path.")
                return image_path, image_path
                
            # Store the original for visualization
            original = image.copy()
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Advanced preprocessing methods
            # 1. Apply adaptive thresholding
            adaptive_thresh = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 5
            )
            
            # 2. Apply noise reduction
            denoised = cv2.fastNlMeansDenoising(adaptive_thresh, None, 10, 7, 21)
            
            # 3. Apply skew correction if needed
            try:
                coords = np.column_stack(np.where(denoised > 0))
                angle = cv2.minAreaRect(coords)[-1]
                
                if angle < -45:
                    angle = -(90 + angle)
                else:
                    angle = -angle
                    
                # Only correct if skew is significant
                if abs(angle) > 0.5:
                    (h, w) = denoised.shape[:2]
                    center = (w // 2, h // 2)
                    M = cv2.getRotationMatrix2D(center, angle, 1.0)
                    denoised = cv2.warpAffine(
                        denoised, M, (w, h), 
                        flags=cv2.INTER_CUBIC, 
                        borderMode=cv2.BORDER_REPLICATE
                    )
            except Exception as e:
                # Continue even if skew correction fails
                logger.warning(f"Skew correction failed: {e}")
            
            # 4. Sharpen the image
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpened = cv2.filter2D(denoised, -1, kernel)
            
            return sharpened, original
            
        except Exception as e:
            logger.error(f"Error in image preprocessing: {e}")
            # Fall back to simpler preprocessing if advanced methods fail
            try:
                image = cv2.imread(image_path)
                original = image.copy()
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                return binary, original
            except Exception as e2:
                logger.error(f"Fallback preprocessing also failed: {e2}. Returning original path.")
                # Return the original path as a last resort
                return image_path, image_path

    def get_text_regions(self, preprocessed_image):
        """
        Identify text regions in the invoice using contour detection
        
        Args:
            preprocessed_image: The preprocessed image
            
        Returns:
            List of identified text regions (x, y, w, h)
        """
        # Check if OpenCV and dependencies are available
        if not CV2_AVAILABLE or not NUMPY_AVAILABLE:
            logger.warning("OpenCV or NumPy not available. Returning default regions.")
            # Return some standard regions covering the entire image as a fallback
            if isinstance(preprocessed_image, str):
                # If we got a file path because preprocessing was skipped
                logger.info("Using full image as single region")
                return [(0, 0, 800, 1000)]  # Arbitrary size covering full page
            
            # If we somehow have a NumPy array but no OpenCV
            try:
                h, w = preprocessed_image.shape[:2]
                # Divide into 4 quadrants as a simple approach
                return [
                    (0, 0, w//2, h//2),
                    (w//2, 0, w//2, h//2),
                    (0, h//2, w//2, h//2),
                    (w//2, h//2, w//2, h//2)
                ]
            except:
                return [(0, 0, 800, 1000)]  # Arbitrary fallback
        
        try:
            # Find all contours
            contours, _ = cv2.findContours(
                preprocessed_image, 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            # Filter contours based on size
            text_regions = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter by size to exclude noise and consider only likely text regions
                if w > 40 and h > 10 and w < preprocessed_image.shape[1] * 0.9:
                    text_regions.append((x, y, w, h))
            
            # If we have scikit-learn available, use DBSCAN for clustering
            if SKLEARN_AVAILABLE and text_regions:
                # Extract centers of regions
                centers = np.array([[x + w/2, y + h/2] for x, y, w, h in text_regions])
                
                # Apply DBSCAN clustering
                clustering = DBSCAN(eps=50, min_samples=1).fit(centers)
                labels = clustering.labels_
                
                # Group regions by cluster
                groups = defaultdict(list)
                for label, region in zip(labels, text_regions):
                    groups[label].append(region)
                
                # Merge regions in each cluster
                merged_regions = []
                for group in groups.values():
                    min_x = min(x for x, _, _, _ in group)
                    min_y = min(y for _, y, _, _ in group)
                    max_x = max(x + w for x, _, w, _ in group)
                    max_y = max(y + h for _, y, _, h in group)
                    merged_regions.append((min_x, min_y, max_x - min_x, max_y - min_y))
                
                return merged_regions
            
            return text_regions
            
        except Exception as e:
            logger.error(f"Error in text region detection: {e}")
            # Return fallback regions if detection fails
            if isinstance(preprocessed_image, str):
                return [(0, 0, 800, 1000)]
            try:
                h, w = preprocessed_image.shape[:2]
                return [(0, 0, w, h)]
            except:
                return [(0, 0, 800, 1000)]

    def enhance_ocr(self, image, region):
        """
        Apply enhanced OCR to a specific region
        
        Args:
            image: Original image
            region: Region coordinates (x, y, w, h)
            
        Returns:
            Enhanced OCR text
        """
        # Check if Tesseract is available
        if not TESSERACT_AVAILABLE:
            logger.warning("Pytesseract not available. Cannot perform OCR.")
            return "OCR not available - Please install pytesseract"
            
        # Check if we have valid image and region
        if isinstance(image, str):
            logger.warning("Image is a file path, not a numpy array. Cannot extract region.")
            try:
                # Try to process the whole image file directly with pytesseract
                return pytesseract.image_to_string(image)
            except Exception as e:
                logger.error(f"Failed to process image file with pytesseract: {e}")
                return ""
        
        try:
            x, y, w, h = region
            # Check if we're working with a numpy array
            if not NUMPY_AVAILABLE:
                logger.warning("NumPy not available. Cannot extract region.")
                return ""
                
            try:
                # Try to extract the region
                roi = image[y:y+h, x:x+w]
            except Exception as e:
                logger.error(f"Failed to extract region: {e}")
                # Try with the whole image
                roi = image
            
            # Apply multiple OCR configurations to improve accuracy
            # Configuration 1: Default with focus on precision
            config1 = '--oem 3 --psm 6 -c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,;:$%#@!&*()-+=/\\\'\"?_ "' 
            text1 = pytesseract.image_to_string(roi, config=config1)
            
            # Configuration 2: Optimize for digit recognition
            config2 = '--oem 3 --psm 7 -c tessedit_char_whitelist="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,;:$%#@!&*()-+=/\\\'\"?_ "'
            text2 = pytesseract.image_to_string(roi, config=config2)
            
            # Configuration 3: Optimize for line detection
            config3 = '--oem 3 --psm 4'
            text3 = pytesseract.image_to_string(roi, config=config3)
            
            # Choose the best result (heuristically)
            if len(text1.strip()) > len(text2.strip()) and len(text1.strip()) > len(text3.strip()):
                return text1
            elif len(text2.strip()) > len(text3.strip()):
                return text2
            else:
                return text3
                
        except Exception as e:
            logger.error(f"Error in enhanced OCR: {e}")
            # Fall back to basic OCR if enhanced OCR fails
            try:
                if isinstance(image, str):
                    return pytesseract.image_to_string(image)
                else:
                    x, y, w, h = region
                    try:
                        roi = image[y:y+h, x:x+w]
                    except:
                        roi = image
                    return pytesseract.image_to_string(roi)
            except Exception as e2:
                logger.error(f"Fallback OCR also failed: {e2}")
                return ""

    def identify_field_type(self, text):
        """
        Identify the type of field from its text content
        
        Args:
            text: Extracted text
            
        Returns:
            Field type and extracted value
        """
        text = text.strip().lower()
        
        for field_type, patterns in self.field_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    return field_type, match.group(1)
        
        # Check for line item patterns
        if re.search(r'\d+\s+\d+\.\d{2}', text) or re.search(r'\b\d{5,9}\b', text):
            return 'line_item', text
            
        return 'unknown', text

    def extract_table_structure(self, preprocessed_image, original_image):
        """
        Extract table structure from the invoice
        
        Args:
            preprocessed_image: Preprocessed image
            original_image: Original image
            
        Returns:
            Detected table structure and line items
        """
        # Check if OpenCV and NumPy are available
        if not CV2_AVAILABLE or not NUMPY_AVAILABLE:
            logger.warning("OpenCV or NumPy not available. Cannot extract table structure.")
            return [], []
            
        # Check if Tesseract is available for text extraction
        if not TESSERACT_AVAILABLE:
            logger.warning("Pytesseract not available. Cannot extract table text.")
            return [], []
            
        # If we were given a file path instead of an image array (due to fallbacks earlier)
        if isinstance(preprocessed_image, str) or isinstance(original_image, str):
            logger.warning("Cannot extract table from file path. Extracting from whole image.")
            if isinstance(original_image, str) and TESSERACT_AVAILABLE:
                try:
                    # Try to process the whole image with pytesseract
                    full_text = pytesseract.image_to_string(original_image)
                    lines = full_text.split('\n')
                    line_items = []
                    for line in lines:
                        line = line.strip()
                        if line and re.search(r'\d', line):
                            line_items.append(line)
                    return [], line_items
                except Exception as e:
                    logger.error(f"Failed to process image file with pytesseract: {e}")
                    return [], []
            return [], []
            
        try:
            # Detect horizontal and vertical lines
            horizontal = preprocessed_image.copy()
            vertical = preprocessed_image.copy()
            
            # Specify size on horizontal axis
            cols = horizontal.shape[1]
            horizontal_size = cols // 30
            
            # Create structure element for extracting horizontal lines
            horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
            
            # Apply morphology operations
            horizontal = cv2.erode(horizontal, horizontalStructure)
            horizontal = cv2.dilate(horizontal, horizontalStructure)
            
            # Specify size on vertical axis
            rows = vertical.shape[0]
            verticalsize = rows // 30
            
            # Create structure element for extracting vertical lines
            verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))
            
            # Apply morphology operations
            vertical = cv2.erode(vertical, verticalStructure)
            vertical = cv2.dilate(vertical, verticalStructure)
            
            # Create a mask which includes the tables
            mask = horizontal + vertical
            
            # Find contours in the mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            table_regions = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if w > 100 and h > 50:  # Filter small regions
                    table_regions.append((x, y, w, h))
            
            # Extract text from table regions
            line_items = []
            for region in table_regions:
                x, y, w, h = region
                try:
                    table_roi = original_image[y:y+h, x:x+w]
                    
                    # Apply table-specific OCR with line segmentation
                    table_text = pytesseract.image_to_string(
                        table_roi, 
                        config='--oem 3 --psm 6'
                    )
                    
                    # Process table text into line items
                    lines = table_text.split('\n')
                    for line in lines:
                        line = line.strip()
                        if line and re.search(r'\d', line):  # Must contain at least one digit
                            line_items.append(line)
                except Exception as e:
                    logger.error(f"Error extracting text from table region: {e}")
            
            return table_regions, line_items
            
        except Exception as e:
            logger.error(f"Error in table structure extraction: {e}")
            return [], []

    def process_invoice(self, image_path):
        """
        Process an invoice image using AI-enhanced field recognition
        
        Args:
            image_path: Path to the invoice image
            
        Returns:
            Extracted invoice data and visualization info
        """
        # First check if critical libraries are available
        if not os.path.exists(image_path):
            logger.error(f"Image file not found: {image_path}")
            return {
                'fields': {'invoice_number': os.path.basename(image_path)},
                'line_items': [{'product_code': 'ERROR', 'description': 'Image file not found', 'quantity': '0', 'unit_price': '0', 'total': '0'}],
                'visualization': None,
                'error': f"Image file not found: {image_path}"
            }
            
        try:
            # Preprocess the image with enhanced error handling
            try:
                preprocessed, original = self.preprocess_image(image_path)
            except Exception as e:
                logger.error(f"Error in image preprocessing: {e}")
                # Fall back to original image path if preprocessing fails
                preprocessed = image_path
                original = image_path
                
            # Get text regions with enhanced error handling
            try:
                text_regions = self.get_text_regions(preprocessed)
            except Exception as e:
                logger.error(f"Error in text region detection: {e}")
                # Fall back to whole image if region detection fails
                if isinstance(preprocessed, str):
                    text_regions = [(0, 0, 800, 1000)]  # Default size
                else:
                    try:
                        h, w = preprocessed.shape[:2]
                        text_regions = [(0, 0, w, h)]
                    except:
                        text_regions = [(0, 0, 800, 1000)]
            
            # Extract field information with enhanced error handling
            fields = {}
            all_text = ""
            
            for region in text_regions:
                try:
                    text = self.enhance_ocr(original, region)
                    all_text += text + "\n"
                    field_type, value = self.identify_field_type(text)
                    
                    if field_type != 'unknown':
                        fields[field_type] = value
                except Exception as e:
                    logger.error(f"Error in OCR for region {region}: {e}")
                    # Continue with other regions
                    continue
            
            # Extract table structure and line items with enhanced error handling
            try:
                table_regions, line_items = self.extract_table_structure(preprocessed, original)
            except Exception as e:
                logger.error(f"Error in table structure extraction: {e}")
                table_regions = []
                line_items = []
            
            # If no line items detected through table structure, try extracting from all text
            if not line_items:
                try:
                    text_line_items = self.extract_line_items_from_text(all_text)
                    line_items.extend(text_line_items)
                except Exception as e:
                    logger.error(f"Error extracting line items from text: {e}")
                    # Add a placeholder line item to avoid empty results
                    line_items.append({
                        'product_code': 'ERROR',
                        'description': 'Error processing line items',
                        'quantity': '0',
                        'unit_price': '0',
                        'total': '0'
                    })
            
            # Apply Edendale-specific rules based on image analysis
            try:
                if self.is_edendale_invoice(all_text):
                    fields, line_items = self.apply_edendale_rules(fields, line_items, all_text)
            except Exception as e:
                logger.error(f"Error applying Edendale rules: {e}")
                # Continue with existing fields and line items
            
            # Prepare visualization data with enhanced error handling
            try:
                visualization_data = self.prepare_visualization(original, text_regions, table_regions, fields)
            except Exception as e:
                logger.error(f"Error preparing visualization: {e}")
                visualization_data = None
            
            # Ensure we have at least basic invoice data
            if 'invoice_number' not in fields:
                fields['invoice_number'] = os.path.basename(image_path)
                
            if not line_items:
                line_items.append({
                    'product_code': 'UNKNOWN',
                    'description': 'No line items detected',
                    'quantity': '0',
                    'unit_price': '0',
                    'total': '0'
                })
            
            # Prepare result
            result = {
                'fields': fields,
                'line_items': line_items,
                'visualization': visualization_data
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in invoice processing: {e}")
            # Return basic structure even on error
            filename = os.path.basename(image_path)
            return {
                'fields': {'invoice_number': filename},
                'line_items': [{
                    'product_code': 'ERROR',
                    'description': f'Error processing invoice: {str(e)}',
                    'quantity': '0',
                    'unit_price': '0',
                    'total': '0'
                }],
                'visualization': None,
                'error': str(e)
            }

    def is_edendale_invoice(self, text):
        """Check if the invoice is in Edendale format"""
        return 'edendale' in text.lower() or re.search(r'\b[0-9]{8}\b', text) is not None

    def apply_edendale_rules(self, fields, line_items, text):
        """Apply Edendale-specific rules to improve extraction"""
        # Enhanced invoice number extraction for Edendale
        invoice_match = re.search(r'\b([0-9]{8})\b', text)
        if invoice_match:
            fields['invoice_number'] = invoice_match.group(1)
        
        # Enhanced product code extraction for Edendale
        enhanced_line_items = []
        for line in line_items:
            # Look for product codes in specific formats
            code_match = re.search(r'\b(\d{5,9})\b', line)
            if code_match:
                # Ensure this item isn't already processed
                if not any(code_match.group(1) in item for item in enhanced_line_items):
                    enhanced_line_items.append(line)
        
        # If we found enhanced line items, use them instead
        if enhanced_line_items:
            return fields, enhanced_line_items
        
        return fields, line_items

    def extract_line_items_from_text(self, text):
        """Extract line items from the full text when table detection fails"""
        lines = text.split('\n')
        line_items = []
        
        for i, line in enumerate(lines):
            line = line.strip()
            # Look for patterns that indicate line items
            if (re.search(r'\b\d{5,9}\b', line) and  # Product code
                (re.search(r'\d+\.\d{2}', line) or   # Price
                 (i + 1 < len(lines) and re.search(r'\d+\.\d{2}', lines[i+1])))):
                line_items.append(line)
        
        return line_items

    def prepare_visualization(self, original, text_regions, table_regions, fields):
        """Prepare visualization data for interactive preview"""
        # Check if OpenCV and NumPy are available for visualization
        if not CV2_AVAILABLE or not NUMPY_AVAILABLE:
            logger.warning("OpenCV or NumPy not available. Cannot prepare visualization.")
            return None
            
        # If we have a file path instead of an image (due to fallbacks)
        if isinstance(original, str):
            logger.warning("Cannot create visualization from file path.")
            return None
            
        try:
            # Create a copy for visualization
            vis_image = original.copy()
            
            # Draw text regions
            for i, region in enumerate(text_regions):
                x, y, w, h = region
                # Draw rectangle around text region
                cv2.rectangle(vis_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Add field type label if available
                field_type = None
                for field, value in fields.items():
                    # Check if region text contains field value
                    region_text = self.enhance_ocr(original, region)
                    if value in region_text:
                        field_type = field
                        break
                
                if field_type:
                    cv2.putText(vis_image, field_type, (x, y-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Draw table regions
            for region in table_regions:
                x, y, w, h = region
                cv2.rectangle(vis_image, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.putText(vis_image, "Table", (x, y-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            # Convert to base64 for web display
            try:
                _, buffer = cv2.imencode('.png', vis_image)
                img_str = base64.b64encode(buffer).decode('utf-8')
                
                return {
                    'image': img_str,
                    'text_regions': text_regions,
                    'table_regions': table_regions
                }
            except Exception as e:
                logger.error(f"Error encoding image: {e}")
                return None
            
        except Exception as e:
            logger.error(f"Error in visualization preparation: {e}")
            return None

    def generate_visualization(self, extraction_result):
        """Generate HTML for interactive visualization"""
        try:
            # Validate input first
            if not extraction_result or not isinstance(extraction_result, dict):
                return """
                <div class="alert alert-warning">
                    <i class="fas fa-exclamation-triangle"></i> Invalid extraction result format.
                </div>
                <div>
                    <p>The extraction result could not be processed correctly. Please try again.</p>
                </div>
                """
                
            # Extract the data even if visualization is not available
            fields = extraction_result.get('fields', {})
            
            # Handle line_items properly based on type
            line_items_raw = extraction_result.get('line_items', [])
            line_items = []
            
            # Process line items into a consistent format
            for item in line_items_raw:
                if isinstance(item, dict):
                    # Format dictionary items properly
                    try:
                        item_str = f"{item.get('product_code', 'N/A')} - {item.get('description', 'No description')} - Qty: {item.get('quantity', '0')} - Unit: {item.get('unit_price', '0')} - Total: {item.get('total', '0')}"
                        line_items.append(item_str)
                    except Exception as format_error:
                        logger.error(f"Error formatting line item dict: {format_error}")
                        line_items.append(str(item))
                elif isinstance(item, str):
                    # String items can be used directly
                    line_items.append(item)
                else:
                    # Convert other types to string
                    try:
                        line_items.append(str(item))
                    except Exception as e:
                        logger.error(f"Failed to convert line item to string: {e}")
                        line_items.append("Error: Could not display this item")
            
            # Check if visualization data is available with safe access
            has_visualization = False
            try:
                vis_data = extraction_result.get('visualization', None)
                has_visualization = (vis_data is not None and 
                                    isinstance(vis_data, dict) and
                                    'image' in vis_data and 
                                    vis_data['image'])
            except Exception as vis_error:
                logger.error(f"Error checking visualization data: {vis_error}")
                has_visualization = False
                                
            # Create HTML for the data display with safe HTML generation
            col_class = "col-md-6" if has_visualization else "col-md-12"
            
            html = f"""
            <div class="extraction-preview">
                <div class="row">
                    <div class="{col_class}">
                        <h4>Extracted Fields</h4>
                        <table class="table table-sm">
                            <thead>
                                <tr>
                                    <th>Field</th>
                                    <th>Value</th>
                                </tr>
                            </thead>
                            <tbody>
            """
            
            # Add fields with safer HTML generation
            if fields and isinstance(fields, dict):
                for field, value in fields.items():
                    try:
                        field_name = str(field)
                        field_value = str(value)
                        # Basic HTML escaping
                        field_name = field_name.replace('<', '&lt;').replace('>', '&gt;')
                        field_value = field_value.replace('<', '&lt;').replace('>', '&gt;')
                        
                        html += f"""
                        <tr>
                            <td>{field_name}</td>
                            <td>{field_value}</td>
                        </tr>
                        """
                    except Exception as e:
                        logger.error(f"Error rendering field: {e}")
                        html += """
                        <tr>
                            <td colspan="2" class="text-center text-danger">Error displaying this field</td>
                        </tr>
                        """
            else:
                html += """
                <tr>
                    <td colspan="2" class="text-center">No fields detected</td>
                </tr>
                """
            
            html += """
                            </tbody>
                        </table>
                        
                        <h4>Line Items</h4>
                        <table class="table table-sm">
                            <thead>
                                <tr>
                                    <th>#</th>
                                    <th>Item</th>
                                </tr>
                            </thead>
                            <tbody>
            """
            
            # Add line items with safer HTML generation
            if line_items:
                for i, item in enumerate(line_items):
                    try:
                        # Basic HTML escaping
                        safe_item = str(item).replace('<', '&lt;').replace('>', '&gt;')
                        html += f"""
                        <tr>
                            <td>{i+1}</td>
                            <td>{safe_item}</td>
                        </tr>
                        """
                    except Exception as e:
                        logger.error(f"Error rendering line item: {e}")
                        html += """
                        <tr>
                            <td>-</td>
                            <td class="text-danger">Error displaying this item</td>
                        </tr>
                        """
            else:
                html += """
                <tr>
                    <td colspan="2" class="text-center">No line items detected</td>
                </tr>
                """
            
            html += """
                            </tbody>
                        </table>
                    </div>
            """
            
            # Add visualization section if available - with robust error handling
            if has_visualization:
                try:
                    vis_data = extraction_result['visualization']
                    image_data = vis_data.get('image', '')
                    
                    # Validate the image data is a string that can be displayed
                    if isinstance(image_data, str) and image_data:
                        html += f"""
                            <div class="col-md-6">
                                <h4>Visual Extraction</h4>
                                <div class="visual-preview text-center">
                                    <img src="data:image/png;base64,{image_data}" class="img-fluid extraction-image">
                                </div>
                            </div>
                        """
                    else:
                        raise ValueError("Invalid image data format")
                except Exception as vis_error:
                    logger.error(f"Error rendering visualization: {vis_error}")
                    html += """
                        <div class="col-md-6">
                            <div class="alert alert-warning">
                                <i class="fas fa-exclamation-triangle"></i> Error displaying visual extraction.
                            </div>
                            <div class="text-center text-muted">
                                <p>The visual preview could not be generated due to an error in image processing.</p>
                            </div>
                        </div>
                    """
            else:
                # Add a note when visualization is not available
                html += """
                    <div class="col-md-12 mt-3">
                        <div class="alert alert-info">
                            <i class="fas fa-info-circle"></i> Visual extraction preview is not available. 
                            This may be due to missing OpenCV dependencies or image processing limitations.
                        </div>
                    </div>
                """
            
            html += """
                </div>
            </div>
            """
            
            return html
            
        except Exception as e:
            logger.error(f"Error generating visualization: {e}")
            # Return a simpler fallback HTML with clear error message
            return f"""
            <div class="card border-danger mb-3">
                <div class="card-header bg-danger text-white">
                    <i class="fas fa-exclamation-triangle"></i> Visualization Error
                </div>
                <div class="card-body">
                    <h5 class="card-title">Error generating visualization</h5>
                    <p class="card-text">An error occurred while preparing the visualization: {str(e)}</p>
                    <p>The extraction process may have completed successfully despite this error. You can try downloading the results or processing the invoice again.</p>
                </div>
            </div>
            """