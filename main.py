import logging
import sys

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

try:
    from app import app
    logger.info("Successfully imported Flask app")
except ImportError as e:
    logger.error(f"Error importing app: {e}")
    sys.exit(1)
except Exception as e:
    logger.error(f"Unexpected error importing app: {e}")
    sys.exit(1)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
