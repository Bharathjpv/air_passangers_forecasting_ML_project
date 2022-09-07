import logging
import os


LOG_DIR = 'logs'
LOG_FILE_NAME ='logs.log'
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(filename=os.path.join(LOG_DIR, LOG_FILE_NAME),
                    format='[ %(asctime)s ] %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

logger = logging.getLogger()