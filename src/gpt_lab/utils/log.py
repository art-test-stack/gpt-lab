import logging

def setup_logger():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logging.basicConfig(level=logging.INFO, handlers=[handler])
    
setup_logger()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)