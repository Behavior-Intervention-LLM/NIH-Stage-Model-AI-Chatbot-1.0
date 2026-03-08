"""
configuration
"""
import logging
import sys
from app.config import settings


def setup_logging():
    """Settings"""
    level = logging.DEBUG if settings.DEBUG else logging.INFO
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)


logger = setup_logging()
