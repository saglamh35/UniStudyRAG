"""
UniStudyRAG - Utility Functions
Merkezi logger yapılandırması ve yardımcı fonksiyonlar
"""

import logging
import sys
from typing import Optional


def setup_logger(
    name: str = "UniStudyRAG",
    level: int = logging.INFO,
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Merkezi logger yapılandırması.
    
    Args:
        name: Logger adı
        level: Log seviyesi (default: INFO)
        log_file: Log dosyası yolu (None ise sadece console'a yazar)
        
    Returns:
        logging.Logger: Yapılandırılmış logger nesnesi
    """
    logger = logging.getLogger(name)
    
    # Logger zaten yapılandırılmışsa, mevcut handler'ları temizle
    if logger.handlers:
        logger.handlers.clear()
    
    logger.setLevel(level)
    
    # Formatter oluştur
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (eğer belirtilmişse)
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Logger'ın parent'a propagate etmesini engelle
    logger.propagate = False
    
    return logger


# Global logger instance
logger = setup_logger()

