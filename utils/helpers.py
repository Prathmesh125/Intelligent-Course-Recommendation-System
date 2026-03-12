"""
utils/helpers.py
----------------
Common utility functions used across the NLPRec system.
Provides file operations, validation, formatting, and other helpers.
"""

import os
import json
import logging
from typing import Any, Dict, Optional
from datetime import datetime

log = logging.getLogger("NLPRec-Utils")


def safe_json_load(filepath: str, default: Any = None) -> Any:
    """
    Safely load JSON from a file with error handling.
    
    Args:
        filepath: Path to the JSON file
        default: Default value to return if file doesn't exist or is invalid
        
    Returns:
        Loaded JSON data or default value
    """
    if not os.path.exists(filepath):
        log.debug(f"JSON file not found: {filepath}")
        return default if default is not None else {}
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        log.error(f"Invalid JSON in {filepath}: {e}")
        return default if default is not None else {}
    except Exception as e:
        log.error(f"Error loading JSON from {filepath}: {e}")
        return default if default is not None else {}


def safe_json_save(data: Any, filepath: str) -> bool:
    """
    Safely save data to a JSON file with error handling.
    
    Args:
        data: Data to serialize to JSON
        filepath: Path to save the file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        log.error(f"Error saving JSON to {filepath}: {e}")
        return False


def ensure_dir(directory: str) -> None:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        directory: Path to the directory
    """
    os.makedirs(directory, exist_ok=True)


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to a maximum length with a suffix.
    
    Args:
        text: Text to truncate
        max_length: Maximum length before truncation
        suffix: Suffix to add when truncating
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename by removing or replacing invalid characters.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename safe for use on filesystems
    """
    # Replace invalid characters with underscore
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    return filename.strip()


def format_timestamp(timestamp: Optional[float] = None) -> str:
    """
    Format a timestamp to ISO format string.
    
    Args:
        timestamp: Unix timestamp (uses current time if None)
        
    Returns:
        ISO-formatted timestamp string
    """
    if timestamp is None:
        return datetime.now().isoformat()
    return datetime.fromtimestamp(timestamp).isoformat()


def validate_rating(rating: float) -> bool:
    """
    Validate that a rating is within acceptable bounds.
    
    Args:
        rating: Rating value to validate
        
    Returns:
        True if rating is valid (0.0 to 5.0)
    """
    return 0.0 <= rating <= 5.0


def clamp(value: float, min_val: float, max_val: float) -> float:
    """
    Clamp a value between min and max bounds.
    
    Args:
        value: Value to clamp
        min_val: Minimum bound
        max_val: Maximum bound
        
    Returns:
        Clamped value
    """
    return max(min_val, min(value, max_val))


def merge_dicts(*dicts: Dict) -> Dict:
    """
    Merge multiple dictionaries, with later dicts taking precedence.
    
    Args:
        *dicts: Variable number of dictionaries to merge
        
    Returns:
        Merged dictionary
    """
    result = {}
    for d in dicts:
        result.update(d)
    return result
