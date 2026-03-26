"""
utils/helpers.py
----------------
Common utility functions used across the NLPRec system.
Provides file operations, validation, formatting, and other helpers.
"""

import os
import json
import logging
import time
from typing import Any, Dict, Optional, Callable, List
from datetime import datetime
from functools import wraps

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


def validate_difficulty(difficulty: str, valid_options: list = None) -> str:
    """Validate a difficulty level; returns the value or 'All' if invalid."""
    if valid_options is None:
        valid_options = ["All", "Beginner", "Intermediate", "Advanced"]
    if difficulty in valid_options:
        return difficulty
    log.warning("Invalid difficulty %r, defaulting to 'All'", difficulty)
    return "All"


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

def retry_on_failure(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """
    Decorator to retry a function on failure with exponential backoff.
    
    Args:
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff: Multiplier for delay after each failure
        exceptions: Tuple of exception types to catch and retry
        
    Returns:
        Decorated function with retry logic
        
    Example:
        @retry_on_failure(max_attempts=3, delay=1.0)
        def fetch_data():
            # May fail transiently
            return requests.get(url)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None
            
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_attempts:
                        log.error(
                            f"Function {func.__name__} failed after {max_attempts} attempts: {e}"
                        )
                        raise
                    
                    log.warning(
                        f"Attempt {attempt}/{max_attempts} failed for {func.__name__}: {e}. "
                        f"Retrying in {current_delay:.1f}s..."
                    )
                    time.sleep(current_delay)
                    current_delay *= backoff
            
            # Should not reach here, but raise last exception if it does
            if last_exception:
                raise last_exception
        
        return wrapper
    return decorator


def retry_operation(
    operation: Callable,
    max_attempts: int = 3,
    delay: float = 1.0,
    on_error: Optional[Callable[[Exception], None]] = None
) -> Any:
    """
    Retry an operation with exponential backoff (functional approach).
    
    Args:
        operation: Function to execute
        max_attempts: Maximum number of attempts
        delay: Initial delay between retries in seconds
        on_error: Optional callback for handling errors
        
    Returns:
        Result of the operation
        
    Raises:
        Last exception if all attempts fail
    """
    current_delay = delay
    
    for attempt in range(1, max_attempts + 1):
        try:
            return operation()
        except Exception as e:
            if on_error:
                on_error(e)
            
            if attempt == max_attempts:
                log.error(f"Operation failed after {max_attempts} attempts: {e}")
                raise
            
            log.warning(f"Attempt {attempt}/{max_attempts} failed: {e}. Retrying in {current_delay:.1f}s...")
            time.sleep(current_delay)
            current_delay *= 2


# ── Pagination helper ─────────────────────────────────────────────────────────
def paginate_results(items: List[Any], page: int, page_size: int = 10) -> Dict[str, Any]:
    """Slice a list into a page and return metadata alongside the page items."""
    total = len(items)
    page = max(1, page)
    total_pages = max(1, (total + page_size - 1) // page_size)
    page = min(page, total_pages)
    start = (page - 1) * page_size
    return {
        "items": items[start: start + page_size],
        "page": page,
        "page_size": page_size,
        "total": total,
        "total_pages": total_pages,
        "has_next": page < total_pages,
        "has_prev": page > 1,
    }


# ── Human-readable duration formatter ────────────────────────────────────────
def format_duration(seconds: float) -> str:
    """Convert a duration in seconds to a compact human-readable string (e.g. '2h 4m 30s')."""
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    parts = []
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    if secs or not parts:
        parts.append(f"{secs}s")
    return " ".join(parts)