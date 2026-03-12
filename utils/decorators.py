"""
utils/decorators.py
-------------------
Utility decorators for NLPRec system.
Provides timing, caching, and monitoring decorators.
"""

import time
import logging
import functools
from typing import Callable, Any

log = logging.getLogger("NLPRec-Decorators")


def timing(func: Callable) -> Callable:
    """
    Decorator to measure and log execution time of a function.
    
    Args:
        func: Function to time
        
    Returns:
        Wrapped function that logs execution time
        
    Example:
        @timing
        def slow_function():
            time.sleep(1)
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        log.info(f"{func.__name__} took {elapsed:.4f}s")
        return result
    return wrapper


def retry(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """
    Decorator to retry a function on failure with exponential backoff.
    
    Args:
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff: Multiplier for delay after each attempt
        
    Returns:
        Decorator function
        
    Example:
        @retry(max_attempts=3, delay=1.0, backoff=2.0)
        def unstable_api_call():
            # code that might fail
            pass
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None
            
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts:
                        log.warning(
                            f"{func.__name__} failed (attempt {attempt}/{max_attempts}): {e}. "
                            f"Retrying in {current_delay:.1f}s..."
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        log.error(
                            f"{func.__name__} failed after {max_attempts} attempts: {e}"
                        )
            
            raise last_exception
        return wrapper
    return decorator


def log_calls(log_args: bool = False, log_result: bool = False):
    """
    Decorator to log function calls with optional argument and result logging.
    
    Args:
        log_args: Whether to log function arguments
        log_result: Whether to log function return value
        
    Returns:
        Decorator function
        
    Example:
        @log_calls(log_args=True, log_result=True)
        def calculate(x, y):
            return x + y
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            func_name = func.__name__
            
            if log_args:
                log.debug(f"Calling {func_name} with args={args}, kwargs={kwargs}")
            else:
                log.debug(f"Calling {func_name}")
            
            result = func(*args, **kwargs)
            
            if log_result:
                log.debug(f"{func_name} returned: {result}")
            
            return result
        return wrapper
    return decorator


def memoize(func: Callable) -> Callable:
    """
    Simple memoization decorator for caching function results.
    
    Args:
        func: Function to memoize
        
    Returns:
        Wrapped function with caching
        
    Note:
        Only works with hashable arguments.
        
    Example:
        @memoize
        def fibonacci(n):
            if n < 2:
                return n
            return fibonacci(n-1) + fibonacci(n-2)
    """
    cache = {}
    
    @functools.wraps(func)
    def wrapper(*args):
        if args in cache:
            return cache[args]
        result = func(*args)
        cache[args] = result
        return result
    
    return wrapper


def deprecated(message: str = "This function is deprecated"):
    """
    Decorator to mark a function as deprecated and log a warning when called.
    
    Args:
        message: Custom deprecation message
        
    Returns:
        Decorator function
        
    Example:
        @deprecated("Use new_function() instead")
        def old_function():
            pass
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            log.warning(f"{func.__name__} is deprecated: {message}")
            return func(*args, **kwargs)
        return wrapper
    return decorator
