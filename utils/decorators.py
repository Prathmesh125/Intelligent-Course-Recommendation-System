"""
utils/decorators.py
-------------------
Utility decorators for NLPRec system.
Provides timing, caching, and monitoring decorators.
"""

import time
import logging
import functools
from collections import defaultdict
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

def rate_limit(max_calls: int, time_window: float):
    """
    Decorator to rate limit function calls.
    
    Args:
        max_calls: Maximum number of calls allowed in time window
        time_window: Time window in seconds
        
    Returns:
        Decorator function
        
    Example:
        @rate_limit(max_calls=10, time_window=60)
        def api_call():
            # Limited to 10 calls per minute
            pass
    """
    call_times = defaultdict(list)
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            now = time.time()
            func_key = func.__name__
            
            # Remove old calls outside the time window
            call_times[func_key] = [
                t for t in call_times[func_key] 
                if now - t < time_window
            ]
            
            # Check if limit exceeded
            if len(call_times[func_key]) >= max_calls:
                oldest_call = call_times[func_key][0]
                wait_time = time_window - (now - oldest_call)
                log.warning(
                    f"Rate limit exceeded for {func_key}. "
                    f"Wait {wait_time:.1f}s before calling again."
                )
                raise RuntimeError(
                    f"Rate limit exceeded: {max_calls} calls per {time_window}s. "
                    f"Retry after {wait_time:.1f}s"
                )
            
            # Record this call
            call_times[func_key].append(now)
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def validate_input(**type_checks):
    """
    Decorator to validate function input types.
    
    Args:
        **type_checks: Mapping of parameter names to expected types
        
    Returns:
        Decorator function
        
    Example:
        @validate_input(name=str, age=int, score=float)
        def process_user(name, age, score):
            pass
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get function signature
            import inspect
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # Validate types
            for param_name, expected_type in type_checks.items():
                if param_name in bound_args.arguments:
                    actual_value = bound_args.arguments[param_name]
                    if not isinstance(actual_value, expected_type):
                        raise TypeError(
                            f"{func.__name__}() argument '{param_name}' must be "
                            f"{expected_type.__name__}, got {type(actual_value).__name__}"
                        )
            
            return func(*args, **kwargs)
        return wrapper
    return decorator