"""
Enhanced Error Handling Utilities

Provides centralized error handling, categorization, and recovery strategies
for the agentic RAG system.
"""

import logging
import traceback
import asyncio
from typing import Dict, Any, Optional, Callable, TypeVar, Union
from datetime import datetime
from enum import Enum
from functools import wraps

logger = logging.getLogger(__name__)

T = TypeVar('T')


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for better handling"""
    USER_INPUT = "user_input"
    SYSTEM_CONFIG = "system_config"
    EXTERNAL_API = "external_api"
    DATABASE = "database"
    NETWORK = "network"
    RESOURCE = "resource"
    LOGIC = "logic"
    UNKNOWN = "unknown"


class AgenticError(Exception):
    """Base exception for agentic system errors"""
    
    def __init__(
        self,
        message: str,
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        details: Optional[Dict[str, Any]] = None,
        recoverable: bool = True
    ):
        super().__init__(message)
        self.message = message
        self.category = category
        self.severity = severity
        self.details = details or {}
        self.recoverable = recoverable
        self.timestamp = datetime.utcnow()


class QueryProcessingError(AgenticError):
    """Error during query processing"""
    
    def __init__(self, message: str, query: str = "", **kwargs):
        super().__init__(message, ErrorCategory.LOGIC, **kwargs)
        self.query = query


class ExternalAPIError(AgenticError):
    """Error when calling external APIs (OpenAI, Discord, etc.)"""
    
    def __init__(self, message: str, api_name: str = "", **kwargs):
        super().__init__(message, ErrorCategory.EXTERNAL_API, **kwargs)
        self.api_name = api_name


class DatabaseError(AgenticError):
    """Database operation errors"""
    
    def __init__(self, message: str, operation: str = "", **kwargs):
        super().__init__(message, ErrorCategory.DATABASE, **kwargs)
        self.operation = operation


class ConfigurationError(AgenticError):
    """Configuration-related errors"""
    
    def __init__(self, message: str, config_key: str = "", **kwargs):
        super().__init__(message, ErrorCategory.SYSTEM_CONFIG, ErrorSeverity.HIGH, **kwargs)
        self.config_key = config_key


class ResourceError(AgenticError):
    """Resource exhaustion or limit errors"""
    
    def __init__(self, message: str, resource_type: str = "", **kwargs):
        super().__init__(message, ErrorCategory.RESOURCE, **kwargs)
        self.resource_type = resource_type


class ErrorHandler:
    """Centralized error handling and recovery"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.error_counts = {}
        self.circuit_breakers = {}
        
        # Configuration
        self.max_retries = self.config.get("max_retries", 3)
        self.retry_delay = self.config.get("retry_delay", 1.0)
        self.circuit_breaker_threshold = self.config.get("circuit_breaker_threshold", 5)
        self.enable_notifications = self.config.get("enable_notifications", True)
        
        logger.info("Error handler initialized")
    
    def categorize_error(self, error: Exception) -> ErrorCategory:
        """Categorize an error based on its type and message"""
        if isinstance(error, AgenticError):
            return error.category
        
        error_str = str(error).lower()
        error_type = type(error).__name__.lower()
        
        # API-related errors
        if any(api in error_str for api in ["openai", "discord", "api", "http", "request"]):
            return ErrorCategory.EXTERNAL_API
        
        # Database errors
        if any(db in error_type for db in ["sqlite", "database", "connection", "operational"]):
            return ErrorCategory.DATABASE
        
        # Network errors
        if any(net in error_type for net in ["connection", "timeout", "network"]):
            return ErrorCategory.NETWORK
        
        # Resource errors
        if any(res in error_str for res in ["memory", "disk", "limit", "quota", "rate"]):
            return ErrorCategory.RESOURCE
        
        # Configuration errors
        if any(conf in error_str for conf in ["config", "key", "token", "missing"]):
            return ErrorCategory.SYSTEM_CONFIG
        
        return ErrorCategory.UNKNOWN
    
    def assess_severity(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> ErrorSeverity:
        """Assess error severity based on type and context"""
        if isinstance(error, AgenticError):
            return error.severity
        
        error_type = type(error).__name__.lower()
        error_str = str(error).lower()
        
        # Critical errors
        if any(critical in error_str for critical in ["critical", "fatal", "corrupt"]):
            return ErrorSeverity.CRITICAL
        
        # High severity
        if any(high in error_type for high in ["system", "config", "key"]):
            return ErrorSeverity.HIGH
        
        # Medium severity (default)
        if any(medium in error_str for medium in ["failed", "error", "exception"]):
            return ErrorSeverity.MEDIUM
        
        return ErrorSeverity.LOW
    
    def is_recoverable(self, error: Exception) -> bool:
        """Determine if an error is recoverable"""
        if isinstance(error, AgenticError):
            return error.recoverable
        
        error_str = str(error).lower()
        error_type = type(error).__name__.lower()
        
        # Non-recoverable errors
        non_recoverable = [
            "authentication", "unauthorized", "forbidden", "not found",
            "invalid token", "config", "syntax", "import"
        ]
        
        if any(nr in error_str for nr in non_recoverable):
            return False
        
        # Typically recoverable errors
        recoverable = ["timeout", "connection", "temporary", "rate limit", "busy"]
        
        if any(r in error_str for r in recoverable):
            return True
        
        return True  # Default to recoverable
    
    async def handle_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
        operation: str = "unknown"
    ) -> Dict[str, Any]:
        """Handle an error with appropriate recovery strategy"""
        category = self.categorize_error(error)
        severity = self.assess_severity(error, context)
        recoverable = self.is_recoverable(error)
        
        error_info = {
            "error": str(error),
            "type": type(error).__name__,
            "category": category.value,
            "severity": severity.value,
            "recoverable": recoverable,
            "operation": operation,
            "timestamp": datetime.utcnow().isoformat(),
            "context": context or {}
        }
        
        # Log error with appropriate level
        self._log_error(error, error_info)
        
        # Track error for circuit breaker
        self._track_error(operation, category)
        
        # Notify if enabled and severity is high
        if self.enable_notifications and severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
            await self._send_notification(error_info)
        
        return error_info
    
    def _log_error(self, error: Exception, error_info: Dict[str, Any]):
        """Log error with appropriate level"""
        severity = ErrorSeverity(error_info["severity"])
        
        if severity == ErrorSeverity.CRITICAL:
            logger.critical(f"CRITICAL ERROR in {error_info['operation']}: {error}", exc_info=True)
        elif severity == ErrorSeverity.HIGH:
            logger.error(f"HIGH SEVERITY ERROR in {error_info['operation']}: {error}")
        elif severity == ErrorSeverity.MEDIUM:
            logger.warning(f"ERROR in {error_info['operation']}: {error}")
        else:
            logger.info(f"Low severity error in {error_info['operation']}: {error}")
    
    def _track_error(self, operation: str, category: ErrorCategory):
        """Track errors for circuit breaker pattern"""
        key = f"{operation}:{category.value}"
        self.error_counts[key] = self.error_counts.get(key, 0) + 1
        
        # Implement circuit breaker
        if self.error_counts[key] >= self.circuit_breaker_threshold:
            self.circuit_breakers[key] = datetime.utcnow()
            logger.warning(f"Circuit breaker activated for {key}")
    
    def is_circuit_open(self, operation: str, category: ErrorCategory) -> bool:
        """Check if circuit breaker is open for operation"""
        key = f"{operation}:{category.value}"
        return key in self.circuit_breakers
    
    async def _send_notification(self, error_info: Dict[str, Any]):
        """Send error notification (placeholder for actual implementation)"""
        logger.info(f"Error notification would be sent: {error_info['severity']} in {error_info['operation']}")
    
    def retry_with_backoff(
        self,
        max_retries: Optional[int] = None,
        delay: Optional[float] = None,
        backoff_factor: float = 2.0,
        recoverable_only: bool = True
    ):
        """Decorator for retrying operations with exponential backoff"""
        max_retries = max_retries or self.max_retries
        delay = delay or self.retry_delay
        
        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            @wraps(func)
            async def wrapper(*args, **kwargs) -> T:
                last_error = None
                
                for attempt in range(max_retries + 1):
                    try:
                        if asyncio.iscoroutinefunction(func):
                            return await func(*args, **kwargs)
                        else:
                            return func(*args, **kwargs)
                    
                    except Exception as e:
                        last_error = e
                        
                        # Check if we should retry
                        if attempt == max_retries:
                            break
                        
                        if recoverable_only and not self.is_recoverable(e):
                            break
                        
                        # Calculate delay with exponential backoff
                        current_delay = delay * (backoff_factor ** attempt)
                        logger.info(f"Retrying operation after {current_delay:.2f}s (attempt {attempt + 1}/{max_retries + 1})")
                        
                        await asyncio.sleep(current_delay)
                
                # All retries failed
                error_info = await self.handle_error(last_error, {"attempts": max_retries + 1}, func.__name__)
                raise QueryProcessingError(
                    f"Operation failed after {max_retries + 1} attempts: {last_error}",
                    details=error_info
                )
            
            return wrapper
        return decorator
    
    def safe_execute(
        self,
        fallback_value: Any = None,
        suppress_errors: bool = False
    ):
        """Decorator for safe execution with fallback values"""
        def decorator(func: Callable[..., T]) -> Callable[..., Union[T, Any]]:
            @wraps(func)
            async def wrapper(*args, **kwargs) -> Union[T, Any]:
                try:
                    if asyncio.iscoroutinefunction(func):
                        return await func(*args, **kwargs)
                    else:
                        return func(*args, **kwargs)
                
                except Exception as e:
                    await self.handle_error(e, operation=func.__name__)
                    
                    if suppress_errors:
                        logger.debug(f"Suppressed error in {func.__name__}: {e}")
                        return fallback_value
                    else:
                        raise
            
            return wrapper
        return decorator
    
    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics"""
        return {
            "error_counts": dict(self.error_counts),
            "active_circuit_breakers": list(self.circuit_breakers.keys()),
            "total_errors": sum(self.error_counts.values())
        }


# Global error handler instance
global_error_handler = ErrorHandler()


# Convenience decorators using global handler
def retry_on_error(max_retries: int = 3, delay: float = 1.0):
    """Convenience decorator for retrying with global error handler"""
    return global_error_handler.retry_with_backoff(max_retries, delay)


def safe_async(fallback_value: Any = None):
    """Convenience decorator for safe async execution"""
    return global_error_handler.safe_execute(fallback_value) 