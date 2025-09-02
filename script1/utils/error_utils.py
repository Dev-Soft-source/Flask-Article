from typing import Optional, Dict, Any
import traceback
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.theme import Theme

# Define custom theme for error messages
error_theme = Theme({
    "debug": "dim",
    "info": "bold blue",
    "warning": "bold yellow",
    "error": "bold red",
    "critical": "bold white on red",
    "context": "dim",
    "traceback": "dim red"
})

console = Console(theme=error_theme)

class ErrorHandler:
    """
    Standardized error handler for both scripts.
    Provides consistent error formatting and handling.
    """
    
    def __init__(self, show_traceback: bool = True):
        self.show_traceback = show_traceback
        
    def handle_error(self, error: Exception, context: Optional[Dict[str, Any]] = None,
                    severity: str = "error") -> None:
        """
        Handle and display an error with standardized formatting.
        
        Args:
            error: The exception that occurred
            context: Additional context about the error
            severity: Error severity level (debug, info, warning, error, critical)
        """
        # Validate severity level
        if severity not in ["debug", "info", "warning", "error", "critical"]:
            severity = "error"
            
        # Format message based on severity
        panel_content = Text()
        
        if severity in ["error", "critical"]:
            error_type = type(error).__name__
            panel_content.append(f"{error_type}: ", style=severity)
            panel_content.append(str(error))
        else:
            # For non-error messages, show clean text without exception type
            panel_content.append(str(error), style=severity)
        
        # Add context if provided
        if context:
            panel_content.append("\n\nContext:", style="context")
            for key, value in context.items():
                panel_content.append(f"\n{key}: {value}", style="context")
                
        # Add traceback if enabled
        if self.show_traceback:
            tb = traceback.format_exc()
            if tb and tb != "NoneType: None\n":
                panel_content.append("\n\nTraceback:", style="traceback")
                panel_content.append(f"\n{tb}", style="traceback")
                
        # Display the error
        console.print(Panel(
            panel_content,
            title=f"[{severity.upper()}]",
            border_style=severity
        ))

def format_error_message(error: Exception, context: Optional[Dict[str, Any]] = None) -> str:
    """
    Format an error message with context for logging.
    
    Args:
        error: The exception that occurred
        context: Additional context about the error
        
    Returns:
        Formatted error message string
    """
    error_type = type(error).__name__
    message = f"{error_type}: {str(error)}"
    
    if context:
        context_str = ", ".join(f"{k}={v}" for k, v in context.items())
        message += f" [Context: {context_str}]"
        
    return message