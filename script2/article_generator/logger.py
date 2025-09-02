# بِسْمِ اللهِ الرَّحْمٰنِ الرَّحِيْمِ

import os
import json
import time
from datetime import datetime
import traceback
from typing import List, Dict, Any, Optional
from contextlib import contextmanager

from utils.rich_provider import RichProvider, provider
from utils.error_utils import ErrorHandler, format_error_message

class RAGLogger(RichProvider):
    _instance: Optional['RAGLogger'] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_log_storage()
            cls._instance._error_handler = ErrorHandler(show_traceback=True)
        return cls._instance
    
    def _init_log_storage(self):
        """Initialize storage for logs."""
        self.log_dir = "logs"
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Create subdirectories for different log types
        os.makedirs(os.path.join(self.log_dir, "url_access"), exist_ok=True)
        os.makedirs(os.path.join(self.log_dir, "content_scraping"), exist_ok=True)
        os.makedirs(os.path.join(self.log_dir, "embeddings"), exist_ok=True)
        os.makedirs(os.path.join(self.log_dir, "rag_search"), exist_ok=True)
        os.makedirs(os.path.join(self.log_dir, "rag_context"), exist_ok=True)
        
        # Initialize session log
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_log_file = os.path.join(self.log_dir, f"session_{self.session_id}.log")
        
        # Create initial session log file
        with open(self.session_log_file, "w", encoding='utf-8') as f:
            f.write(f"RAG Session Log: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
        
        # Initialize counters and stats
        self.stats = {
            "urls_accessed": 0,
            "successful_urls": 0,
            "failed_urls": 0,
            "content_scraped_bytes": 0,
            "embedding_chunks_created": 0,
            "rag_searches_performed": 0,
            "rag_contexts_generated": 0,
            "errors": 0,
            "start_time": time.time()
        }
        
    def _truncate_long_string(self, text, max_length=500):
        """Truncate long strings to show first and last parts with ellipsis in the middle.
        
        Args:
            text: The string to truncate
            max_length: Maximum length to show on each end of the string
            
        Returns:
            Truncated string with ellipsis in the middle if longer than 2*max_length
        """
        if not isinstance(text, str):
            try:
                text = str(text)
            except:
                return text
                
        if len(text) <= 2 * max_length + 3:  # +3 for the ellipsis
            return text
        
        return f"{text[:max_length]}...{text[-max_length:]}"
        
        # Initialize counters and stats
        self.stats = {
            "urls_accessed": 0,
            "successful_urls": 0,
            "failed_urls": 0,
            "content_scraped_bytes": 0,
            "embedding_chunks_created": 0,
            "rag_searches_performed": 0,
            "rag_contexts_generated": 0,
            "errors": 0,
            "start_time": time.time()
        }
        
        # Initialize session log
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_log_file = os.path.join(self.log_dir, f"session_{self.session_id}.log")
        
        with open(self.session_log_file, "w", encoding='utf-8') as f:
            f.write(f"RAG Session Log: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
    
    def _log_to_session(self, message: str):
        """Log message to the session log file."""
        with open(self.session_log_file, "a", encoding='utf-8') as f:
            timestamp = datetime.now().strftime("%H:%M:%S")
            f.write(f"[{timestamp}] {message}\n")
    
    def info(self, message: str) -> None:
        """Log an info message to the console and session log."""
        super().info(message)
        self._log_to_session(f"INFO: {message}")
    
    def warning(self, message: str) -> None:
        """Log a warning message to the console and session log."""
        super().warning(message)
        self._log_to_session(f"WARNING: {message}")
    
    def error(self, message: str, show_traceback: bool = True):
        """Log an error message with traceback if requested."""
        from rich.panel import Panel
        from rich.syntax import Syntax
        from rich.text import Text
        
        try:
            if isinstance(message, Exception):
                # Handle direct exceptions
                context = {"logger": "RAGLogger"}
                
                # Special handling for HTTPErrors
                if hasattr(message, 'response'):
                    resp = message.response
                    status = f"{resp.status_code} {resp.reason}"
                    url = resp.url
                    context.update({
                        "status": status,
                        "url": url,
                        "headers": dict(resp.headers)
                    })
                    
                    error_panel = Panel(
                        Text(f"HTTP {status}\nURL: {url}\n", style="bold red") +
                        Text(f"Error message: {str(message)}"),
                        title="API Request Failed",
                        border_style="red"
                    )
                    self.console.print(error_panel)
                    
                    # Include response content if available
                    if resp.content:
                        try:
                            resp_json = resp.json()
                            error_syntax = Syntax(
                                json.dumps(resp_json, indent=2),
                                "json",
                                theme="monokai"
                            )
                            self.console.print("\n[bold]Response:[/]")
                            self.console.print(error_syntax)
                        except:
                            self.console.print("\n[bold]Response:[/]")
                            self.console.print(resp.text[:1000] + "...")
                
                # Handle other exceptions
                else:
                    error_panel = Panel(
                        Text(f"{message.__class__.__name__}: {str(message)}", style="bold red"),
                        title="Error Occurred",
                        border_style="red"
                    )
                    self.console.print(error_panel)
                
                # Process traceback
                trace = traceback.format_exc()
                if trace and trace != "NoneType: None\n":
                    self._error_handler.handle_error(Exception(trace), context, severity="debug")
                    if show_traceback:
                        trace_syntax = Syntax(trace, "python", theme="monokai")
                        self.console.print("\n[bold]Traceback:[/]")
                        self.console.print(trace_syntax)
                
                error_message = format_error_message(message, context)
                self._log_to_session(f"ERROR: {error_message}")
            else:
                # Handle string messages
                try:
                    raise Exception(message)
                except Exception as e:
                    error_panel = Panel(
                        Text(message, style="bold red"),
                        title="Error Occurred",
                        border_style="red"
                    )
                    self.console.print(error_panel)
                    
                    context = {"logger": "RAGLogger", "custom_message": True}
                    self._error_handler.handle_error(e, context, severity="error")
                    
                    trace = traceback.format_exc()
                    if show_traceback and trace != "NoneType: None\n":
                        trace_syntax = Syntax(trace, "python", theme="monokai")
                        self.console.print("\n[bold]Traceback:[/]")
                        self.console.print(trace_syntax)
                    
                    self._log_to_session(f"ERROR: {message}")
                    
            # Update error count
            self.stats["errors"] += 1
            
        except Exception as e:
            self._log_to_session(f"CRITICAL: Failed to log error: {str(e)}")
            self._error_handler.handle_error(e, severity="critical")
    
    def success(self, message: str) -> None:
        """Log a success message to the console and session log."""
        super().success(message)
        self._log_to_session(f"SUCCESS: {message}")
    
    def debug(self, message: str) -> None:
        """Log a debug message to the console and session log."""
        # Check if this is an API-related message that needs truncation
        if any(x in str(message).lower() for x in ["api payload", "api request", "api response"]):
            truncated_message = self._truncate_long_string(message)
            super().debug(truncated_message)
            # Still log the full message to the file
            self._log_to_session(f"DEBUG: {message}")
        else:
            super().debug(message)
            self._log_to_session(f"DEBUG: {message}")
    
    def log_url_access(self, url: str, status_code: int = 0, success: bool = True) -> None:
        """Log URL access attempt with status."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_data = {
            "timestamp": timestamp,
            "url": url,
            "status_code": status_code,
            "success": success
        }
        
        # Update stats
        self.stats["urls_accessed"] += 1
        if success:
            self.stats["successful_urls"] += 1
            self.success(f"URL access successful: {url} (Status: {status_code})")
        else:
            self.stats["failed_urls"] += 1
            # Create an exception for the failed URL access
            try:
                raise Exception(f"URL access failed: {url} (Status: {status_code})")
            except Exception as e:
                context = {"url": url, "status_code": status_code}
                self._error_handler.handle_error(e, context, severity="error")
        
        # Save to file
        self.log_to_file("url_access", log_data)

    def log_content_scraping(self, url: str, content_length: int, title: str = None) -> None:
        """Log content scraping results."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_data = {
            "timestamp": timestamp,
            "url": url,
            "content_length": content_length,
            "title": title or "Unknown Title"
        }
        
        # Update stats
        self.stats["content_scraped_bytes"] += content_length
        
        # Log message
        self.info(f"Scraped content from {url}: {content_length} bytes, Title: {title or 'Unknown'}")
        
        # Save to file
        self.log_to_file("content_scraping", log_data)

    def log_embedding_generation(self, text_chunks: int, vector_dim: int, source: str = None) -> None:
        """Log embedding generation statistics."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_data = {
            "timestamp": timestamp,
            "text_chunks": text_chunks,
            "vector_dimension": vector_dim,
            "source": source or "Unknown"
        }
        
        # Update stats
        self.stats["embedding_chunks_created"] += text_chunks
        
        # Log message
        self.info(f"Generated embeddings for {text_chunks} chunks, dimension: {vector_dim}")
        
        # Save to file
        self.log_to_file("embeddings", log_data)

    def log_rag_search(self, query: str, results_count: int, top_score: float = 0.0) -> None:
        """Log RAG retrieval search results."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_data = {
            "timestamp": timestamp,
            "query": query,
            "results_count": results_count,
            "top_score": top_score
        }
        
        # Update stats
        self.stats["rag_searches_performed"] += 1
        
        # Log message
        self.info(f"RAG search for '{query}' returned {results_count} results (top score: {top_score:.4f})")
        
        # Save to file
        self.log_to_file("rag_search", log_data)

    def log_rag_context(self, context_length: int, chunks_used: int, sources: List[str] = None) -> None:
        """Log RAG context generation statistics."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_data = {
            "timestamp": timestamp,
            "context_length": context_length,
            "chunks_used": chunks_used,
            "sources": sources or []
        }
        
        # Update stats
        self.stats["rag_contexts_generated"] += 1
        
        # Log message
        self.info(f"Generated RAG context with {context_length} chars from {chunks_used} chunks")
        
        # Save to file
        self.log_to_file("rag_context", log_data)

    def log_to_file(self, log_type: str, data: Dict[str, Any]) -> str:
        """Save log data to a file in the appropriate directory."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = os.path.join(self.log_dir, log_type, f"{log_type}_{timestamp}.json")
            
            with open(log_file, "w", encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            return log_file
        except Exception as e:
            context = {"log_type": log_type, "data_size": len(str(data))}
            self._error_handler.handle_error(e, context, severity="error")
            return ""

    def display_rag_stats(self) -> None:
        """Display RAG system statistics in a formatted way."""
        try:
            elapsed_time = time.time() - self.stats["start_time"]
            self.console.print("\n[bold cyan]RAG System Statistics:[/]")
            self.console.print(f"[cyan]Session Duration: {elapsed_time:.2f} seconds[/]")
            self.console.print(f"[cyan]URLs Accessed: {self.stats['urls_accessed']} (Success: {self.stats['successful_urls']}, Failed: {self.stats['failed_urls']})[/]")
            self.console.print(f"[cyan]Content Scraped: {self.stats['content_scraped_bytes']/1024:.2f} KB[/]")
            self.console.print(f"[cyan]Embedding Chunks Created: {self.stats['embedding_chunks_created']}[/]")
            self.console.print(f"[cyan]RAG Searches Performed: {self.stats['rag_searches_performed']}[/]")
            self.console.print(f"[cyan]RAG Contexts Generated: {self.stats['rag_contexts_generated']}[/]")
            self.console.print(f"[cyan]Errors Encountered: {self.stats['errors']}[/]\n")
        except Exception as e:
            context = {"stats": self.stats}
            self._error_handler.handle_error(e, context, severity="error")

# Create global instance
logger = RAGLogger()