# بِسْمِ اللهِ الرَّحْمٰنِ الرَّحِيْمِ

from abc import ABC, abstractmethod
from typing import Optional, Any, Dict, List, Union
from rich.console import Console
from utils.error_utils import ErrorHandler

# Initialize error handler
error_handler = ErrorHandler()
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
    MofNCompleteColumn
)
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax
from rich.markdown import Markdown
from rich.box import Box
from contextlib import contextmanager
import os
import json
import time
from datetime import datetime
import traceback

class LoggerInterface(ABC):
    @abstractmethod
    def info(self, message: str) -> None: pass
    
    @abstractmethod
    def warning(self, message: str) -> None: pass
    
    @abstractmethod
    def error(self, message: str, show_traceback: bool = True) -> None: pass
    
    @abstractmethod
    def success(self, message: str) -> None: pass
    
    @abstractmethod
    def debug(self, message: str) -> None: pass
    
    @abstractmethod
    def progress(self, message: str) -> None: pass
    
    @abstractmethod
    @contextmanager
    def progress_bar(self) -> Any: pass
    
    @property
    @abstractmethod
    def console(self) -> Console: pass
    
    @abstractmethod
    def log_url_access(self, url: str, status_code: int = 0, success: bool = True) -> None: pass
    
    @abstractmethod
    def log_content_scraping(self, url: str, content_length: int, title: str = None) -> None: pass
    
    @abstractmethod
    def log_embedding_generation(self, text_chunks: int, vector_dim: int, source: str = None) -> None: pass
    
    @abstractmethod
    def log_rag_search(self, query: str, results_count: int, top_score: float = 0.0) -> None: pass
    
    @abstractmethod
    def log_rag_context(self, context_length: int, chunks_used: int, sources: List[str] = None) -> None: pass
    
    @abstractmethod
    def log_to_file(self, log_type: str, data: Dict[str, Any]) -> str: pass
    
    @abstractmethod
    def display_rag_stats(self) -> None: pass

class RichLogger(LoggerInterface):
    _instance: Optional['RichLogger'] = None
    _console: Optional[Console] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._console = Console()
            cls._instance._init_log_storage()
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
    
    @property
    def console(self) -> Console:
        return self._console
    
    def _log_to_session(self, message: str):
        """Log message to the session log file."""
        with open(self.session_log_file, "a", encoding='utf-8') as f:
            timestamp = datetime.now().strftime("%H:%M:%S")
            f.write(f"[{timestamp}] {message}\n")
    
    def info(self, message: str) -> None:
        self.console.print(f"[cyan]{message}[/]")
        self._log_to_session(f"INFO: {message}")
    
    def warning(self, message: str) -> None:
        self.console.print(f"[yellow]{message}[/]")
        self._log_to_session(f"WARNING: {message}")
    
    def error(self, message: str, show_traceback: bool = True):
        """Log an error message with traceback if requested"""
        # Use centralized error handler
        error_handler.handle_error(Exception(message), severity="error")
        
        # Add traceback information when show_traceback is True
        if show_traceback:
            trace = traceback.format_exc()
            if trace and trace != "NoneType: None\n":
                error_handler.handle_error(Exception(trace), severity="debug")
                
        # Update error count
        self.stats["errors"] += 1
    
    def success(self, message: str) -> None:
        self.console.print(f"[green]{message}[/]")
        self._log_to_session(f"SUCCESS: {message}")
    
    def debug(self, message: str) -> None:
        # Check if this is an API-related message that needs truncation
        if any(x in str(message).lower() for x in ["api payload", "api request", "api response"]):
            truncated_message = self._truncate_long_string(message)
            self.console.print(f"[dim]{truncated_message}[/]")
            # Still log the full message to the file
            self._log_to_session(f"DEBUG: {message}")
        else:
            self.console.print(f"[dim]{message}[/]")
            self._log_to_session(f"DEBUG: {message}")
    
    def progress(self, message: str) -> None:
        self.console.print(f"[blue]{message}[/]")
        self._log_to_session(f"PROGRESS: {message}")
    
    @contextmanager
    def progress_bar(self):
        """Create a progress bar context using the singleton console instance."""
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(bar_width=50, complete_style="green"),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=self.console,
            transient=True,
            expand=True,
            refresh_per_second=10
        ) as progress:
            yield progress

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
            self.error(f"URL access failed: {url} (Status: {status_code})")
        
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
        self.info(f"RAG search for '{query}': Found {results_count} results (Top score: {top_score:.4f})")
        
        # Save to file
        self.log_to_file("rag_search", log_data)

    def log_rag_context(self, context_length: int, chunks_used: int, sources: List[str] = None) -> None:
        """Log RAG context generation."""
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
        source_count = len(sources) if sources else 0
        self.info(f"Generated RAG context: {context_length} chars from {chunks_used} chunks ({source_count} sources)")
        
        # Save to file
        self.log_to_file("rag_context", log_data)

    def log_to_file(self, log_type: str, data: Dict[str, Any]) -> str:
        """Save log data to a file in the appropriate directory."""
        log_subdir = os.path.join(self.log_dir, log_type)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = os.path.join(log_subdir, f"{log_type}_{timestamp}.json")
        
        with open(filename, "w", encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        return filename

    def display_rag_stats(self) -> None:
        """Display comprehensive RAG statistics in a Rich table."""
        # Calculate elapsed time
        elapsed_time = time.time() - self.stats["start_time"]
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        elapsed_str = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
        
        # Create stats table
        table = Table(title="RAG System Statistics", box=Box.DOUBLE_EDGE)
        
        # Add columns
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        # Add rows
        table.add_row("Session Duration", elapsed_str)
        table.add_row("URLs Accessed", str(self.stats["urls_accessed"]))
        table.add_row("Successful URL Accesses", str(self.stats["successful_urls"]))
        table.add_row("Failed URL Accesses", str(self.stats["failed_urls"]))
        table.add_row("Content Scraped", f"{self.stats['content_scraped_bytes'] / 1024:.2f} KB")
        table.add_row("Embedding Chunks Created", str(self.stats["embedding_chunks_created"]))
        table.add_row("RAG Searches Performed", str(self.stats["rag_searches_performed"]))
        table.add_row("RAG Contexts Generated", str(self.stats["rag_contexts_generated"]))
        
        # Success rate calculation
        if self.stats["urls_accessed"] > 0:
            success_rate = (self.stats["successful_urls"] / self.stats["urls_accessed"]) * 100
            table.add_row("URL Access Success Rate", f"{success_rate:.2f}%")
        
        # Display the table
        self.console.print("\n")
        self.console.print(table)
        self.console.print("\n")
        
        # Log to session file
        self._log_to_session(f"=== RAG STATS ===")
        for key, value in self.stats.items():
            if key != "start_time":
                self._log_to_session(f"{key}: {value}")
        self._log_to_session(f"Session Duration: {elapsed_str}")
        self._log_to_session("================")

# Global logger instance
logger = RichLogger() 