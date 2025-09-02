# بِسْمِ اللهِ الرَّحْمٰنِ الرَّحِيْمِ

from abc import ABC, abstractmethod
from typing import Optional, Any, Dict
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
    MofNCompleteColumn
)
from contextlib import contextmanager

class ProviderInterface(ABC):
    @abstractmethod
    def info(self, message: str) -> None: pass
    
    @abstractmethod
    def warning(self, message: str) -> None: pass
    
    @abstractmethod
    def error(self, message: str) -> None: pass
    
    @abstractmethod
    def success(self, message: str) -> None: pass
    
    @abstractmethod
    def debug(self, message: str) -> None: pass
    
    @abstractmethod
    def progress(self, message: str) -> None: pass
    
    @abstractmethod
    def token_usage(self, message: str) -> None: pass
    
    @abstractmethod
    @contextmanager
    def progress_bar(self) -> Any: pass
    
    @property
    @abstractmethod
    def console(self) -> Console: pass

class RichProvider(ProviderInterface):
    _instance: Optional['RichProvider'] = None
    _console: Optional[Console] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._console = Console()
        return cls._instance
    
    @property
    def console(self) -> Console:
        return self._console
    
    def info(self, message: str) -> None:
        self.console.print(f"[cyan]{message}[/]")
    
    def warning(self, message: str) -> None:
        self.console.print(f"[yellow]{message}[/]")
    
    def error(self, message: str) -> None:
        self.console.print(f"[red]{message}[/]")
    
    def success(self, message: str) -> None:
        self.console.print(f"[green]{message}[/]")
    
    def debug(self, message: str) -> None:
        self.console.print(f"[dim]{message}[/]")
    
    def progress(self, message: str) -> None:
        self.console.print(f"[blue]{message}[/]")
    
    def token_usage(self, message: str) -> None:
        self.console.print(f"[magenta]{message}[/]")
    
    def display_token_stats(self, stats: Dict[str, Any]) -> None:
        """Display token usage statistics in a formatted way."""
        self.console.print("\n[bold magenta]Token Usage Statistics:[/]")
        self.console.print(f"[magenta]Prompt Tokens: {stats['prompt_tokens']:,}[/]")
        self.console.print(f"[magenta]Completion Tokens: {stats['completion_tokens']:,}[/]")
        self.console.print(f"[magenta]Total Tokens Used: {stats['total_tokens_used']:,}[/]")
        self.console.print(f"[magenta]Context Window Usage: {stats['usage_percentage']:.1f}% ({stats['total_tokens']:,}/{stats['max_tokens']:,})[/]")
        self.console.print(f"[magenta]Available Tokens: {stats['available_tokens']:,}[/]\n")
    
    # Track if a progress bar is currently active
    _active_progress = None
    
    @contextmanager
    def progress_bar(self):
        """
        Create a progress bar context using the singleton console instance.
        Handles nested progress bars by reusing the active one if it exists.
        """
        # If there's already an active progress bar, reuse it
        if RichProvider._active_progress is not None:
            yield RichProvider._active_progress
        else:
            # Create a new progress bar
            progress = Progress(
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
            )
            
            try:
                # Set as active and start
                RichProvider._active_progress = progress
                progress.start()
                yield progress
            finally:
                # Clean up - stop and clear the active progress
                progress.stop()
                RichProvider._active_progress = None

# Global provider instance
provider = RichProvider() 