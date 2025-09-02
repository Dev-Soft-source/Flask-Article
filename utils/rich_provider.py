# بِسْمِ اللهِ الرَّحْمٰنِ الرَّحِيْمِ

"""
Simplified rich provider for testing purposes.
"""

class RichProvider:
    """Simplified rich provider for testing purposes."""
    
    def info(self, message):
        """Display an info message."""
        print(f"[INFO] {message}")
    
    def success(self, message):
        """Display a success message."""
        print(f"[SUCCESS] {message}")
    
    def warning(self, message):
        """Display a warning message."""
        print(f"[WARNING] {message}")
    
    def error(self, message):
        """Display an error message."""
        print(f"[ERROR] {message}")
    
    def progress_bar(self):
        """Return a dummy progress bar context manager."""
        class DummyProgressBar:
            def __enter__(self):
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                pass
            
            def add_task(self, description, total=100):
                return 0
            
            def update(self, task_id, advance=1):
                pass
        
        return DummyProgressBar()
    
    def display_token_stats(self, stats):
        """Display token usage statistics."""
        print(f"[INFO] Token usage statistics: {stats}")

# Create a singleton instance
provider = RichProvider()
