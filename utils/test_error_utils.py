import unittest
from unittest.mock import patch
from .error_utils import ErrorHandler, format_error_message

class TestErrorUtils(unittest.TestCase):
    def test_error_handler_basic_error(self):
        """Test handling a basic error"""
        handler = ErrorHandler()
        error = ValueError("Invalid value")
        
        with patch('utils.error_utils.console.print') as mock_print:
            handler.handle_error(error)
            
            # Verify the error was printed
            mock_print.assert_called_once()
            # Get the rendered output from the console
            output = str(mock_print.call_args[0][0].renderable)
            self.assertIn("ValueError", output)
            self.assertIn("Invalid value", output)

    def test_error_handler_with_context(self):
        """Test handling an error with context"""
        handler = ErrorHandler()
        error = TypeError("Invalid type")
        context = {"file": "test.py", "line": 42}
        
        with patch('utils.error_utils.console.print') as mock_print:
            handler.handle_error(error, context=context)
            
            # Verify context was included
            mock_print.assert_called_once()
            # Get the rendered output from the console
            output = str(mock_print.call_args[0][0].renderable)
            self.assertIn("file: test.py", output)
            self.assertIn("line: 42", output)

    def test_format_error_message(self):
        """Test formatting error messages"""
        error = RuntimeError("Something went wrong")
        context = {"user": "test_user", "action": "process_data"}
        
        # Test without context
        message = format_error_message(error)
        self.assertEqual("RuntimeError: Something went wrong", message)
        
        # Test with context
        message = format_error_message(error, context)
        self.assertEqual(
            "RuntimeError: Something went wrong [Context: user=test_user, action=process_data]",
            message
        )

    def test_error_handler_severity_levels(self):
        """Test different severity levels"""
        handler = ErrorHandler()
        error = Exception("Test error")
        
        for severity in ["debug", "info", "warning", "error", "critical"]:
            with patch('utils.error_utils.console.print') as mock_print:
                handler.handle_error(error, severity=severity)
                
                # Verify severity level was used
                mock_print.assert_called_once()
                # Get the rendered output from the console
                panel = mock_print.call_args[0][0]
                self.assertIn(severity.upper(), panel.title)
                mock_print.reset_mock()

    def test_invalid_severity_level(self):
        """Test handling with invalid severity level"""
        handler = ErrorHandler()
        error = Exception("Test error")
        
        with patch('utils.error_utils.console.print') as mock_print:
            handler.handle_error(error, severity="invalid")
            
            # Verify default severity (error) was used
            panel = mock_print.call_args[0][0]
            self.assertIn("ERROR", panel.title)

if __name__ == '__main__':
    unittest.main()