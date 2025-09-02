# Flask Web App Plan for Article Generation System

## Overview

This document outlines the plan for creating a Flask web application that provides a user-friendly interface for the existing article generation scripts (script1 and script2). The web app will allow users to configure settings, input keywords, generate articles, and view results through an intuitive browser interface.

## Current System Understanding

### Script Structure
- Two separate scripts (script1 and script2) with similar functionality but different implementations
- Both scripts generate articles based on keywords provided in CSV/text files
- Both use LLM models (OpenAI or OpenRouter) for content generation
- Both have extensive configuration options

### Key Differences Between Scripts
- **Input Format**:
  - script1 uses a simpler format (keyword,image_keyword)
  - script2 supports more complex CSV with subtitles and image keywords
- **Configuration Options**:
  - script1 and script2 have some unique configuration parameters
  - script2 has more advanced RAG (Retrieval-Augmented Generation) capabilities
  - Different default values for various settings
- **Command-line Interface**:
  - script1 takes a simple keywords file as an argument
  - script2 uses more complex command-line arguments with flags
- **Output Formatting**:
  - Subtle differences in how articles are formatted and saved
  - Different approaches to handling images and external content

### Input Handling
- Both scripts use CSV files as input
- script1 uses a simpler format (keyword,image_keyword)
- script2 supports more complex CSV with subtitles and image keywords
- Both now use a unified CSV processor, but with different processing logic

### Configuration
- Both scripts have extensive configuration options in Config classes
- Configuration includes API keys, model settings, article structure, etc.
- Environment variables are used for sensitive information
- Each script has its own set of default values and required parameters

### Article Generation
- Both scripts generate articles with titles, outlines, sections, conclusions, etc.
- Additional features include summaries, FAQs, meta descriptions, etc.
- Output can be saved as markdown or uploaded to WordPress
- Different implementation details for content generation

## Project Structure

```
flask_app/
├── app.py                  # Main Flask application
├── config.py               # Configuration for the Flask app
├── templates/              # HTML templates
│   ├── base.html           # Base template with common elements
│   ├── index.html          # Home page with script selection
│   ├── script_selection.html # Script selection page
│   ├── config_editor/      # Configuration editor templates
│   │   ├── script1.html    # Script1-specific configuration
│   │   └── script2.html    # Script2-specific configuration
│   ├── input_editor/       # Input editor templates
│   │   ├── script1.html    # Script1-specific input format
│   │   └── script2.html    # Script2-specific input format
│   ├── generation.html     # Article generation page
│   └── results.html        # Results display page
├── static/                 # Static files
│   ├── css/                # CSS files
│   ├── js/                 # JavaScript files
│   └── img/                # Image files
├── utils/                  # Utility functions
│   ├── script_runners/     # Script runner modules
│   │   ├── script1_runner.py # Functions to run script1
│   │   └── script2_runner.py # Functions to run script2
│   ├── config_handlers/    # Configuration handlers
│   │   ├── script1_config.py # Script1 configuration handler
│   │   └── script2_config.py # Script2 configuration handler
│   └── common/             # Common utilities
│       ├── session.py      # Session management
│       └── validators.py   # Input validation
└── requirements.txt        # Dependencies
```

## Pages and Functionality

### 1. Home Page (index.html)
- Welcome message and brief explanation
- **Prominent script selection interface**:
  - Clear comparison between script1 and script2 capabilities
  - Visual cards/buttons for selecting which script to use
  - Remember last used script preference
- Quick start guide and links to other pages
- Dashboard with recent generation statistics

### 2. Script Selection Page (script_selection.html)
- Detailed comparison of script1 vs script2
- Feature comparison table highlighting differences
- Use cases for each script
- Option to set default script preference
- Links to script-specific documentation

### 3. Configuration Editor (config_editor/*.html)
- **Script-specific configuration forms**:
  - Script1-specific configuration options (config_editor/script1.html)
  - Script2-specific configuration options (config_editor/script2.html)
- Organized into sections:
  - API Keys & Credentials
  - Article Structure
  - Content Generation Settings
  - Feature Toggles
  - Output Settings
  - Advanced Settings
- Save/load configuration profiles (per script)
- Tooltips explaining each option
- Validation for required fields and value ranges
- Visual indicators for script-specific options

### 4. Input Editor (input_editor/*.html)
- **Script-specific input forms**:
  - Script1 input format (simple keyword,image_keyword) (input_editor/script1.html)
  - Script2 input format (complex with subtitles) (input_editor/script2.html)
- Option to upload CSV file (with format validation for the selected script)
- Interactive editor for adding/removing keywords
- Preview of the input data
- Validation of input format based on selected script
- Templates and examples for each script format

### 5. Generation Page (generation.html)
- Start/stop article generation
- **Script-specific generation options and parameters**
- Real-time progress updates
- Log display showing console output
- Cancel button to stop generation
- Estimated time remaining
- Resource usage monitoring
- Clear indication of which script is being used

### 6. Results Page (results.html)
- Display generated articles
- Options to download, view, or edit articles
- Summary of generation results
- Filter and search capabilities
- Preview of article content
- Indication of which script generated each article

## Backend Implementation

### Flask Routes
- `/` - Home page with script selection
- `/script-selection` - Detailed script comparison page
- `/config/<script_id>` - Script-specific configuration editor
- `/input/<script_id>` - Script-specific input editor
- `/generate/<script_id>` - Script-specific article generation
- `/results` - Results display (with filtering by script)
- `/api/<script_id>/...` - Script-specific API endpoints for AJAX requests
- `/session/set-script/<script_id>` - Set active script in session

### Script Integration
- **Script-specific wrapper modules**:
  - Separate modules for script1 and script2
  - Common interface for both scripts
  - Script-specific parameter handling
- Capture and stream console output to the web interface
- Handle errors and provide meaningful feedback
- Run generation in background threads to prevent blocking
- Clear indication of which script is being used

### Configuration Management
- **Script-specific configuration handlers**:
  - Separate handlers for script1 and script2 config classes
  - Mapping between web form fields and script-specific config parameters
  - Handling of unique parameters for each script
- Load/save configuration from/to files (per script)
- Validate configuration values based on script-specific requirements
- Provide sensible defaults for each script
- Support for multiple configuration profiles (per script)
- Visual indicators for script-specific options

### Input Management
- **Script-specific input processors**:
  - Script1 format processor (simple keyword,image_keyword)
  - Script2 format processor (complex with subtitles)
- Parse and validate CSV files based on selected script format
- Generate CSV files from web form input in the correct format for each script
- Save/load input data with script-specific validation
- Clear templates and examples for each script's input format

## User Experience Enhancements

### Real-time Updates
- Use WebSockets or Server-Sent Events for real-time progress updates
- Show generation progress with progress bars
- Display log messages as they occur

### Responsive Design
- Mobile-friendly interface
- Accessible design elements
- Dark/light mode toggle

### Error Handling
- Clear error messages
- Validation of inputs before submission
- Recovery options for failed operations
- Detailed logs for troubleshooting

## Security Considerations

### API Key Management
- Store API keys securely
- Don't expose keys in client-side code
- Option to use environment variables
- Mask sensitive information in logs

### Input Validation
- Sanitize all user inputs
- Prevent command injection
- Validate file uploads

## Implementation Plan

### Phase 1: Setup and Basic Structure
1. Set up Flask application structure
2. Create base templates and static files
3. **Implement script selection mechanism and session management**
4. Implement basic routing with script-specific routes
5. Design responsive UI framework with script selection UI

### Phase 2: Script-Specific Configuration Management
1. **Analyze differences between script1 and script2 configuration options**
2. Create script-specific configuration editor interfaces
3. Implement save/load functionality for each script
4. Add validation and tooltips for script-specific options
5. Connect to existing Config classes with appropriate mappings

### Phase 3: Script-Specific Input Management
1. **Create separate input editor interfaces for each script format**
2. Implement CSV upload/download with format validation per script
3. Add interactive editing features with script-specific validation
4. Connect to existing CSV processors with appropriate format handling

### Phase 4: Script-Specific Integration
1. **Create separate wrapper modules for script1 and script2**
2. Implement common interface for both scripts
3. Implement progress tracking and output capture
4. Handle script-specific errors and timeouts
5. Add background processing with script identification

### Phase 5: Results Display with Script Identification
1. Create results interface with script filtering
2. Implement article viewing and download
3. Add filtering and search with script type as a filter option
4. Create summary statistics with script comparison

### Phase 6: Script Comparison and Selection Features
1. **Implement detailed script comparison page**
2. Add feature comparison table
3. Create script selection wizard for new users
4. Add script recommendation based on input requirements

### Phase 7: Testing and Refinement
1. Test all functionality with both scripts
2. Optimize performance
3. Improve error handling for script-specific issues
4. Add user documentation with clear script comparison

## Technical Requirements

### Dependencies
- Flask
- Flask-SocketIO (for real-time updates)
- Bootstrap or Tailwind CSS (for responsive design)
- jQuery or Vue.js (for interactive UI)
- Python 3.8+

### Browser Compatibility
- Chrome, Firefox, Safari, Edge (latest versions)
- Mobile browsers

## Conclusion

This Flask web application will provide a user-friendly interface for the existing article generation scripts, making them more accessible to users without technical knowledge. The app will maintain all the functionality of the original scripts while adding a modern, intuitive interface and additional features for managing configurations and viewing results.

A key focus of the application is to provide clear script selection and differentiation, allowing users to:

1. **Understand the differences** between script1 and script2
2. **Choose the appropriate script** for their specific needs
3. **Configure each script** with its unique parameters
4. **Input data in the correct format** for each script
5. **Track which script** generated which articles

This approach ensures that users can leverage the strengths of each script while maintaining a consistent user experience across the application. The script-specific interfaces will highlight the unique capabilities of each script while hiding unnecessary complexity.
