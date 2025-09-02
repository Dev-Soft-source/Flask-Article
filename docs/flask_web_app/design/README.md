# Article Generator Flask Web App UI Prototype

This folder contains HTML prototypes for the Article Generator Flask web application. These prototypes demonstrate the user interface design and interactions using Tailwind CSS.

## Prototype Pages

1. **[Home Page (index.html)](index.html)** - The main landing page with script selection
2. **[Script Selection (script_selection.html)](script_selection.html)** - Detailed comparison between Script 1 and Script 2
3. **[Script 1 Configuration (config_editor_script1.html)](config_editor_script1.html)** - Configuration editor for Script 1
4. **[Script 1 Input Editor (input_editor_script1.html)](input_editor_script1.html)** - Input editor for Script 1
5. **[Generation Page (generation.html)](generation.html)** - Article generation monitoring page
6. **[Results Page (results.html)](results.html)** - View and manage generated articles

## How to Use

1. Open any of the HTML files in a web browser to view the prototype
2. Navigate between pages using the navigation links
3. Interact with the UI elements to see how they would function in the final application

## Design Notes

- The prototypes use Tailwind CSS via CDN for styling
- Font Awesome is used for icons
- The design is responsive and works on both desktop and mobile devices
- Dark mode support is implemented with theme toggle and persistent preferences
- JavaScript is included for basic interactions, but no actual functionality is implemented

## Dark Mode

The prototype includes dark mode support:

- Toggle between light and dark modes using the sun/moon icon in the navigation bar
- Theme preference is saved in localStorage and persists between sessions
- System preference for dark/light mode is respected by default
- See [dark_mode_implementation.md](dark_mode_implementation.md) for implementation details

## Implementation Plan

These prototypes will serve as a reference for implementing the actual Flask web application. The final implementation will:

1. Convert these HTML templates to Flask templates
2. Implement the backend functionality to connect with the article generation scripts
3. Add proper form handling, validation, and data persistence
4. Implement real-time updates for the generation process

## Next Steps

After reviewing these prototypes, the next step is to implement the Flask application structure and begin converting these static HTML files into dynamic Flask templates.
