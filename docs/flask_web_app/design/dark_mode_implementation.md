# Dark Mode Implementation Guide

This document explains how to implement dark mode in the Flask web app HTML prototypes.

## Overview

Dark mode has been implemented in the `index.html` file and should be applied to all other HTML files in the prototype. The implementation uses Tailwind CSS's dark mode feature with the `class` strategy.

## Key Components

1. **HTML Root Class**: The `<html>` tag has a `class="light"` attribute that serves as the default theme.

2. **Tailwind Configuration**: The dark mode is configured in the head section:
   ```javascript
   tailwind.config = {
       darkMode: 'class',
       theme: {
           extend: {
               colors: {
                   // Custom colors can be defined here
               }
           }
       }
   }
   ```

3. **Theme Toggle Button**: A button in the navigation bar toggles between light and dark modes:
   ```html
   <button id="theme-toggle" class="text-white focus:outline-none">
       <i class="fas fa-moon text-xl dark:hidden"></i>
       <i class="fas fa-sun text-xl hidden dark:block"></i>
   </button>
   ```

4. **Dark Mode Classes**: Elements use `dark:` variants for styling in dark mode:
   ```html
   <div class="bg-white dark:bg-gray-800 text-gray-800 dark:text-white">
   ```

5. **JavaScript for Theme Toggling**:
   ```javascript
   // Dark mode toggle
   document.getElementById('theme-toggle').addEventListener('click', function() {
       if (document.documentElement.classList.contains('dark')) {
           // Switch to light mode
           document.documentElement.classList.remove('dark');
           localStorage.setItem('theme', 'light');
       } else {
           // Switch to dark mode
           document.documentElement.classList.add('dark');
           localStorage.setItem('theme', 'dark');
       }
   });

   // Check for saved theme preference or respect OS preference
   document.addEventListener('DOMContentLoaded', function() {
       // Check for saved theme preference
       const savedTheme = localStorage.getItem('theme');
       
       if (savedTheme === 'dark') {
           document.documentElement.classList.add('dark');
       } else if (savedTheme === 'light') {
           document.documentElement.classList.remove('dark');
       } else {
           // If no saved preference, check OS preference
           if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
               document.documentElement.classList.add('dark');
           }
       }
   });
   ```

## Common Color Mappings

When implementing dark mode in other HTML files, use these common color mappings:

| Light Mode | Dark Mode |
|------------|-----------|
| `bg-gray-50` | `dark:bg-gray-900` |
| `bg-white` | `dark:bg-gray-800` |
| `bg-gray-100` | `dark:bg-gray-700` |
| `text-gray-800` | `dark:text-white` |
| `text-gray-700` | `dark:text-gray-300` |
| `text-gray-600` | `dark:text-gray-300` |
| `border-gray-200` | `dark:border-gray-700` |
| `bg-indigo-600` | `dark:bg-indigo-700` |
| `bg-indigo-700` | `dark:bg-indigo-900` |

## Implementation Steps for Other Files

1. Add `class="light"` to the `<html>` tag
2. Add the Tailwind configuration in the `<head>` section
3. Add the theme toggle button to the navigation bar
4. Add dark mode variants to all color classes
5. Add the theme toggle JavaScript to the end of the file
6. Add transition classes (`transition-colors duration-200`) to elements that change color

## Testing

After implementing dark mode in a file:
1. Open the file in a browser
2. Click the theme toggle button to switch between light and dark modes
3. Refresh the page to ensure the theme preference is saved
4. Check that all elements have appropriate contrast in both modes
