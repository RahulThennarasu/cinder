/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
    "./public/index.html"
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        // Claude-inspired colors
        claude: {
          bg: '#1E1E1E',         // Main background (VS Code-like)
          card: '#252526',        // Card background
          sidebar: '#252526',     // Sidebar background
          border: '#3E3E42',      // Border color
          highlight: '#3E3E42',   // Highlighted area
          hover: '#2A2D2E',       // Hover state
          text: {
            primary: '#CCCCCC',    // Primary text
            secondary: '#9CA3AF',  // Secondary text
            muted: '#6B7280',      // Muted text
          }
        },
        // VS Code dark theme-inspired syntax colors
        syntax: {
          string: '#CE9178',       // Strings
          keyword: '#569CD6',      // Keywords
          function: '#DCDCAA',     // Functions
          comment: '#6A9955',      // Comments
          variable: '#9CDCFE',     // Variables
          number: '#B5CEA8',       // Numbers
          type: '#4EC9B0',         // Types
          class: '#4EC9B0',        // Classes
          constant: '#4FC1FF',     // Constants
          property: '#9CDCFE',     // Properties
          operator: '#D4D4D4',     // Operators
        },
        // Accent colors
        accent: {
          primary: '#007ACC',      // VS Code blue
          purple: '#C586C0',       // VS Code purple
          yellow: '#DCDCAA',       // VS Code yellow
          green: '#6A9955',        // VS Code green
          red: '#F44747',          // VS Code red
          orange: '#CE9178',       // VS Code orange
        }
      },
      fontFamily: {
        sans: ['Inter', 'ui-sans-serif', 'system-ui', '-apple-system', 'sans-serif'],
        mono: ['Menlo', 'Monaco', 'Consolas', '"Liberation Mono"', '"Courier New"', 'monospace'],
      },
      boxShadow: {
        'claude': '0 4px 6px -1px rgba(0, 0, 0, 0.2), 0 2px 4px -1px rgba(0, 0, 0, 0.1)',
        'card': '0 2px 5px rgba(0, 0, 0, 0.2)',
      },
      spacing: {
        '72': '18rem',
        '84': '21rem',
        '96': '24rem',
      },
      borderRadius: {
        'claude': '0.625rem',
      }
    },
  },
  plugins: [],
}