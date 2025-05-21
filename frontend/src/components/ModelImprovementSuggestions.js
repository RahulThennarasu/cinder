import React, { useState, useEffect } from 'react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';

const ModelImprovementSuggestions = () => {
  const [suggestions, setSuggestions] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('pytorch');
  const [activeSection, setActiveSection] = useState(null);
  const [generatingCode, setGeneratingCode] = useState(false);
  const [generatedCode, setGeneratedCode] = useState({});
  
  useEffect(() => {
    const fetchSuggestions = async () => {
      try {
        setLoading(true);
        // This endpoint was defined in your backend code
        const response = await fetch('http://localhost:8000/api/model-improvements?detail_level=comprehensive');
        
        if (!response.ok) {
          throw new Error(`Server responded with status: ${response.status}`);
        }
        
        const data = await response.json();
        setSuggestions(data);
        
        // Initialize generated code storage
        const codeStorage = {};
        if (data && data.suggestions && data.suggestions.length > 0) {
          data.suggestions.forEach((suggestion, index) => {
            codeStorage[index] = {};
          });
          setGeneratedCode(codeStorage);
          
          // Set the first suggestion as active by default
          setActiveSection(0);
        }
        
      } catch (err) {
        console.error('Error fetching improvement suggestions:', err);
        setError('Could not load improvement suggestions. Check if the server is running.');
      } finally {
        setLoading(false);
      }
    };
    
    fetchSuggestions();
  }, []);
  
  const generateCode = async (suggestionIndex, framework) => {
    if (generatingCode) return; // Prevent multiple simultaneous requests
    
    try {
      setGeneratingCode(true);
      
      // Check if we already have generated code for this suggestion and framework
      if (generatedCode[suggestionIndex] && generatedCode[suggestionIndex][framework]) {
        // We already have code for this, no need to generate again
        setActiveTab(framework);
        return;
      }
      
      // Get the suggestion title to use as category
      const suggestion = suggestions.suggestions[suggestionIndex];
      const category = suggestion.title.toLowerCase().replace(/\s+/g, '_');
      
      // Call the API to generate code
      const response = await fetch(
        `http://localhost:8000/api/generate-code-example?framework=${framework}&category=${category}`
      );
      
      if (!response.ok) {
        throw new Error(`Server responded with status: ${response.status}`);
      }
      
      const data = await response.json();
      
      // Update the generated code state
      setGeneratedCode(prev => ({
        ...prev,
        [suggestionIndex]: {
          ...prev[suggestionIndex],
          [framework]: data.code
        }
      }));
      
      // Switch to the generated code's framework tab
      setActiveTab(framework);
      
    } catch (err) {
      console.error('Error generating code:', err);
      // Set error code in the generated code state
      setGeneratedCode(prev => ({
        ...prev,
        [suggestionIndex]: {
          ...prev[suggestionIndex],
          [framework]: `# Error generating code: ${err.message}\n# Please try again later.`
        }
      }));
    } finally {
      setGeneratingCode(false);
    }
  };
  
  // Function to get code for the current suggestion and framework
  const getCurrentCode = () => {
    if (activeSection === null) return '';
    
    // Check if we have generated code for this suggestion and framework
    if (generatedCode[activeSection] && generatedCode[activeSection][activeTab]) {
      return generatedCode[activeSection][activeTab];
    }
    
    // Otherwise return a placeholder
    return `# Click "Generate Code" to create a ${activeTab} implementation`;
  };
  
  // Render loading state
  if (loading) {
    return (
      <div className="loading-container">
        <div className="spinner"></div>
        <p>Loading model improvement suggestions...</p>
      </div>
    );
  }
  
  // Render error state
  if (error) {
    return (
      <div className="error-container">
        <h3>Error</h3>
        <p>{error}</p>
        <button onClick={() => window.location.reload()}>Retry</button>
      </div>
    );
  }
  
  return (
    <div className="model-suggestions">
      <div className="model-suggestions-header">
        <div className="metrics">
          <div className="metric">
            <span className="metric-value">{suggestions?.model_accuracy ? (suggestions.model_accuracy * 100).toFixed(1) : 0}%</span>
            <span className="metric-label">Current Accuracy</span>
          </div>
          <div className="metric">
            <span className="metric-value">{suggestions?.error_rate ? (suggestions.error_rate * 100).toFixed(1) : 0}%</span>
            <span className="metric-label">Error Rate</span>
          </div>
          <div className="metric">
            <span className={`metric-value ${suggestions?.improvement_potential || ''}`}>
              {suggestions?.improvement_potential ? suggestions.improvement_potential.charAt(0).toUpperCase() + suggestions.improvement_potential.slice(1) : 'Unknown'}
            </span>
            <span className="metric-label">Potential</span>
          </div>
        </div>
        
        <div className="framework-selector">
          <div className="framework-label">Code Examples:</div>
          <div className="framework-options">
            <button 
              className={activeTab === 'pytorch' ? 'active' : ''}
              onClick={() => setActiveTab('pytorch')}
            >
              PyTorch
            </button>
            <button 
              className={activeTab === 'tensorflow' ? 'active' : ''}
              onClick={() => setActiveTab('tensorflow')}
            >
              TensorFlow
            </button>
            <button 
              className={activeTab === 'sklearn' ? 'active' : ''}
              onClick={() => setActiveTab('sklearn')}
            >
              scikit-learn
            </button>
          </div>
        </div>
      </div>
      
      {!suggestions?.suggestions || suggestions.suggestions.length === 0 ? (
        <div className="no-suggestions">
          <p>No improvement suggestions available. This may indicate that your model is already well-optimized.</p>
        </div>
      ) : (
        <div className="suggestions-container">
          <div className="suggestions-sidebar">
            {suggestions.suggestions.map((suggestion, index) => (
              <div 
                key={index}
                className={`suggestion-item ${activeSection === index ? 'active' : ''}`}
                onClick={() => setActiveSection(index)}
              >
                <div className="suggestion-icon"></div>
                <div className="suggestion-info">
                  <span className="suggestion-title">{suggestion.title}</span>
                  <span className="suggestion-category">{suggestion.category}</span>
                </div>
              </div>
            ))}
          </div>
          
          <div className="suggestion-content">
            {activeSection !== null && (
              <div className="suggestion-details">
                <h2 className="suggestion-title">{suggestions.suggestions[activeSection].title}</h2>
                
                <div className="suggestion-section">
                  <h3>Issue</h3>
                  <p>{suggestions.suggestions[activeSection].issue}</p>
                </div>
                
                <div className="suggestion-section">
                  <h3>Recommendation</h3>
                  <p>{suggestions.suggestions[activeSection].suggestion}</p>
                </div>
                
                <div className="suggestion-section">
                  <h3>Implementation</h3>
                  <div className="code-container">
                    <div className="code-header">
                      <div className="generate-options">
                        <button 
                          onClick={() => generateCode(activeSection, activeTab)}
                          disabled={generatingCode}
                          className="generate-button"
                        >
                          {generatingCode ? 'Generating...' : 'Generate Code'}
                        </button>
                        <div className="api-message">
                          {generatingCode && <span>Using Gemini AI to create code...</span>}
                        </div>
                      </div>
                    </div>
                    
                    <SyntaxHighlighter 
                      language="python"
                      style={vscDarkPlus}
                      showLineNumbers={true}
                    >
                      {getCurrentCode()}
                    </SyntaxHighlighter>
                    
                    <div className="code-actions">
                      <button
                        onClick={() => {
                          const code = getCurrentCode();
                          navigator.clipboard.writeText(code);
                          alert('Code copied to clipboard!');
                        }}
                        disabled={!generatedCode[activeSection] || !generatedCode[activeSection][activeTab]}
                      >
                        Copy Code
                      </button>
                    </div>
                  </div>
                </div>
                
                <div className="suggestion-footer">
                  <div className="impact-indicator">
                    <span className="impact-label">Expected Impact:</span>
                    <span className="impact-value">{suggestions.suggestions[activeSection].expected_impact || 'Medium'}</span>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
      
      <style jsx>{`
        .model-suggestions {
          background-color: #f8f9fa;
          border-radius: 8px;
          overflow: hidden;
          box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        }
        
        .model-suggestions-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 16px 24px;
          background-color: white;
          border-bottom: 1px solid #e9ecef;
        }
        
        .metrics {
          display: flex;
          gap: 24px;
        }
        
        .metric {
          display: flex;
          flex-direction: column;
        }
        
        .metric-value {
          font-size: 24px;
          font-weight: 600;
        }
        
        .metric-value.high {
          color: #e74c32;
        }
        
        .metric-value.medium {
          color: #f59e0b;
        }
        
        .metric-value.low {
          color: #10b981;
        }
        
        .metric-label {
          font-size: 14px;
          color: #6c757d;
        }
        
        .framework-selector {
          display: flex;
          align-items: center;
          gap: 12px;
        }
        
        .framework-label {
          font-size: 14px;
          font-weight: 500;
        }
        
        .framework-options {
          display: flex;
          gap: 8px;
        }
        
        .framework-options button {
          padding: 6px 12px;
          border: 1px solid #dee2e6;
          background-color: white;
          border-radius: 4px;
          font-size: 14px;
          cursor: pointer;
          transition: all 0.2s;
        }
        
        .framework-options button.active {
          background-color: #e74c32;
          color: white;
          border-color: #e74c32;
        }
        
        .suggestions-container {
          display: flex;
          height: 600px;
        }
        
        .suggestions-sidebar {
          width: 280px;
          background-color: white;
          border-right: 1px solid #e9ecef;
          overflow-y: auto;
        }
        
        .suggestion-item {
          display: flex;
          align-items: center;
          padding: 16px;
          border-bottom: 1px solid #f1f3f5;
          cursor: pointer;
          transition: background-color 0.2s;
        }
        
        .suggestion-item:hover {
          background-color: #f8f9fa;
        }
        
        .suggestion-item.active {
          background-color: #e9ecef;
        }
        
        .suggestion-icon {
          width: 8px;
          height: 8px;
          border-radius: 50%;
          background-color: #e74c32;
          margin-right: 12px;
        }
        
        .suggestion-info {
          display: flex;
          flex-direction: column;
        }
        
        .suggestion-title {
          font-size: 16px;
          font-weight: 500;
        }
        
        .suggestion-category {
          font-size: 12px;
          color: #6c757d;
          text-transform: uppercase;
        }
        
        .suggestion-content {
          flex: 1;
          padding: 24px;
          overflow-y: auto;
          background-color: #f8f9fa;
        }
        
        .suggestion-details h2 {
          margin-top: 0;
          margin-bottom: 16px;
          font-size: 24px;
          font-weight: 600;
        }
        
        .suggestion-section {
          margin-bottom: 24px;
        }
        
        .suggestion-section h3 {
          font-size: 18px;
          font-weight: 500;
          margin-bottom: 8px;
        }
        
        .suggestion-section p {
          margin: 0;
          line-height: 1.5;
        }
        
        .code-container {
          background-color: #1e1e1e;
          border-radius: 8px;
          overflow: hidden;
          margin-top: 12px;
        }
        
        .code-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 8px 12px;
          background-color: #252525;
        }
        
        .generate-options {
          display: flex;
          align-items: center;
          gap: 12px;
        }
        
        .generate-button {
          padding: 6px 12px;
          background-color: #4a4a4a;
          color: white;
          border: none;
          border-radius: 4px;
          font-size: 14px;
          cursor: pointer;
          transition: background-color 0.2s;
        }
        
        .generate-button:hover:not(:disabled) {
          background-color: #5a5a5a;
        }
        
        .generate-button:disabled {
          opacity: 0.7;
          cursor: not-allowed;
        }
        
        .api-message {
          font-size: 12px;
          color: #aaa;
        }
        
        .code-actions {
          display: flex;
          justify-content: flex-end;
          padding: 8px 12px;
          background-color: #2d2d2d;
        }
        
        .code-actions button {
          padding: 6px 12px;
          background-color: #434547;
          color: white;
          border: none;
          border-radius: 4px;
          font-size: 12px;
          cursor: pointer;
          transition: background-color 0.2s;
        }
        
        .code-actions button:hover:not(:disabled) {
          background-color: #555;
        }
        
        .code-actions button:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }
        
        .suggestion-footer {
          margin-top: 32px;
          padding-top: 16px;
          border-top: 1px solid #e9ecef;
          display: flex;
          justify-content: flex-end;
        }
        
        .impact-indicator {
          display: flex;
          align-items: center;
          gap: 8px;
        }
        
        .impact-label {
          font-size: 14px;
          color: #6c757d;
        }
        
        .impact-value {
          font-size: 14px;
          font-weight: 500;
        }
        
        .loading-container {
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          height: 400px;
        }
        
        .spinner {
          width: 40px;
          height: 40px;
          border-radius: 50%;
          border: 3px solid #e9ecef;
          border-top-color: #e74c32;
          animation: spin 1s linear infinite;
          margin-bottom: 16px;
        }
        
        @keyframes spin {
          to { transform: rotate(360deg); }
        }
        
        .error-container {
          padding: 24px;
          text-align: center;
        }
        
        .error-container h3 {
          margin-top: 0;
          color: #e74c32;
        }
        
        .error-container button {
          padding: 8px 16px;
          background-color: #e74c32;
          color: white;
          border: none;
          border-radius: 4px;
          margin-top: 16px;
          cursor: pointer;
        }
        
        .no-suggestions {
          padding: 48px 24px;
          text-align: center;
          color: #6c757d;
        }
      `}</style>
    </div>
  );
};

export default ModelImprovementSuggestions;