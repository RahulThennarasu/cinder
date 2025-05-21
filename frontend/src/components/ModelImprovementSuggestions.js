import React, { useState, useEffect } from 'react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { atomDark } from 'react-syntax-highlighter/dist/esm/styles/prism';
import './ModelImprovementSuggestions.css';

const ModelImprovementSuggestions = () => {
  const [suggestions, setSuggestions] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('pytorch');
  const [activeSection, setActiveSection] = useState(null);
  const [generatingCode, setGeneratingCode] = useState(false);
  const [generatedCode, setGeneratedCode] = useState({});
  const [copied, setCopied] = useState(false);
  
  useEffect(() => {
    const fetchSuggestions = async () => {
      try {
        setLoading(true);
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
        setError('Could not load improvement suggestions. Check if the server is running and that the API endpoint is accessible.');
      } finally {
        setLoading(false);
      }
    };
    
    fetchSuggestions();
  }, []);

  useEffect(() => {
    if (copied) {
      const timer = setTimeout(() => {
        setCopied(false);
      }, 2000);
      return () => clearTimeout(timer);
    }
  }, [copied]);
  
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
    return `# Generate code to see a ${activeTab} implementation for this improvement\n# Click the "Generate Code" button above to begin`;
  };

  const copyToClipboard = (text) => {
    navigator.clipboard.writeText(text);
    setCopied(true);
  };

  // Get impact level color in a more subtle way
  const getImpactColor = (impact) => {
    if (!impact) return '#6B7280'; // Default gray
    const level = impact.toLowerCase();
    if (level.includes('high')) return '#e74c32'; // Indigo for high
    if (level.includes('medium')) return '#e74c32'; // Blue for medium
    return '#e74c32'; // Emerald for low
  };

  // Format category for display
  const formatCategory = (category) => {
    if (!category) return '';
    return category
      .split('_')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ');
  };
  
  // Render loading state
  if (loading) {
    return (
      <div className="model-suggestion-loading">
        <div className="model-suggestion-spinner"></div>
        <p className="model-suggestion-loading-text">Loading model improvement suggestions...</p>
        <p className="model-suggestion-loading-subtext">Analyzing model performance and identifying optimization opportunities</p>
      </div>
    );
  }
  
  // Render error state
  if (error) {
    return (
      <div className="model-suggestion-error">
        <h3 className="model-suggestion-error-title">Unable to Load Suggestions</h3>
        <p className="model-suggestion-error-message">{error}</p>
        <div className="model-suggestion-error-help">
          <h4>Troubleshooting Steps:</h4>
          <ol>
            <li>Verify that the server is running on <code>http://localhost:8000</code></li>
            <li>Check that the API endpoint <code>/api/model-improvements</code> is available</li>
            <li>Confirm network connectivity between your frontend and backend</li>
            <li>Check server logs for potential errors or exceptions</li>
          </ol>
        </div>
        <button 
          className="model-suggestion-error-button"
          onClick={() => window.location.reload()}
        >
          Retry
        </button>
      </div>
    );
  }
  
  return (
    <div className={`model-suggestions-container model-suggestions-framework-${activeTab}`}>
      <div className="model-suggestions-header">
        <div className="model-suggestions-metrics">
          <div className="model-suggestions-metric">
            <span className="model-suggestions-metric-value">
              {suggestions?.model_accuracy ? (suggestions.model_accuracy * 100).toFixed(1) : 0}%
            </span>
            <span className="model-suggestions-metric-label">Model Accuracy</span>
          </div>
          <div className="model-suggestions-metric">
            <span className="model-suggestions-metric-value">
              {suggestions?.error_rate ? (suggestions.error_rate * 100).toFixed(1) : 0}%
            </span>
            <span className="model-suggestions-metric-label">Error Rate</span>
          </div>
          <div className="model-suggestions-metric">
            <span className="model-suggestions-metric-label">Improvement Potential</span>
            <div className="improvement-potential-badge">
              <span className="improvement-potential-indicator"></span>
              <code className="improvement-potential-value">
                {suggestions?.improvement_potential 
                  ? suggestions.improvement_potential.charAt(0).toUpperCase() + suggestions.improvement_potential.slice(1) 
                  : 'Unknown'}
              </code>
            </div>
          </div>
        </div>
        
        <div className="model-suggestions-framework-selector">
          <div className="model-suggestions-framework-label">Code Examples:</div>
          <div className="model-suggestions-framework-options">
            <button 
              className={`model-suggestions-framework-button ${activeTab === 'pytorch' ? 'active' : ''}`}
              onClick={() => setActiveTab('pytorch')}
            >
              PyTorch
            </button>
            <button 
              className={`model-suggestions-framework-button ${activeTab === 'tensorflow' ? 'active' : ''}`}
              onClick={() => setActiveTab('tensorflow')}
            >
              TensorFlow
            </button>
            <button 
              className={`model-suggestions-framework-button ${activeTab === 'sklearn' ? 'active' : ''}`}
              onClick={() => setActiveTab('sklearn')}
            >
              scikit-learn
            </button>
          </div>
        </div>
      </div>
      
      {!suggestions?.suggestions || suggestions.suggestions.length === 0 ? (
        <div className="model-suggestions-empty">
          <h3>No Improvement Suggestions Available</h3>
          <p>Your model appears to be performing well based on our analysis. There are no significant issues identified that require immediate attention.</p>
          <p>Consider exploring advanced techniques or fine-tuning hyperparameters if you want to optimize further.</p>
        </div>
      ) : (
        <div className="model-suggestions-content">
          <div className="model-suggestions-sidebar">
            <h3 className="model-suggestions-sidebar-title">Improvement Areas</h3>
            <div className="model-suggestions-sidebar-description">
              Click on a suggestion to see details and implementation examples
            </div>
            {suggestions.suggestions.map((suggestion, index) => (
              <div 
                key={index}
                className={`model-suggestions-sidebar-item ${activeSection === index ? 'active' : ''}`}
                onClick={() => setActiveSection(index)}
              >
                <div className="model-suggestions-sidebar-item-header">
                  <div 
                    className="model-suggestions-sidebar-item-icon"
                    style={{ 
                      backgroundColor: getImpactColor(suggestion.expected_impact)
                    }}
                  ></div>
                  <div className="model-suggestions-sidebar-item-title">
                    {suggestion.title}
                  </div>
                </div>
                <div className="model-suggestions-sidebar-item-category">
                  {formatCategory(suggestion.category)}
                </div>
              </div>
            ))}
          </div>
          
          <div className="model-suggestions-detail">
            {activeSection !== null && (
              <div className="model-suggestions-detail-content">
                <h2 className="model-suggestions-detail-title">
                  {suggestions.suggestions[activeSection].title}
                </h2>
                
                <div className="model-suggestions-detail-category">
                  {formatCategory(suggestions.suggestions[activeSection].category)}
                  
                  <span className={`model-suggestions-framework-badge ${activeTab}`}>
                    {activeTab}
                  </span>
                </div>
                
                <div className="model-suggestions-detail-section">
                  <h3 className="model-suggestions-detail-section-title">Issue Identified</h3>
                  <div className="model-suggestions-detail-section-content">
                    <p>{suggestions.suggestions[activeSection].issue}</p>
                    <div className="model-suggestions-detail-section-explanation">
                      <div className="model-suggestions-detail-section-explanation-icon"></div>
                      <div className="model-suggestions-detail-section-explanation-text">
                        This issue can affect your model's performance by introducing bias, reducing generalization, or limiting accuracy on certain samples.
                      </div>
                    </div>
                  </div>
                </div>
                
                <div className="model-suggestions-detail-section">
                  <h3 className="model-suggestions-detail-section-title">Recommended Solution</h3>
                  <div className="model-suggestions-detail-section-content">
                    <p>{suggestions.suggestions[activeSection].suggestion}</p>
                    
                    <div className="model-suggestions-algorithm-metrics">
                      <div className="model-suggestions-algorithm-metric">
                        <span className="model-suggestions-algorithm-metric-label">Solution Complexity</span>
                        <div className="model-suggestions-complexity">
                          <div className="model-suggestions-complexity-bars">
                            {[...Array(3)].map((_, i) => (
                              <div 
                                key={i} 
                                className={`model-suggestions-complexity-bar ${i < 2 ? 'active' : ''}`}
                              ></div>
                            ))}
                          </div>
                          <span className="model-suggestions-complexity-label">Medium</span>
                        </div>
                      </div>
                    </div>
                    
                    <div className="model-suggestions-detail-section-impact">
                      <div className="model-suggestions-detail-section-impact-label">Expected Impact:</div>
                      <div 
                        className="model-suggestions-detail-section-impact-value"
                        style={{ color: getImpactColor(suggestions.suggestions[activeSection].expected_impact) }}
                      >
                        {suggestions.suggestions[activeSection].expected_impact || 'Medium'}
                      </div>
                    </div>
                  </div>
                </div>
                
                <div className="model-suggestions-detail-section">
                  <h3 className="model-suggestions-detail-section-title">Implementation</h3>
                  <div className="model-suggestions-detail-section-content">
                    <div className="model-suggestions-code-container">
                      <div className="model-suggestions-code-header">
                        <div className="model-suggestions-code-header-left">
                          <button 
                            onClick={() => generateCode(activeSection, activeTab)}
                            disabled={generatingCode}
                            className="model-suggestions-generate-button"
                          >
                            {generatingCode 
                              ? 'Generating Code...' 
                              : generatedCode[activeSection] && generatedCode[activeSection][activeTab]
                                ? 'Regenerate Code'
                                : 'Generate Code'
                            }
                          </button>
                          <div className="model-suggestions-code-framework">
                            <span className="model-suggestions-code-framework-label">Framework:</span>
                            <span className="model-suggestions-code-framework-value">{activeTab}</span>
                          </div>
                        </div>
                        {generatingCode && (
                          <div className="model-suggestions-api-message">
                            Using Gemini API to create optimized implementation...
                          </div>
                        )}
                      </div>
                      
                      <div className="model-suggestions-code-content">
                        <SyntaxHighlighter 
                          language="python"
                          style={atomDark}
                          showLineNumbers={true}
                          customStyle={{ 
                            borderRadius: '4px',
                            margin: 0,
                            padding: '16px',
                            fontSize: '14px',
                            backgroundColor: '#1e1e1e'
                          }}
                        >
                          {getCurrentCode()}
                        </SyntaxHighlighter>
                      </div>
                      
                      <div className="model-suggestions-code-footer">
                        <div className="model-suggestions-code-info">
                          {generatedCode[activeSection] && generatedCode[activeSection][activeTab] && (
                            <span>AI-generated implementation for {activeTab}</span>
                          )}
                        </div>
                        <button
                          onClick={() => copyToClipboard(getCurrentCode())}
                          disabled={!generatedCode[activeSection] || !generatedCode[activeSection][activeTab]}
                          className="model-suggestions-copy-button"
                        >
                          {copied ? 'Copied!' : 'Copy Code'}
                        </button>
                      </div>
                    </div>
                    
                    <div className="model-suggestions-implementation-notes">
                      <h4 className="model-suggestions-implementation-notes-title">Implementation Notes</h4>
                      <ul className="model-suggestions-implementation-notes-list">
                        <li>The code above provides a starting point for addressing the identified issue.</li>
                        <li>Adapt the parameters and hyperparameters to match your specific model architecture.</li>
                        <li>Consider experimenting with different variations of the solution to find optimal results.</li>
                        <li>Always validate the improvement with proper evaluation metrics.</li>
                      </ul>
                    </div>
                    
                    {generatedCode[activeSection] && generatedCode[activeSection][activeTab] && (
                      <div className="model-suggestions-code-annotation">
                        <div className="model-suggestions-code-annotation-title">Key Implementation Details</div>
                        <div className="model-suggestions-code-annotation-content">
                          Pay special attention to the <code>hyperparameters</code> and <code>class weights</code> in the implementation.
                          These values may need adjustment based on your specific dataset characteristics.
                        </div>
                      </div>
                    )}
                  </div>
                </div>
                
                <div className="model-suggestions-detail-footer">
                  <div className="model-suggestions-detail-footer-severity">
                    <span className="model-suggestions-detail-footer-severity-label">Severity:</span>
                    <span className="model-suggestions-detail-footer-severity-value">
                      {suggestions.suggestions[activeSection].severity || 'Medium'}
                    </span>
                  </div>
                  <div className="model-suggestions-detail-footer-next">
                    {activeSection < suggestions.suggestions.length - 1 && (
                      <button 
                        className="model-suggestions-next-button"
                        onClick={() => setActiveSection(activeSection + 1)}
                      >
                        Next Suggestion →
                      </button>
                    )}
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default ModelImprovementSuggestions;