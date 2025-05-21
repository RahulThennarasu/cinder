import React, { useState, useEffect } from 'react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';

const ModelImprovementSuggestions = () => {
  const [suggestions, setSuggestions] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [expandedSuggestion, setExpandedSuggestion] = useState(null);
  const [selectedFramework, setSelectedFramework] = useState('pytorch');
  const [generatingCode, setGeneratingCode] = useState(false);
  
  useEffect(() => {
    const fetchSuggestions = async () => {
      try {
        setLoading(true);
        const response = await fetch('http://localhost:8000/api/model-improvements?detail_level=code');
        
        if (!response.ok) {
          throw new Error(`Server responded with status: ${response.status}`);
        }
        
        const data = await response.json();
        setSuggestions(data);
        setError(null);
      } catch (err) {
        console.error('Error fetching improvement suggestions:', err);
        setError('Could not load improvement suggestions. Check if the server is running.');
      } finally {
        setLoading(false);
      }
    };
    
    fetchSuggestions();
  }, []);
  
  const toggleSuggestion = (index) => {
    if (expandedSuggestion === index) {
      setExpandedSuggestion(null);
    } else {
      setExpandedSuggestion(index);
    }
  };

  const regenerateCode = async (suggestion, framework) => {
    try {
      setGeneratingCode(true);
      
      // Call the server to regenerate code
      const category = suggestion.title.toLowerCase().replace(/\s+/g, '_');
      const response = await fetch(`http://localhost:8000/api/generate-code-example?framework=${framework}&category=${category}`);
      
      if (!response.ok) {
        throw new Error(`Server responded with status: ${response.status}`);
      }
      
      const data = await response.json();
      
      // Update the suggestion with new code
      const updatedSuggestions = suggestions.suggestions.map((s, i) => {
        if (s === suggestion) {
          return {
            ...s,
            code_example: {
              ...s.code_example,
              [framework]: data.code
            }
          };
        }
        return s;
      });
      
      setSuggestions({
        ...suggestions,
        suggestions: updatedSuggestions
      });
      
    } catch (err) {
      console.error('Error regenerating code:', err);
      alert('Failed to regenerate code example. See console for details.');
    } finally {
      setGeneratingCode(false);
    }
  };
  
  if (loading) {
    return (
      <div className="loading-container">
        <div className="loading-spinner"></div>
        <p>Loading model improvement suggestions...</p>
      </div>
    );
  }
  
  if (error) {
    return (
      <div className="error-container">
        <h3>Error Loading Suggestions</h3>
        <p>{error}</p>
        <button 
          className="retry-button" 
          onClick={() => window.location.reload()}
        >
          Retry
        </button>
      </div>
    );
  }
  
  if (!suggestions || !suggestions.suggestions || suggestions.suggestions.length === 0) {
    return (
      <div className="empty-state">
        <h3>No Improvement Suggestions Available</h3>
        <p>Your model may already be well-optimized, or there's not enough data to generate suggestions.</p>
      </div>
    );
  }
  
  // Get available frameworks from code examples
  const getAvailableFrameworks = (suggestion) => {
    if (!suggestion.code_example) return ['pytorch']; // Default
    return Object.keys(suggestion.code_example);
  };
  
  // Helper to determine severity color
  const getSeverityColor = (severity) => {
    switch(severity.toLowerCase()) {
      case 'high': return '#e74c32';
      case 'medium': return '#f59e0b';
      case 'low': return '#10b981';
      default: return '#6b7280';
    }
  };
  
  return (
    <div className="suggestions-container">
      <div className="suggestions-header">
        <div className="suggestions-title-area">
          <h2 className="suggestions-title">Model Improvement Suggestions</h2>
          <p className="suggestions-subtitle">AI-generated recommendations to enhance your model's performance</p>
        </div>
        
        <div className="metrics-container">
          <div className="metric-box">
            <span className="metric-label">Current Accuracy</span>
            <span className="metric-value">{(suggestions.model_accuracy * 100).toFixed(1)}%</span>
          </div>
          
          <div className="metric-box">
            <span className="metric-label">Error Rate</span>
            <span className="metric-value">{(suggestions.error_rate * 100).toFixed(1)}%</span>
          </div>
          
          <div className="metric-box">
            <span className="metric-label">Improvement Potential</span>
            <span className={`metric-value potential-${suggestions.improvement_potential}`}>
              {suggestions.improvement_potential.charAt(0).toUpperCase() + suggestions.improvement_potential.slice(1)}
            </span>
          </div>
        </div>
      </div>
      
      <div className="framework-selector">
        <span className="framework-label">Code Examples For:</span>
        <div className="framework-buttons">
          {['pytorch', 'tensorflow', 'sklearn'].map(fw => (
            <button 
              key={fw}
              className={`framework-button ${selectedFramework === fw ? 'active' : ''}`}
              onClick={() => setSelectedFramework(fw)}
            >
              {fw === 'pytorch' ? 'PyTorch' : 
               fw === 'tensorflow' ? 'TensorFlow' : 
               'scikit-learn'}
            </button>
          ))}
        </div>
      </div>
      
      <div className="suggestions-list">
        {suggestions.suggestions.map((suggestion, index) => (
          <div 
            key={index}
            className={`suggestion-card ${expandedSuggestion === index ? 'expanded' : ''}`}
            style={{borderLeftColor: getSeverityColor(suggestion.severity)}}
          >
            <div 
              className="suggestion-header"
              onClick={() => toggleSuggestion(index)}
            >
              <div className="suggestion-title-area">
                <div className="suggestion-category">{suggestion.category.replace('_', ' ').toUpperCase()}</div>
                <div className="suggestion-title">{suggestion.title}</div>
              </div>
              
              <div className="suggestion-severity" 
                style={{backgroundColor: getSeverityColor(suggestion.severity) + '20', color: getSeverityColor(suggestion.severity)}}>
                {suggestion.severity}
              </div>
              
              <div className="suggestion-toggle">
                {expandedSuggestion === index ? '−' : '+'}
              </div>
            </div>
            
            {expandedSuggestion === index && (
              <div className="suggestion-content">
                <div className="suggestion-details">
                  <div className="detail-section">
                    <h4 className="detail-title">Issue</h4>
                    <p className="detail-text">{suggestion.issue}</p>
                  </div>
                  
                  <div className="detail-section">
                    <h4 className="detail-title">Recommendation</h4>
                    <p className="detail-text">{suggestion.suggestion}</p>
                  </div>
                  
                  <div className="detail-section">
                    <h4 className="detail-title">Expected Impact</h4>
                    <p className="detail-text">{suggestion.expected_impact}</p>
                  </div>
                </div>
                
                {suggestion.code_example && suggestion.code_example[selectedFramework] && (
                  <div className="code-section">
                    <div className="code-header">
                      <h4 className="code-title">Implementation Example</h4>
                      
                      <div className="code-actions">
                        <div className="framework-pills">
                          {getAvailableFrameworks(suggestion).map(fw => (
                            <button 
                              key={fw}
                              className={`framework-pill ${selectedFramework === fw ? 'active' : ''}`}
                              onClick={(e) => {
                                e.stopPropagation();
                                setSelectedFramework(fw);
                              }}
                            >
                              {fw === 'pytorch' ? 'PyTorch' : 
                              fw === 'tensorflow' ? 'TensorFlow' : 
                              'scikit-learn'}
                            </button>
                          ))}
                        </div>
                        
                        <button 
                          className="regenerate-button"
                          onClick={(e) => {
                            e.stopPropagation();
                            regenerateCode(suggestion, selectedFramework);
                          }}
                          disabled={generatingCode}
                        >
                          {generatingCode ? 'Generating...' : 'Regenerate Code'}
                        </button>
                      </div>
                    </div>
                    
                    <div className="code-container">
                      <SyntaxHighlighter 
                        language="python"
                        style={vscDarkPlus}
                        showLineNumbers={true}
                        customStyle={{ 
                          marginTop: '0.5rem', 
                          borderRadius: '6px',
                          fontSize: '14px'
                        }}
                      >
                        {suggestion.code_example[selectedFramework]}
                      </SyntaxHighlighter>
                      
                      <div className="code-footer">
                        <div className="ai-badge">
                          <span className="ai-badge-icon">✨</span>
                          <span className="ai-badge-text">Generated by Gemini AI</span>
                        </div>
                        
                        <button 
                          className="copy-button"
                          onClick={(e) => {
                            e.stopPropagation();
                            navigator.clipboard.writeText(suggestion.code_example[selectedFramework]);
                            
                            // Show feedback
                            const button = e.target;
                            const originalText = button.textContent;
                            button.textContent = 'Copied!';
                            setTimeout(() => {
                              button.textContent = originalText;
                            }, 2000);
                          }}
                        >
                          Copy Code
                        </button>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};

export default ModelImprovementSuggestions;