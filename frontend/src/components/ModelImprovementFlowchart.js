import React, { useState, useEffect } from 'react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';

const ModelImprovementFlowchart = ({ modelInfo, errorAnalysis, confidenceAnalysis }) => {
  const [expandedNodes, setExpandedNodes] = useState({});
  const [generatingCode, setGeneratingCode] = useState({});
  const [generatedCode, setGeneratedCode] = useState({});
  const [copied, setCopied] = useState(null);

  // Add this framework detection function
  const getFramework = () => {
    return modelInfo && modelInfo.framework 
      ? modelInfo.framework.toLowerCase() 
      : 'sklearn';
  };

  // Store the framework for use in code generation
  const framework = getFramework();
  
  // Toggle node expansion
  const toggleNode = (nodeId) => {
    setExpandedNodes(prev => ({
      ...prev,
      [nodeId]: !prev[nodeId]
    }));
  };

  // Helper to determine if we have high bias (underfitting)
  const hasHighBias = () => {
    return modelInfo && modelInfo.accuracy < 0.8;
  };

  // Helper to determine if we have high variance (overfitting)
  const hasHighVariance = () => {
    // If we have training history, we could compare training vs validation accuracy
    // For now, we'll use a heuristic based on confidence analysis
    return confidenceAnalysis && 
           confidenceAnalysis.avg_correct_confidence > 0.9 && 
           confidenceAnalysis.avg_incorrect_confidence > 0.7;
  };

  // Helper to check for class imbalance
  const hasClassImbalance = () => {
    return errorAnalysis && 
           errorAnalysis.error_types && 
           errorAnalysis.error_types.some(type => 
             type.count > (errorAnalysis.error_count * 0.7));
  };

  // Determine what flowchart branch to highlight based on model issues
  const determineModelIssue = () => {
    if (hasHighBias()) return 'high_bias';
    if (hasHighVariance()) return 'high_variance';
    if (hasClassImbalance()) return 'class_imbalance';
    return 'optimize';
  };

  // Clear copied state after a delay
  useEffect(() => {
    if (copied) {
      const timer = setTimeout(() => {
        setCopied(null);
      }, 2000);
      return () => clearTimeout(timer);
    }
  }, [copied]);

  const activeIssue = determineModelIssue();

  // Function to generate code using Gemini API
  const generateCode = async (nodeId, targetFramework) => {
    if (generatingCode[nodeId]) return; // Prevent multiple simultaneous requests
    
    try {
      // Update generating state
      setGeneratingCode(prev => ({
        ...prev,
        [nodeId]: true
      }));
      
      // Map node IDs to categories for the API
      const categoryMap = {
        'start': 'model_evaluation',
        'high_bias': 'high_bias_detection',
        'increase_complexity': 'increase_model_complexity',
        'feature_engineering': 'feature_engineering',
        'high_variance': 'high_variance_detection',
        'add_regularization': 'regularization',
        'more_data': 'data_augmentation',
        'class_imbalance': 'class_imbalance_detection',
        'sampling_techniques': 'sampling_techniques',
        'class_weights': 'class_weights',
        'optimize': 'hyperparameter_tuning'
      };
      
      const category = categoryMap[nodeId] || nodeId;
      
      // Call the API to generate code
      const response = await fetch(
        `http://localhost:8000/api/generate-code-example?framework=${targetFramework}&category=${category}`
      );
      
      if (!response.ok) {
        throw new Error(`Server responded with status: ${response.status}`);
      }
      
      const data = await response.json();
      
      // Update the generated code state
      setGeneratedCode(prev => ({
        ...prev,
        [nodeId]: {
          ...prev[nodeId],
          [targetFramework]: data.code
        }
      }));
      
    } catch (err) {
      console.error(`Error generating code for ${nodeId}:`, err);
      // Set error code in the generated code state
      setGeneratedCode(prev => ({
        ...prev,
        [nodeId]: {
          ...prev[nodeId],
          [targetFramework]: `# Error generating code: ${err.message}\n# Please try again later.`
        }
      }));
    } finally {
      // Update generating state
      setGeneratingCode(prev => ({
        ...prev,
        [nodeId]: false
      }));
    }
  };

  // Function to copy code to clipboard
  const copyToClipboard = (text, nodeId) => {
    navigator.clipboard.writeText(text);
    setCopied(nodeId);
  };

  // Function to get code for a specific node and framework
  const getNodeCode = (nodeId) => {
    if (generatedCode[nodeId] && generatedCode[nodeId][framework]) {
      return generatedCode[nodeId][framework];
    }
    return "# Click the 'Generate with Bit' button to create customized code\n# for improving your model based on this suggestion.";
  };

  // Render a code box with generate button
  const renderCodeBox = (nodeId, title) => {
    const isGenerating = generatingCode[nodeId] || false;
    const hasGeneratedCode = generatedCode[nodeId] && generatedCode[nodeId][framework];
    const isCopied = copied === nodeId;
    const code = getNodeCode(nodeId);
    
    return (
      <div className="flowchart-code-container">
        <div className="flowchart-code-header">
          <div className="flowchart-code-title">{title || 'Code Example'}</div>
          <div className="flowchart-code-actions">
            <button 
              className="flowchart-generate-button"
              onClick={() => generateCode(nodeId, framework)}
              disabled={isGenerating}
            >
              <div className="bit-indicator">
                <div className="bit-offset">
                  <div className="offset-back"></div>
                  <div className="offset-front"></div>
                </div>
                <span className="bit-text">
                  {isGenerating 
                    ? 'Generating...' 
                    : hasGeneratedCode 
                      ? 'Regenerate' 
                      : 'Bit'}
                </span>
              </div>
            </button>
            
            {hasGeneratedCode && (
              <button
                className="flowchart-copy-button"
                onClick={() => copyToClipboard(code, nodeId)}
                disabled={isGenerating}
              >
                {isCopied ? 'Copied!' : 'Copy Code'}
              </button>
            )}
          </div>
        </div>
        <div className="flowchart-code-box">
          <SyntaxHighlighter 
           language="python"
            style={vscDarkPlus}
            showLineNumbers={true}
            customStyle={{ margin: 0 }}
          >
            {code}
          </SyntaxHighlighter>
        </div>
        {hasGeneratedCode && (
          <div className="flowchart-code-footer">
            <div className="flowchart-code-info">
              <span>Generated by Bit {framework}</span>
            </div>
          </div>
        )}
      </div>
    );
  };

  // Render a flowchart node
  const renderNode = (nodeId, title, content, includeCode = false, isActive = false) => {
    const isExpanded = expandedNodes[nodeId];
    
    return (
      <div className={`flowchart-node ${isActive ? 'flowchart-node-active' : ''}`}>
        <div 
          className="flowchart-node-header"
          onClick={() => toggleNode(nodeId)}
        >
          <div className="flowchart-node-title">{title}</div>
          <div className="flowchart-node-toggle">{isExpanded ? '−' : '+'}</div>
        </div>
        
        {isExpanded && (
          <div className="flowchart-node-content">
            <div className="flowchart-node-description">
              {content}
            </div>
            
            {includeCode && renderCodeBox(nodeId, `${title} - ${framework} Implementation`)}
          </div>
        )}
      </div>
    );
  };

  // Render the connector lines between nodes
  const renderConnector = (direction = 'vertical') => (
    <div className={`flowchart-connector ${direction}`}>
      <div className="flowchart-connector-line"></div>
      {direction === 'branch' && (
        <>
          <div className="flowchart-connector-line horizontal left"></div>
          <div className="flowchart-connector-line horizontal right"></div>
        </>
      )}
    </div>
  );

  return (
    <div className="flowchart-container">
      <h3 className="card-title">Model Improvement Flowchart</h3>
      
      {/* Root node */}
      {renderNode(
        'start',
        'Evaluate Model Performance',
        <div>
          <p>Start by evaluating your model's performance metrics:</p>
          <ul>
            <li>Accuracy: {modelInfo ? `${(modelInfo.accuracy * 100).toFixed(2)}%` : 'N/A'}</li>
            <li>Error Rate: {errorAnalysis ? `${(errorAnalysis.error_rate * 100).toFixed(2)}%` : 'N/A'}</li>
            <li>Bias/Variance Status: {activeIssue === 'high_bias' ? 'High Bias' : 
                                     activeIssue === 'high_variance' ? 'High Variance' : 'Balanced'}</li>
          </ul>
        </div>,
        true,
        true
      )}

      {renderConnector('branch')}
      
      <div className="flowchart-branches">
        {/* Left branch - High Bias */}
        <div className="flowchart-branch">
          {renderNode(
            'high_bias',
            'High Bias (Underfitting)',
            <div>
              <p>Your model seems to have high bias (underfitting). This means it's too simple to capture the underlying patterns.</p>
              <p><strong>Symptoms:</strong></p>
              <ul>
                <li>Low training and validation accuracy</li>
                <li>Similar performance on training and validation sets</li>
              </ul>
            </div>,
            false,
            activeIssue === 'high_bias'
          )}
          
          {renderConnector()}
          
          {renderNode(
            'increase_complexity',
            'Increase Model Complexity',
            <div>
              <p>Try increasing your model's complexity:</p>
              <ul>
                <li>Add more layers or neurons</li>
                <li>Use a more powerful model architecture</li>
                <li>Reduce regularization if it's too strong</li>
              </ul>
            </div>,
            true,
            activeIssue === 'high_bias'
          )}
          
          {renderConnector()}
          
          {renderNode(
            'feature_engineering',
            'Feature Engineering',
            <div>
              <p>Create better features to help your model learn:</p>
              <ul>
                <li>Add polynomial features</li>
                <li>Create interaction terms</li>
                <li>Apply domain-specific transformations</li>
                <li>Use automated feature engineering tools</li>
              </ul>
            </div>,
            true,
            activeIssue === 'high_bias'
          )}
        </div>
        
        {/* Middle branch - High Variance */}
        <div className="flowchart-branch">
          {renderNode(
            'high_variance',
            'High Variance (Overfitting)',
            <div>
              <p>Your model seems to have high variance (overfitting). This means it's too complex and capturing noise.</p>
              <p><strong>Symptoms:</strong></p>
              <ul>
                <li>High training accuracy but lower validation accuracy</li>
                <li>Large gap between training and validation performance</li>
              </ul>
            </div>,
            false,
            activeIssue === 'high_variance'
          )}
          
          {renderConnector()}
          
          {renderNode(
            'add_regularization',
            'Add Regularization',
            <div>
              <p>Apply regularization techniques to reduce overfitting:</p>
              <ul>
                <li>L1/L2 regularization</li>
                <li>Dropout</li>
                <li>Early stopping</li>
                <li>Batch normalization</li>
              </ul>
            </div>,
            true,
            activeIssue === 'high_variance'
          )}
          
          {renderConnector()}
          
          {renderNode(
            'more_data',
            'Get More Training Data',
            <div>
              <p>Increase your training data to help the model generalize:</p>
              <ul>
                <li>Collect more real data</li>
                <li>Use data augmentation techniques</li>
                <li>Generate synthetic samples</li>
                <li>Use transfer learning from related tasks</li>
              </ul>
            </div>,
            true,
            activeIssue === 'high_variance'
          )}
        </div>
        
        {/* Right branch - Class Imbalance */}
        <div className="flowchart-branch">
          {renderNode(
            'class_imbalance',
            'Class Imbalance',
            <div>
              <p>Your model may be affected by class imbalance. This means some classes have many more samples than others.</p>
              <p><strong>Symptoms:</strong></p>
              <ul>
                <li>Good accuracy but poor precision/recall for minority classes</li>
                <li>Most errors occur in underrepresented classes</li>
              </ul>
            </div>,
            false,
            activeIssue === 'class_imbalance'
          )}
          
          {renderConnector()}
          
          {renderNode(
            'sampling_techniques',
            'Sampling Techniques',
            <div>
              <p>Balance your dataset using sampling techniques:</p>
              <ul>
                <li>Oversampling minority classes</li>
                <li>Undersampling majority classes</li>
                <li>SMOTE (Synthetic Minority Over-sampling Technique)</li>
                <li>Hybrid approaches</li>
              </ul>
            </div>,
            true,
            activeIssue === 'class_imbalance'
          )}
          
          {renderConnector()}
          
          {renderNode(
            'class_weights',
            'Class Weights & Metrics',
            <div>
              <p>Adjust the learning process to handle imbalance:</p>
              <ul>
                <li>Use class weights in loss function</li>
                <li>Use F1-score or AUC-ROC instead of accuracy</li>
                <li>Adjust decision threshold</li>
                <li>Use focal loss for extreme imbalance</li>
              </ul>
            </div>,
            true,
            activeIssue === 'class_imbalance'
          )}
        </div>
      </div>
      
      {renderConnector()}
      
      {/* Common optimization node */}
      {renderNode(
        'optimize',
        'Fine-tune & Optimize',
        <div>
          <p>Once you've addressed the main issues, optimize your model:</p>
          <ul>
            <li>Tune hyperparameters systematically</li>
            <li>Try ensemble methods</li>
            <li>Implement cross-validation</li>
            <li>Consider advanced architectures</li>
          </ul>
        </div>,
        true,
        activeIssue === 'optimize'
      )}
      
      {/* Add CSS for the code generation buttons and container */}
      <style jsx>{`
        .flowchart-code-container {
          border: 1px solid #e5e7eb;
          border-radius: 8px;
          overflow: hidden;
          margin-top: 16px;
          background-color: #ffffff;
        }
        
        .flowchart-code-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 8px 12px;
          background-color: #f9fafb;
          border-bottom: 1px solid #e5e7eb;
        }
        
        .flowchart-code-title {
          font-size: 14px;
          font-weight: 500;
          color: #374151;
        }
        
        .flowchart-code-actions {
          display: flex;
          gap: 8px;
        }
        
        .flowchart-generate-button {
          background-color: #f9f8fb;
          color: white;
          border: none;
          padding: 6px 12px;
          border-radius: 4px;
          font-size: 13px;
          cursor: pointer;
          transition: background-color 0.2s;
        }
        
        .flowchart-generate-button:disabled {
          opacity: 0.7;
          cursor: not-allowed;
        }
        
        .flowchart-copy-button {
          background-color: #f3f4f6;
          color: #374151;
          border: 1px solid #e5e7eb;
          padding: 6px 12px;
          border-radius: 12px;
          font-size: 13px;
          cursor: pointer;
          transition: all 0.2s;
        }
        
        .flowchart-copy-button:hover:not(:disabled) {
          background-color: #e5e7eb;
        }
        
        .flowchart-copy-button:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }
        
        .flowchart-code-footer {
          padding: 8px 12px;
          font-size: 12px;
          color: #6b7280;
          background-color: #f9fafb;
          border-top: 1px solid #e5e7eb;
        }
        .bit-indicator {
          display: flex;
          align-items: center;
          gap: 6px;
        }

        .bit-offset {
          position: relative;
          width: 14px;
          height: 14px;
        }

        .offset-back {
          position: absolute;
          top: 2px;
          left: 2px;
          width: 10px;
          height: 10px;
          background-color: rgba(255, 255, 255, 0.7);
          border-radius: 2px;
        }

        .offset-front {
          position: absolute;
          top: 0;
          left: 0;
          width: 10px;
          height: 10px;
          background-color: white;
          border-radius: 2px;
        }

        .bit-text {
          font-weight: 500;
        }

        .flowchart-generate-button {
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 8px;
        }
      `}</style>
    </div>
  );
};

export default ModelImprovementFlowchart;