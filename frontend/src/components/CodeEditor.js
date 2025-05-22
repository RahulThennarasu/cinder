import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vs } from 'react-syntax-highlighter/dist/esm/styles/prism';

// Debounce utility
const debounce = (func, wait) => {
  let timeout;
  return function executedFunction(...args) {
    const later = () => {
      clearTimeout(timeout);
      func(...args);
    };
    clearTimeout(timeout);
    timeout = setTimeout(later, wait);
  };
};

const CodeEditor = ({ modelInfo }) => {
  const [code, setCode] = useState('');
  const [originalCode, setOriginalCode] = useState('');
  const [isEditing, setIsEditing] = useState(false);
  const [hasChanges, setHasChanges] = useState(false);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(false);
  
  // AI analysis states
  const [bitSuggestions, setBitSuggestions] = useState([]);
  const [analyzing, setAnalyzing] = useState(false);
  const [showSuggestions, setShowSuggestions] = useState(true);
  const [selectedSuggestion, setSelectedSuggestion] = useState(null);
  const [autoFixing, setAutoFixing] = useState(false);
  
  const textareaRef = useRef(null);
  const suggestionsPanelRef = useRef(null);

  // Load model code when component mounts
  useEffect(() => {
    loadModelCode();
  }, []);

  // Check for changes
  useEffect(() => {
    setHasChanges(code !== originalCode);
  }, [code, originalCode]);

  // Auto-resize textarea and apply syntax highlighting
  useEffect(() => {
    if (isEditing && textareaRef.current) {
      const textarea = textareaRef.current;
      textarea.style.height = 'auto';
      textarea.style.height = textarea.scrollHeight + 'px';
    }
  }, [code, isEditing]);

  // Bit Code Analysis - the main feature
  const analyzeCode = useCallback(
    debounce(async (codeToAnalyze) => {
      if (!codeToAnalyze.trim() || codeToAnalyze.length < 50) return;
      
      try {
        setAnalyzing(true);
        
        // Create analysis context from model info
        const analysisContext = {
          code: codeToAnalyze,
          framework: modelInfo?.framework || 'unknown',
          modelMetrics: {
            accuracy: modelInfo?.accuracy || 0,
            precision: modelInfo?.precision || 0,
            recall: modelInfo?.recall || 0,
            f1: modelInfo?.f1 || 0,
            dataset_size: modelInfo?.dataset_size || 0
          },
          analysisType: 'ml_code_review'
        };

        const response = await fetch('http://localhost:8000/api/analyze-code', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(analysisContext)
        });

        if (response.ok) {
          const suggestions = await response.json();
          setBitSuggestions(suggestions.suggestions || []);
        } else {
          // Fallback to mock suggestions if API not available
          generateMockSuggestions(codeToAnalyze);
        }
      } catch (err) {
        console.error('Bit analysis failed:', err);
        generateMockSuggestions(codeToAnalyze);
      } finally {
        setAnalyzing(false);
      }
    }, 2000), // 2 second debounce
    [modelInfo]
  );

  // Generate mock suggestions based on code patterns and model metrics
  const generateMockSuggestions = (codeToAnalyze) => {
    const suggestions = [];
    const framework = modelInfo?.framework?.toLowerCase() || 'unknown';
    const accuracy = modelInfo?.accuracy || 0;
    
    // Pattern-based analysis
    const codeLines = codeToAnalyze.toLowerCase().split('\n');
    
    // Check for common ML issues
    if (accuracy < 0.8) {
      suggestions.push({
        type: 'performance',
        severity: 'high',
        line: findLineWithPattern(codeLines, ['model =', 'class ']),
        title: 'Low Model Accuracy Detected',
        message: `Your model accuracy is ${(accuracy * 100).toFixed(1)}%. Consider increasing model complexity or improving data quality.`,
        suggestion: framework === 'pytorch' 
          ? 'Try adding more layers: nn.Linear(hidden_size, hidden_size * 2)'
          : framework === 'tensorflow'
          ? 'Add more dense layers: tf.keras.layers.Dense(128, activation="relu")'
          : 'Try RandomForestClassifier with more estimators',
        autoFix: generateAutoFix('increase_complexity', framework)
      });
    }

    // Check for missing regularization
    if (!codeToAnalyze.includes('dropout') && !codeToAnalyze.includes('Dropout') && framework !== 'sklearn') {
      suggestions.push({
        type: 'overfitting',
        severity: 'medium',
        line: findLineWithPattern(codeLines, ['forward', 'model.add', 'sequential']),
        title: 'Missing Regularization',
        message: 'No dropout layers detected. This may lead to overfitting.',
        suggestion: framework === 'pytorch'
          ? 'Add: self.dropout = nn.Dropout(0.3)'
          : 'Add: tf.keras.layers.Dropout(0.3)',
        autoFix: generateAutoFix('add_dropout', framework)
      });
    }

    // Check for hardcoded learning rates
    if (codeToAnalyze.includes('lr=0.01') || codeToAnalyze.includes('learning_rate=0.01')) {
      suggestions.push({
        type: 'optimization',
        severity: 'low',
        line: findLineWithPattern(codeLines, ['optimizer', 'adam', 'sgd']),
        title: 'Hardcoded Learning Rate',
        message: 'Hardcoded learning rates may not be optimal for your specific problem.',
        suggestion: 'Use adaptive learning rate or scheduler',
        autoFix: generateAutoFix('adaptive_lr', framework)
      });
    }

    // Check for class imbalance handling
    if (!codeToAnalyze.includes('class_weight') && !codeToAnalyze.includes('WeightedRandomSampler')) {
      suggestions.push({
        type: 'data',
        severity: 'medium',
        line: findLineWithPattern(codeLines, ['fit(', 'train(', 'dataloader']),
        title: 'Class Imbalance Not Addressed',
        message: 'Consider handling class imbalance with weights or sampling techniques.',
        suggestion: 'Add class_weight="balanced" or use weighted sampling',
        autoFix: generateAutoFix('class_weights', framework)
      });
    }

    setBitSuggestions(suggestions);
  };

  const findLineWithPattern = (lines, patterns) => {
    for (let i = 0; i < lines.length; i++) {
      for (const pattern of patterns) {
        if (lines[i].includes(pattern)) {
          return i + 1;
        }
      }
    }
    return 1;
  };

  const generateAutoFix = (fixType, framework) => {
    const fixes = {
      increase_complexity: {
        pytorch: `# Add more layers for better capacity
self.layer3 = nn.Linear(hidden_size, hidden_size * 2)
self.layer4 = nn.Linear(hidden_size * 2, num_classes)`,
        tensorflow: `# Add more dense layers
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(64, activation='relu'))`,
        sklearn: `# Use more complex model
model = RandomForestClassifier(n_estimators=200, max_depth=15)`
      },
      add_dropout: {
        pytorch: `# Add dropout for regularization
self.dropout1 = nn.Dropout(0.3)
self.dropout2 = nn.Dropout(0.5)`,
        tensorflow: `# Add dropout layers
model.add(tf.keras.layers.Dropout(0.3))`,
        sklearn: '# Consider using different regularization parameters'
      },
      adaptive_lr: {
        pytorch: `# Use learning rate scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)`,
        tensorflow: `# Use adaptive learning rate
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
callbacks = [tf.keras.callbacks.ReduceLROnPlateau(patience=5)]`,
        sklearn: '# Sklearn optimizers handle learning rates automatically'
      },
      class_weights: {
        pytorch: `# Handle class imbalance with weighted sampling
from torch.utils.data import WeightedRandomSampler
sampler = WeightedRandomSampler(weights, len(weights))`,
        tensorflow: `# Use class weights
class_weights = {0: 1.0, 1: 2.0}  # Adjust based on your data
model.fit(X, y, class_weight=class_weights)`,
        sklearn: `# Use balanced class weights
model = RandomForestClassifier(class_weight='balanced')`
      }
    };

    return fixes[fixType]?.[framework] || '# Auto-fix not available for this framework';
  };

  // Trigger analysis when code changes
  useEffect(() => {
    if (isEditing && code.trim()) {
      analyzeCode(code);
    }
  }, [code, isEditing, analyzeCode]);

  const loadModelCode = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const response = await fetch('http://localhost:8000/api/model-code');
      
      if (response.ok) {
        const data = await response.json();
        setCode(data.code);
        setOriginalCode(data.code);
      } else {
        const templateCode = getTemplateCode();
        setCode(templateCode);
        setOriginalCode(templateCode);
      }
    } catch (err) {
      console.error('Error loading model code:', err);
      const templateCode = getTemplateCode();
      setCode(templateCode);
      setOriginalCode(templateCode);
    } finally {
      setLoading(false);
    }
  };

  const getTemplateCode = () => {
    const framework = modelInfo?.framework?.toLowerCase() || 'pytorch';
    
    if (framework === 'pytorch') {
      return `import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

class NeuralNetwork(nn.Module):
    def __init__(self, input_size=10, hidden_size=20, num_classes=2):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.relu(out)
        out = self.layer2(out)
        return F.log_softmax(out, dim=1)

def train_model(model, train_loader, num_epochs=10):
    """Train the model with the provided data."""
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    for epoch in range(num_epochs):
        total_loss = 0
        for inputs, labels in train_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}')`;
    } else if (framework === 'tensorflow') {
      return `import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

def create_model(input_shape, num_classes=2):
    """Create a TensorFlow/Keras model."""
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(input_shape,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax' if num_classes > 2 else 'sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy' if num_classes > 2 else 'binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model(model, X_train, y_train, X_val, y_val, epochs=20):
    """Train the model."""
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=32,
        validation_data=(X_val, y_val),
        verbose=1
    )
    return history`;
    } else {
      return `import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def create_model(model_type='random_forest'):
    """Create a scikit-learn model."""
    if model_type == 'random_forest':
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
    else:
        raise ValueError("Unknown model type")
    
    return model

def train_and_evaluate_model(model, X_train, X_test, y_train, y_test):
    """Train and evaluate the model."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.4f}')
    return accuracy`;
    }
  };

  const applyAutoFix = async (suggestion) => {
    if (!suggestion.autoFix) return;
    
    setAutoFixing(true);
    
    try {
      // Find the appropriate line to insert the fix
      const lines = code.split('\n');
      const insertLine = Math.max(0, (suggestion.line || 1) - 1);
      
      // Insert the auto-fix code
      const fixLines = suggestion.autoFix.split('\n');
      lines.splice(insertLine, 0, '', ...fixLines, '');
      
      const newCode = lines.join('\n');
      setCode(newCode);
      
      // Remove the suggestion after applying fix
      setBitSuggestions(prev => prev.filter(s => s !== suggestion));
      
    } catch (err) {
      console.error('Error applying auto-fix:', err);
    } finally {
      setAutoFixing(false);
    }
  };

  const saveCode = async () => {
    try {
      setSaving(true);
      setError(null);
      setSuccess(false);
      
      const response = await fetch('http://localhost:8000/api/model-code', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ code }),
      });
      
      if (response.ok) {
        setOriginalCode(code);
        setSuccess(true);
        setTimeout(() => setSuccess(false), 3000);
      } else {
        throw new Error('Failed to save code');
      }
    } catch (err) {
      console.error('Error saving code:', err);
      setError('Failed to save code. The save endpoint may not be implemented yet.');
    } finally {
      setSaving(false);
    }
  };

  const resetCode = () => {
    setCode(originalCode);
    setIsEditing(false);
    setBitSuggestions([]);
  };

  const downloadCode = () => {
    const extension = 'py';
    const filename = `model_code.${extension}`;
    
    const blob = new Blob([code], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const formatCode = () => {
    const formatted = code
      .split('\n')
      .map(line => line.trimRight())
      .join('\n')
      .replace(/\n\n\n+/g, '\n\n');
    
    setCode(formatted);
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Tab') {
      e.preventDefault();
      const start = e.target.selectionStart;
      const end = e.target.selectionEnd;
      
      if (e.shiftKey) {
        const lines = code.split('\n');
        const startLine = code.substring(0, start).split('\n').length - 1;
        const endLine = code.substring(0, end).split('\n').length - 1;
        
        for (let i = startLine; i <= endLine; i++) {
          if (lines[i].startsWith('    ')) {
            lines[i] = lines[i].substring(4);
          } else if (lines[i].startsWith('\t')) {
            lines[i] = lines[i].substring(1);
          }
        }
        
        const newCode = lines.join('\n');
        setCode(newCode);
        
        setTimeout(() => {
          const newStart = Math.max(0, start - 4);
          e.target.setSelectionRange(newStart, newStart);
        }, 0);
      } else {
        const beforeTab = code.substring(0, start);
        const afterTab = code.substring(end);
        const newCode = beforeTab + '    ' + afterTab;
        setCode(newCode);
        
        setTimeout(() => {
          e.target.setSelectionRange(start + 4, start + 4);
        }, 0);
      }
    }
  };

  const getSeverityColor = (severity) => {
    switch (severity) {
      case 'high': return '#e74c32';
      case 'medium': return '#e74c32';
      case 'low': return '#e74c32';
      default: return '#e74c32';
    }
  };

  const getSeverityIcon = (severity) => {
    switch (severity) {
      case 'high': return '●';
      case 'medium': return '●';
      case 'low': return '●';
      default: return '●';
    }
  };

  if (loading) {
    return (
      <div style={{ 
        padding: '2rem', 
        textAlign: 'center',
        backgroundColor: '#ffffff',
        color: '#333333',
        minHeight: '400px',
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'center',
        alignItems: 'center'
      }}>
        <div style={{ 
          width: '40px', 
          height: '40px', 
          border: '3px solid #f3f3f3', 
          borderTop: '3px solid #e74c32', 
          borderRadius: '50%', 
          animation: 'spin 1s linear infinite',
          margin: '0 auto 1rem'
        }}></div>
        <p>Loading model code...</p>
        <style>
          {`
            @keyframes spin {
              0% { transform: rotate(0deg); }
              100% { transform: rotate(360deg); }
            }
          `}
        </style>
      </div>
    );
  }

  return (
    <div style={{ backgroundColor: '#ffffff', color: '#333333', minHeight: '100vh', display: 'flex' }}>
      {/* Main Editor */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
        {/* Header */}
        <div style={{ 
          display: 'flex', 
          justifyContent: 'space-between', 
          alignItems: 'center', 
          padding: '1rem 1.5rem',
          backgroundColor: '#f8f8f8',
          borderBottom: '1px solid #e1e1e1',
          flexWrap: 'wrap',
          gap: '1rem'
        }}>
          <div>
            <h3 style={{ margin: 0, fontSize: '1.1rem', fontWeight: '600', color: '#333333' }}>
              {analyzing && (
                <span style={{ 
                  marginLeft: '1rem',
                  padding: '0.2rem 0.5rem',
                  backgroundColor: '#e74c32',
                  borderRadius: '0.25rem',
                  fontSize: '0.7rem',
                  fontWeight: '500',
                  color: 'white'
                }}>
                  Bit analyzing...
                </span>
              )}
            </h3>
            <p style={{ margin: '0.25rem 0 0 0', fontSize: '0.8rem', color: '#666666' }}>
              Real-time code analysis with Bit
              {modelInfo?.framework && (
                <span style={{ 
                  marginLeft: '0.5rem',
                  padding: '0.2rem 0.5rem',
                  backgroundColor: '#e74c32',
                  borderRadius: '0.25rem',
                  fontSize: '0.7rem',
                  fontWeight: '500',
                  color: 'white'
                }}>
                  {modelInfo.framework}
                </span>
              )}
            </p>
          </div>
          
          <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
            <button
              onClick={() => setShowSuggestions(!showSuggestions)}
              style={{
                padding: '0.5rem 1rem',
                backgroundColor: showSuggestions ? '#e74c32' : '#f3f4f6',
                color: showSuggestions ? 'white' : '#374151',
                border: 'none',
                borderRadius: '0.25rem',
                fontSize: '0.8rem',
                fontWeight: '500',
                cursor: 'pointer',
                transition: 'all 0.2s'
              }}
            >
              Bit ({bitSuggestions.length})
            </button>
            
            <button
              onClick={() => setIsEditing(!isEditing)}
              style={{
                padding: '0.5rem 1rem',
                backgroundColor: '#e74c32',
                color: 'white',
                border: 'none',
                borderRadius: '0.25rem',
                fontSize: '0.8rem',
                fontWeight: '500',
                cursor: 'pointer',
                transition: 'all 0.2s'
              }}
            >
              {isEditing ? 'View' : 'Edit'}
            </button>
            
            <button
              onClick={downloadCode}
              style={{
                padding: '0.5rem 1rem',
                backgroundColor: '#e74c32',
                color: 'white',
                border: 'none',
                borderRadius: '0.25rem',
                fontSize: '0.8rem',
                fontWeight: '500',
                cursor: 'pointer',
                transition: 'all 0.2s'
              }}
            >
              Download
            </button>
          </div>
        </div>

        {/* Status Messages */}
        {error && (
          <div style={{
            padding: '0.75rem 1.5rem',
            backgroundColor: '#fee',
            borderBottom: '1px solid #fcc',
            fontSize: '0.8rem',
            color: '#800'
          }}>
            {error}
          </div>
        )}

        {success && (
          <div style={{
            padding: '0.75rem 1.5rem',
            backgroundColor: '#efe',
            borderBottom: '1px solid #cfc',
            fontSize: '0.8rem',
            color: '#080'
          }}>
            Code saved successfully
          </div>
        )}

        {/* Changes indicator */}
        {hasChanges && isEditing && (
          <div style={{
            padding: '0.75rem 1.5rem',
            backgroundColor: '#fff8e1',
            borderBottom: '1px solid #ffe0b2',
            fontSize: '0.8rem',
            color: '#e65100',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center'
          }}>
            <span>You have unsaved changes</span>
            <div style={{ display: 'flex', gap: '0.5rem' }}>
              <button
                onClick={resetCode}
                style={{
                  padding: '0.25rem 0.75rem',
                  fontSize: '0.7rem',
                  backgroundColor: '#f5f5f5',
                  color: '#333333',
                  border: '1px solid #e1e1e1',
                  borderRadius: '0.25rem',
                  cursor: 'pointer'
                }}
              >
                Reset
              </button>
              <button
                onClick={saveCode}
                disabled={saving}
                style={{
                  padding: '0.25rem 0.75rem',
                  fontSize: '0.7rem',
                  backgroundColor: '#e74c32',
                  color: 'white',
                  border: 'none',
                  borderRadius: '0.25rem',
                  cursor: saving ? 'not-allowed' : 'pointer',
                  opacity: saving ? 0.7 : 1
                }}
              >
                {saving ? 'Saving...' : 'Save'}
              </button>
            </div>
          </div>
        )}

        {/* Status Bar */}
        <div style={{
          padding: '0.4rem 1.5rem',
          backgroundColor: '#e74c32',
          color: 'white',
          fontSize: '0.75rem',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center'
        }}>
          <span>model_code.py</span>
          <span>
            {code.split('\n').length} lines | {isEditing ? 'Editing' : 'Read-only'} | 
            {bitSuggestions.length > 0 && ` ${bitSuggestions.length} suggestions`}
          </span>
        </div>

        {/* Code Editor */}
        <div style={{ 
          height: 'calc(100vh - 200px)', 
          overflow: 'hidden', 
          position: 'relative',
          display: 'flex',
          flexDirection: 'column',
          flex: 1
        }}>
          {isEditing ? (
            <div style={{ display: 'flex', height: '100%', flex: 1 }}>
              {/* Line Numbers */}
              <div style={{
                width: '60px',
                backgroundColor: '#fafafa',
                borderRight: '1px solid #e1e1e1',
                padding: '1rem 0.5rem',
                fontSize: '0.8rem',
                color: '#999999',
                fontFamily: 'Consolas, "Courier New", monospace',
                lineHeight: '1.5',
                textAlign: 'right',
                userSelect: 'none',
                overflowY: 'auto'
              }}>
                {code.split('\n').map((_, index) => (
                  <div key={index + 1} style={{ height: '1.2em' }}>
                    {index + 1}
                  </div>
                ))}
              </div>
              
              {/* Code Editor with Syntax Highlighting Overlay */}
              <div style={{ position: 'relative', flex: 1, height: '100%', overflow: 'hidden' }}>
                {/* Syntax Highlighted Background */}
                <div 
                  ref={(el) => {
                    if (el && textareaRef.current) {
                      const syncScroll = () => {
                        el.scrollTop = textareaRef.current.scrollTop;
                        el.scrollLeft = textareaRef.current.scrollLeft;
                      };
                      textareaRef.current.addEventListener('scroll', syncScroll);
                    }
                  }}
                  style={{
                    position: 'absolute',
                    top: 0,
                    left: 0,
                    right: 0,
                    bottom: 0,
                    overflow: 'hidden',
                    pointerEvents: 'none',
                    zIndex: 1
                  }}
                >
                  <SyntaxHighlighter
                    language="python"
                    style={vs}
                    showLineNumbers={false}
                    customStyle={{
                      margin: 0,
                      padding: '1rem',
                      backgroundColor: 'transparent',
                      fontSize: '0.85rem',
                      lineHeight: '1.5',
                      fontFamily: 'Consolas, "Courier New", monospace',
                      minHeight: '100%'
                    }}
                    codeTagProps={{
                      style: {
                        fontFamily: 'Consolas, "Courier New", monospace',
                        backgroundColor: 'transparent'
                      }
                    }}
                  >
                    {code}
                  </SyntaxHighlighter>
                </div>
                
                {/* Editable Textarea */}
                <textarea
                  ref={textareaRef}
                  value={code}
                  onChange={(e) => setCode(e.target.value)}
                  onKeyDown={handleKeyDown}
                  style={{
                    position: 'absolute',
                    top: 0,
                    left: 0,
                    width: '100%',
                    height: '100%',
                    padding: '1rem',
                    border: 'none',
                    outline: 'none',
                    fontFamily: 'Consolas, "Courier New", monospace',
                    fontSize: '0.85rem',
                    lineHeight: '1.5',
                    backgroundColor: 'transparent',
                    color: 'transparent',
                    resize: 'none',
                    whiteSpace: 'pre',
                    tabSize: 4,
                    zIndex: 2,
                    caretColor: '#333333',
                    overflow: 'auto'
                  }}
                  spellCheck={false}
                  placeholder=""
                />
              </div>
            </div>
          ) : (
            // Read-only syntax highlighted view
            <div style={{ height: '100%', overflow: 'auto' }}>
              <SyntaxHighlighter
                language="python"
                style={vs}
                showLineNumbers={true}
                lineNumberStyle={{
                  minWidth: '3em',
                  paddingRight: '1em',
                  textAlign: 'right',
                  color: '#999999',
                  borderRight: '1px solid #e1e1e1',
                  marginRight: '1em',
                  userSelect: 'none'
                }}
                customStyle={{
                  margin: 0,
                  padding: '1rem',
                  backgroundColor: '#ffffff',
                  fontSize: '0.85rem',
                  lineHeight: '1.5',
                  fontFamily: 'Consolas, "Courier New", monospace',
                  minHeight: '100%'
                }}
                codeTagProps={{
                  style: {
                    fontFamily: 'Consolas, "Courier New", monospace'
                  }
                }}
              >
                {code}
              </SyntaxHighlighter>
            </div>
          )}
        </div>
      </div>

      {/* Bit Suggestions Panel */}
      {showSuggestions && (
        <div 
          ref={suggestionsPanelRef}
          style={{ 
            width: '400px', 
            backgroundColor: '#f8fafc', 
            borderLeft: '1px solid #e1e1e1',
            display: 'flex',
            flexDirection: 'column',
            maxHeight: '100vh',
            overflow: 'hidden'
          }}
        >
          {/* Suggestions Header */}
          <div style={{
            padding: '1rem',
            borderBottom: '1px solid #e1e1e1',
            backgroundColor: '#ffffff'
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <h4 style={{ margin: 0, fontSize: '1rem', fontWeight: '600' }}>
                Bit
              </h4>
              <button
                onClick={() => setShowSuggestions(false)}
                style={{
                  background: 'none',
                  border: 'none',
                  fontSize: '1.2rem',
                  cursor: 'pointer',
                  color: '#666666'
                }}
              >
                ×
              </button>
            </div>
            <p style={{ 
              margin: '0.5rem 0 0 0', 
              fontSize: '0.8rem', 
              color: '#666666' 
            }}>
              {analyzing ? 'Analyzing your code...' : 
               bitSuggestions.length > 0 ? `${bitSuggestions.length} suggestions found` : 
               'Write code to get suggestions'}
            </p>
          </div>

          {/* Model Performance Context */}
          {modelInfo && (
            <div style={{
              padding: '1rem',
              backgroundColor: '#ffffff',
              borderBottom: '1px solid #e1e1e1'
            }}>
              <h5 style={{ margin: '0 0 0.5rem 0', fontSize: '0.9rem', fontWeight: '600' }}>
                Current Model Performance
              </h5>
              <div style={{ fontSize: '0.8rem', color: '#666666' }}>
                <div>Accuracy: <strong>{((modelInfo.accuracy || 0) * 100).toFixed(1)}%</strong></div>
                {modelInfo.precision && <div>Precision: <strong>{(modelInfo.precision * 100).toFixed(1)}%</strong></div>}
                {modelInfo.recall && <div>Recall: <strong>{(modelInfo.recall * 100).toFixed(1)}%</strong></div>}
                <div>Framework: <strong>{modelInfo.framework || 'Unknown'}</strong></div>
              </div>
            </div>
          )}

          {/* Suggestions List */}
          <div style={{ flex: 1, overflow: 'auto', padding: '0.5rem' }}>
            {analyzing && (
              <div style={{
                padding: '2rem',
                textAlign: 'center',
                color: '#666666'
              }}>
                <div style={{
                  width: '30px',
                  height: '30px',
                  border: '3px solid #f3f3f3',
                  borderTop: '3px solid #e74c32',
                  borderRadius: '50%',
                  animation: 'spin 1s linear infinite',
                  margin: '0 auto 1rem'
                }}></div>
                <p>Bit is analyzing your code...</p>
              </div>
            )}

            {!analyzing && bitSuggestions.length === 0 && (
              <div style={{
                padding: '2rem',
                textAlign: 'center',
                color: '#666666'
              }}>
                <div style={{ fontSize: '2rem', marginBottom: '1rem' }}>B</div>
                <p>Start editing your code to receive suggestions from Bit for improving your ML model</p>
              </div>
            )}

            {bitSuggestions.map((suggestion, index) => (
              <div
                key={index}
                style={{
                  backgroundColor: '#ffffff',
                  border: '1px solid #e1e1e1',
                  borderRadius: '0.5rem',
                  margin: '0.5rem 0',
                  overflow: 'hidden',
                  boxShadow: '0 1px 3px rgba(0, 0, 0, 0.1)',
                  borderLeft: `4px solid ${getSeverityColor(suggestion.severity)}`
                }}
              >
                {/* Suggestion Header */}
                <div
                  style={{
                    padding: '0.75rem',
                    borderBottom: '1px solid #f1f1f1',
                    cursor: 'pointer',
                    backgroundColor: selectedSuggestion === index ? '#f8fafc' : '#ffffff'
                  }}
                  onClick={() => setSelectedSuggestion(selectedSuggestion === index ? null : index)}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                    <div style={{ flex: 1 }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.25rem' }}>
                        <span style={{ color: getSeverityColor(suggestion.severity) }}>
                          {getSeverityIcon(suggestion.severity)}
                        </span>
                        <span style={{
                          fontSize: '0.75rem',
                          fontWeight: '600',
                          color: getSeverityColor(suggestion.severity),
                          textTransform: 'uppercase'
                        }}>
                          {suggestion.type}
                        </span>
                        {suggestion.line && (
                          <span style={{
                            fontSize: '0.7rem',
                            color: '#666666',
                            backgroundColor: '#f1f1f1',
                            padding: '0.1rem 0.3rem',
                            borderRadius: '0.2rem'
                          }}>
                            Line {suggestion.line}
                          </span>
                        )}
                      </div>
                      <h6 style={{
                        margin: 0,
                        fontSize: '0.9rem',
                        fontWeight: '600',
                        color: '#333333'
                      }}>
                        {suggestion.title}
                      </h6>
                      <p style={{
                        margin: '0.25rem 0 0 0',
                        fontSize: '0.8rem',
                        color: '#666666',
                        lineHeight: 1.4
                      }}>
                        {suggestion.message}
                      </p>
                    </div>
                    <span style={{
                      fontSize: '1rem',
                      color: '#666666',
                      marginLeft: '0.5rem'
                    }}>
                      {selectedSuggestion === index ? '−' : '+'}
                    </span>
                  </div>
                </div>

                {/* Expanded Content */}
                {selectedSuggestion === index && (
                  <div style={{ padding: '0.75rem', backgroundColor: '#f8fafc' }}>
                    {/* Suggestion Details */}
                    <div style={{ marginBottom: '0.75rem' }}>
                      <h6 style={{
                        margin: '0 0 0.5rem 0',
                        fontSize: '0.8rem',
                        fontWeight: '600',
                        color: '#333333'
                      }}>
                        Recommended Solution
                      </h6>
                      <p style={{
                        margin: 0,
                        fontSize: '0.8rem',
                        color: '#555555',
                        lineHeight: 1.4
                      }}>
                        {suggestion.suggestion}
                      </p>
                    </div>

                    {/* Auto-fix Code */}
                    {suggestion.autoFix && (
                      <div style={{ marginBottom: '0.75rem' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
                          <h6 style={{
                            margin: 0,
                            fontSize: '0.8rem',
                            fontWeight: '600',
                            color: '#333333'
                          }}>
                            Auto-fix Code
                          </h6>
                          <button
                            onClick={() => applyAutoFix(suggestion)}
                            disabled={autoFixing}
                            style={{
                              padding: '0.25rem 0.5rem',
                              fontSize: '0.7rem',
                              backgroundColor: '#e74c32',
                              color: 'white',
                              border: 'none',
                              borderRadius: '0.25rem',
                              cursor: autoFixing ? 'not-allowed' : 'pointer',
                              opacity: autoFixing ? 0.7 : 1
                            }}
                          >
                            {autoFixing ? 'Applying...' : 'Apply Fix'}
                          </button>
                        </div>
                        <div style={{
                          backgroundColor: '#1e1e1e',
                          borderRadius: '0.25rem',
                          overflow: 'hidden'
                        }}>
                          <SyntaxHighlighter
                            language="python"
                            style={{
                              'hljs': {
                                display: 'block',
                                overflowX: 'auto',
                                padding: '0.5em',
                                color: '#abb2bf',
                                background: '#282c34'
                              },
                              'hljs-comment': { color: '#5c6370', fontStyle: 'italic' },
                              'hljs-keyword': { color: '#c678dd' },
                              'hljs-string': { color: '#98c379' },
                              'hljs-number': { color: '#d19a66' },
                              'hljs-function': { color: '#61dafb' }
                            }}
                            customStyle={{
                              margin: 0,
                              fontSize: '0.75rem',
                              backgroundColor: '#1e1e1e'
                            }}
                          >
                            {suggestion.autoFix}
                          </SyntaxHighlighter>
                        </div>
                      </div>
                    )}

                    {/* Actions */}
                    <div style={{ display: 'flex', gap: '0.5rem' }}>
                      <button
                        onClick={() => setSelectedSuggestion(null)}
                        style={{
                          padding: '0.25rem 0.5rem',
                          fontSize: '0.7rem',
                          backgroundColor: '#f3f4f6',
                          color: '#374151',
                          border: '1px solid #e5e7eb',
                          borderRadius: '0.25rem',
                          cursor: 'pointer'
                        }}
                      >
                        Dismiss
                      </button>
                      {suggestion.line && (
                        <button
                          onClick={() => {
                            // Scroll to line in editor
                            if (textareaRef.current) {
                              const lines = code.split('\n');
                              const targetLine = Math.max(0, suggestion.line - 1);
                              const scrollPos = targetLine * 20; // Approximate line height
                              textareaRef.current.scrollTop = scrollPos;
                              
                              // Focus and select the line
                              const lineStart = lines.slice(0, targetLine).join('\n').length + (targetLine > 0 ? 1 : 0);
                              const lineEnd = lineStart + lines[targetLine]?.length || 0;
                              textareaRef.current.focus();
                              textareaRef.current.setSelectionRange(lineStart, lineEnd);
                            }
                          }}
                          style={{
                            padding: '0.25rem 0.5rem',
                            fontSize: '0.7rem',
                            backgroundColor: '#e74c32',
                            color: 'white',
                            border: 'none',
                            borderRadius: '0.25rem',
                            cursor: 'pointer'
                          }}
                        >
                          Go to Line
                        </button>
                      )}
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>

          {/* Suggestions Footer */}
          <div style={{
            padding: '0.75rem',
            borderTop: '1px solid #e1e1e1',
            backgroundColor: '#ffffff',
            fontSize: '0.7rem',
            color: '#666666',
            textAlign: 'center'
          }}>
            Powered by Bit • Model-aware suggestions
          </div>
        </div>
      )}
    </div>
  );
};

export default CodeEditor;