import React, { useState, useEffect } from 'react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vs } from 'react-syntax-highlighter/dist/esm/styles/prism';

const CodeEditor = ({ modelInfo }) => {
  const [code, setCode] = useState('');
  const [originalCode, setOriginalCode] = useState('');
  const [isEditing, setIsEditing] = useState(false);
  const [hasChanges, setHasChanges] = useState(false);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(false);

  // Load model code when component mounts
  useEffect(() => {
    loadModelCode();
  }, []);

  // Check for changes
  useEffect(() => {
    setHasChanges(code !== originalCode);
  }, [code, originalCode]);

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
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.3),
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
    elif model_type == 'logistic_regression':
        model = LogisticRegression(
            random_state=42,
            max_iter=1000
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
    // Simple code formatting
    const formatted = code
      .split('\n')
      .map(line => line.trimRight())
      .join('\n')
      .replace(/\n\n\n+/g, '\n\n');
    
    setCode(formatted);
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
    <div style={{ backgroundColor: '#ffffff', color: '#333333', minHeight: '100vh' }}>
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
            Model Code Editor
          </h3>
          <p style={{ margin: '0.25rem 0 0 0', fontSize: '0.8rem', color: '#666666' }}>
            View and edit your model's source code
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
            {isEditing ? 'View Mode' : 'Edit Mode'}
          </button>
          
          <button
            onClick={formatCode}
            disabled={!isEditing}
            style={{
              padding: '0.5rem 1rem',
              backgroundColor: isEditing ? '#e74c32' : '#cccccc',
              color: isEditing ? 'white' : '#666666',
              border: 'none',
              borderRadius: '0.25rem',
              fontSize: '0.8rem',
              fontWeight: '500',
              cursor: isEditing ? 'pointer' : 'not-allowed',
              transition: 'all 0.2s'
            }}
          >
            Format
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
          backgroundColor: '#ffe6e6',
          borderBottom: '1px solid #ffcccc',
          fontSize: '0.8rem',
          color: '#cc0000'
        }}>
          {error}
        </div>
      )}

      {success && (
        <div style={{
          padding: '0.75rem 1.5rem',
          backgroundColor: '#e6ffe6',
          borderBottom: '1px solid #ccffcc',
          fontSize: '0.8rem',
          color: '#008000'
        }}>
          Code saved successfully!
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
        <span>{code.split('\n').length} lines | {isEditing ? 'Editing' : 'Read-only'}</span>
      </div>

      {/* Code Editor */}
      <div style={{ height: 'calc(100vh - 200px)', overflow: 'auto' }}>
        {isEditing ? (
          // Simple textarea for editing
          <div style={{ display: 'flex', height: '100%' }}>
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
              height: 'fit-content',
              minHeight: '100%'
            }}>
              {code.split('\n').map((_, index) => (
                <div key={index + 1} style={{ height: '1.2em' }}>
                  {index + 1}
                </div>
              ))}
            </div>
            
            {/* Code Input */}
            <textarea
              value={code}
              onChange={(e) => setCode(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Tab') {
                  e.preventDefault();
                  const start = e.target.selectionStart;
                  const end = e.target.selectionEnd;
                  const newValue = code.substring(0, start) + '    ' + code.substring(end);
                  setCode(newValue);
                  setTimeout(() => {
                    e.target.selectionStart = e.target.selectionEnd = start + 4;
                  }, 0);
                }
              }}
              style={{
                flex: 1,
                padding: '1rem',
                border: 'none',
                outline: 'none',
                fontFamily: 'Consolas, "Courier New", monospace',
                fontSize: '0.85rem',
                lineHeight: '1.5',
                backgroundColor: '#ffffff',
                color: '#333333',
                resize: 'none',
                whiteSpace: 'pre',
                tabSize: 4,
                minHeight: '100%'
              }}
              spellCheck={false}
            />
          </div>
        ) : (
          // Read-only syntax highlighted view
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
              height: '100%'
            }}
            codeTagProps={{
              style: {
                fontFamily: 'Consolas, "Courier New", monospace'
              }
            }}
          >
            {code}
          </SyntaxHighlighter>
        )}
      </div>
    </div>
  );
};

export default CodeEditor;