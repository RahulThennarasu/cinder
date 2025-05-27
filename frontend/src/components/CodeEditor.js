import React, { useState, useEffect, useRef, useCallback } from "react";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { vs } from "react-syntax-highlighter/dist/esm/styles/prism";
import { vscDarkPlus } from "react-syntax-highlighter/dist/esm/styles/prism";

const CodeEditor = ({ modelInfo }) => {
  const [code, setCode] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Analysis states
  const [analyzing, setAnalyzing] = useState(false);
  const [suggestions, setSuggestions] = useState([]);
  const [showSuggestions, setShowSuggestions] = useState(true);
  const [selectedSuggestion, setSelectedSuggestion] = useState(null);
  const [copiedCode, setCopiedCode] = useState(null);

  // Add state for panel width
  const [isResizing, setIsResizing] = useState(false);
  const [startX, setStartX] = useState(0);
  const [panelWidth, setPanelWidth] = useState(500); // Dramatically increased to 1200px

  // Add these state variables
  const [originalCode, setOriginalCode] = useState(""); // Store original code for undo
  const [changeHistory, setChangeHistory] = useState([]); // Track changes for undo
  const [lastChanges, setLastChanges] = useState({ added: [], removed: [] }); // Track last changes
  const [showDiff, setShowDiff] = useState(false); // Toggle diff view

  const [searchText, setSearchText] = useState("");
  const [searchResults, setSearchResults] = useState([]);
  const [searchIndex, setSearchIndex] = useState(0);
  const [showSearch, setShowSearch] = useState(false);

  // Reference to the container
  const containerRef = useRef(null);
  const codeEditorRef = useRef(null);


  useEffect(() => {
    // Load model code when component mounts
    loadModelCode();
  }, [modelInfo]);

  // Reset copied state after a delay
  useEffect(() => {
    if (copiedCode !== null) {
      const timer = setTimeout(() => {
        setCopiedCode(null);
      }, 2000);
      return () => clearTimeout(timer);
    }
  }, [copiedCode]);

  // Add this useEffect to log the panel width whenever it changes
  useEffect(() => {
    console.log("Panel width changed to:", panelWidth);
  }, [panelWidth]);

  const startResize = (mouseDownEvent) => {
    mouseDownEvent.preventDefault();

    setIsResizing(true);
    // Get the initial mouse position
    setStartX(mouseDownEvent.clientX);
  };
  

  const cinderTheme = {
  ...vscDarkPlus,
  'pre[class*="language-"]': {
    ...vscDarkPlus['pre[class*="language-"]'],
    background: '#1E1E1E',
    padding: '1.5rem',
    margin: '0',
    overflow: 'auto',
    borderRadius: '0',
  },
  'code[class*="language-"]': {
    ...vscDarkPlus['code[class*="language-"]'],
    fontFamily: "'JetBrains Mono', 'Consolas', 'Monaco', monospace",
    fontSize: '14px',
    lineHeight: '1.5',
  },
  // Customize colors for specific tokens
  comment: {
    ...vscDarkPlus.comment,
    color: '#6A9955'
  },
  string: {
    ...vscDarkPlus.string,
    color: '#FF9B45' // Cinder primary-light
  },
  keyword: {
    ...vscDarkPlus.keyword,
    color: '#D5451B' // Cinder primary
  },
  function: {
    ...vscDarkPlus.function,
    color: '#DCDCAA'
  },
  boolean: {
    ...vscDarkPlus.boolean,
    color: '#D5451B' // Cinder primary
  },
  number: {
    ...vscDarkPlus.number,
    color: '#FF9B45' // Cinder primary-light
  },
  operator: {
    ...vscDarkPlus.operator,
    color: '#D4D4D4'
  },
  punctuation: {
    ...vscDarkPlus.punctuation,
    color: '#D4D4D4'
  },
  property: {
    ...vscDarkPlus.property,
    color: '#9CDCFE'
  },
  'class-name': {
    ...vscDarkPlus['class-name'],
    color: '#4EC9B0'
  },
  variable: {
    ...vscDarkPlus.variable,
    color: '#9CDCFE'
  },
};

  const scrollToLine = (lineNumber) => {
  if (!codeEditorRef.current) return;
  
  // Get all line elements in the editor
  const lineElements = codeEditorRef.current.querySelectorAll(
    '.react-syntax-highlighter-line-number'
  );
  
  // If we found the line element, scroll to it
  if (lineElements && lineElements[lineNumber]) {
    lineElements[lineNumber].scrollIntoView({
      behavior: 'smooth',
      block: 'center'
    });
  }
};
useEffect(() => {
  const handleKeyDown = (e) => {
    if (e.ctrlKey && e.key === 'f') {
      e.preventDefault();
      setShowSearch(true);
    }
  };
  
  document.addEventListener('keydown', handleKeyDown);
  return () => document.removeEventListener('keydown', handleKeyDown);
}, []);

// Add search function
const searchInCode = () => {
  if (!searchText) return;
  
  const codeLines = code.split('\n');
  const results = [];
  
  codeLines.forEach((line, lineIndex) => {
    let index = line.toLowerCase().indexOf(searchText.toLowerCase());
    while (index !== -1) {
      results.push({ lineIndex, charIndex: index });
      index = line.toLowerCase().indexOf(searchText.toLowerCase(), index + 1);
    }
  });
  
  setSearchResults(results);
  setSearchIndex(0);
  
  if (results.length > 0) {
    scrollToLine(results[0].lineIndex);
  }
};

// Function to navigate to the next search result
const findNext = () => {
  if (searchResults.length === 0) return;
  
  const nextIndex = (searchIndex + 1) % searchResults.length;
  setSearchIndex(nextIndex);
  scrollToLine(searchResults[nextIndex].lineIndex);
};

// Function to navigate to the previous search result
const findPrev = () => {
  if (searchResults.length === 0) return;
  
  const prevIndex = (searchIndex - 1 + searchResults.length) % searchResults.length;
  setSearchIndex(prevIndex);
  scrollToLine(searchResults[prevIndex].lineIndex);
};


  const handleResize = (e) => {
    if (!isResizing) return;

    console.log("Resizing panel...");

    // For a right-side panel, calculate width from right edge of container to mouse position
    const containerRect = containerRef.current.getBoundingClientRect();

    // When dragging left, width should increase
    const newWidth = containerRect.right - e.clientX;
    console.log("New width calculated:", newWidth);

    // Apply constraints - increased max width to 1800px
    const constrainedWidth = Math.max(400, Math.min(1800, newWidth));
    console.log("Constrained width:", constrainedWidth);

    setPanelWidth(constrainedWidth);
  };

  const stopResize = () => {
    setIsResizing(false);
  };

  useEffect(() => {
    if (isResizing) {
      document.addEventListener("mousemove", handleResize);
      document.addEventListener("mouseup", stopResize);
      document.body.style.userSelect = "none";
    } else {
      document.removeEventListener("mousemove", handleResize);
      document.removeEventListener("mouseup", stopResize);
      document.body.style.userSelect = "";
    }

    return () => {
      document.removeEventListener("mousemove", handleResize);
      document.removeEventListener("mouseup", stopResize);
      document.body.style.userSelect = "";
    };
  }, [isResizing, startX, panelWidth]); // Include dependencies used in handleResize

  // Load the model code (read-only)
  const loadModelCode = async () => {
    try {
      setLoading(true);
      const response = await fetch("http://localhost:8000/api/model-code");

      if (response.ok) {
        const data = await response.json();
        const loadedCode = data.code || "# No model code available";
        setCode(loadedCode);
        setOriginalCode(loadedCode); // Save original code
      } else {
        // For demo purposes, load sample code if API fails
        const sampleCode = getSampleModelCode(
          modelInfo?.framework || "pytorch",
        );
        setCode(sampleCode);
        setOriginalCode(sampleCode); // Save original code
      }
    } catch (err) {
      console.error("Error loading model code:", err);
      const sampleCode = getSampleModelCode(modelInfo?.framework || "pytorch");
      setCode(sampleCode);
      setOriginalCode(sampleCode); // Save original code
    } finally {
      setLoading(false);
    }
  };

  // Find the best location to insert code
  const findInsertLocation = (suggestionLine, suggestionType) => {
    const codeLines = code.split("\n");

    // Default to the suggestion line
    let insertLine = Math.min(suggestionLine, codeLines.length);

    // For class-related changes, try to find the class definition
    if (suggestionType === "overfitting" && code.includes("class ")) {
      // Find the class definition
      const classDefMatch = code.match(/class\s+\w+\([^)]*\):/);
      if (classDefMatch) {
        const classDefIndex = code.indexOf(classDefMatch[0]);
        const linesBeforeClass =
          code.substring(0, classDefIndex).split("\n").length - 1;

        // Find the __init__ method
        const initMethodMatch = code.match(/def\s+__init__\s*\([^)]*\):/);
        if (initMethodMatch) {
          const initMethodIndex = code.indexOf(initMethodMatch[0]);
          const linesBeforeInit =
            code.substring(0, initMethodIndex).split("\n").length - 1;

          // Insert after the first line of __init__ method
          insertLine = linesBeforeInit + 1;
        } else {
          // Insert after the class definition
          insertLine = linesBeforeClass + 1;
        }
      }
    }

    // For forward method changes
    if (suggestionType === "performance" && code.includes("def forward(")) {
      const forwardMethodMatch = code.match(/def\s+forward\s*\([^)]*\):/);
      if (forwardMethodMatch) {
        const forwardMethodIndex = code.indexOf(forwardMethodMatch[0]);
        const linesBeforeForward =
          code.substring(0, forwardMethodIndex).split("\n").length - 1;

        // Insert after the first line of forward method
        insertLine = linesBeforeForward + 1;
      }
    }

    return insertLine;
  };

  // Helper: Find appropriate code sections to replace
const findReplacementBoundaries = (suggestionType, suggestionLine) => {
  const codeLines = code.split("\n");
  
  // Default: Just insert at the line without replacement
  let startLine = Math.min(suggestionLine, codeLines.length);
  let endLine = startLine;
  
  // For different suggestion types, find appropriate code sections to replace
  switch (suggestionType.toLowerCase()) {
    case "overfitting":
      // Look for existing dropout layers or regularization code
      return findMethodOrBlockBoundaries(codeLines, startLine, ["dropout", "regularization", "weight_decay"]);
    
    case "performance":
      // For performance issues, look for existing optimization code
      return findMethodOrBlockBoundaries(codeLines, startLine, ["optimizer", "learning_rate", "scheduler"]);
    
    case "data":
      // For data preprocessing issues
      return findMethodOrBlockBoundaries(codeLines, startLine, ["preprocess", "transform", "normalize", "sampler"]);
      
    case "optimization":
      // For learning rate or optimizer suggestions
      return findMethodOrBlockBoundaries(codeLines, startLine, ["optimizer", "adam", "sgd", "learning_rate"]);
      
    case "model_capacity":
      // For model architecture improvements
      if (code.includes("class ")) {
        // If it's inside a class definition, try to find the relevant method
        const methodBoundaries = findMethodBoundaries(codeLines, "forward");
        if (methodBoundaries.startLine > 0) {
          return methodBoundaries;
        }
      }
      return findMethodOrBlockBoundaries(codeLines, startLine, ["layer", "linear", "conv", "dense"]);
      
    default:
      // Just use the insertion point without replacement
      return { startLine, endLine };
  }
};

// Helper: Find method boundaries by name
const findMethodBoundaries = (codeLines, methodName) => {
  const methodPattern = new RegExp(`\\s*def\\s+${methodName}\\s*\\(`);
  let startLine = -1;
  
  // Find the method definition
  for (let i = 0; i < codeLines.length; i++) {
    if (methodPattern.test(codeLines[i])) {
      startLine = i;
      break;
    }
  }
  
  if (startLine === -1) {
    return { startLine: 0, endLine: 0 }; // Method not found
  }
  
  // Find the end of the method (next method or end of indentation)
  const indentMatch = codeLines[startLine].match(/^(\s*)/);
  const baseIndent = indentMatch ? indentMatch[1].length : 0;
  
  let endLine = startLine + 1;
  while (endLine < codeLines.length) {
    // Check if we've reached a line with same or less indentation (excluding empty lines)
    const line = codeLines[endLine];
    if (line.trim() !== '' && !line.startsWith(' '.repeat(baseIndent + 4))) {
      break;
    }
    endLine++;
  }
  
  return { startLine, endLine };
};

// Helper: Find boundaries for a code block containing any of the keywords
const findMethodOrBlockBoundaries = (codeLines, nearLine, keywords) => {
  // First, check nearby lines for any of the keywords
  const searchRadius = 10; // Look 10 lines before and after
  const startSearch = Math.max(0, nearLine - searchRadius);
  const endSearch = Math.min(codeLines.length, nearLine + searchRadius);
  
  let keywordLine = -1;
  
  // Find the first line with a keyword
  for (let i = startSearch; i < endSearch; i++) {
    const line = codeLines[i].toLowerCase();
    if (keywords.some(keyword => line.includes(keyword.toLowerCase()))) {
      keywordLine = i;
      break;
    }
  }
  
  // If we found a keyword line, find the boundaries of its block
  if (keywordLine >= 0) {
    // Find the start of the statement (could be a multi-line statement)
    let startLine = keywordLine;
    while (startLine > 0) {
      // Go back until we find a line that doesn't end with a continuation character
      // or has less indentation
      const prevLine = codeLines[startLine - 1];
      if (!prevLine.trim().endsWith('\\') && 
          (prevLine.trim() === '' || getIndentation(prevLine) < getIndentation(codeLines[keywordLine]))) {
        break;
      }
      startLine--;
    }
    
    // Find the end of the statement or block
    let endLine = keywordLine;
    const baseIndent = getIndentation(codeLines[keywordLine]);
    
    while (endLine < codeLines.length - 1) {
      const nextLine = codeLines[endLine + 1];
      // If we find a line with same or less indentation, we've reached the end of the block
      if (nextLine.trim() !== '' && getIndentation(nextLine) <= baseIndent && !codeLines[endLine].trim().endsWith('\\')) {
        break;
      }
      endLine++;
    }
    
    return { startLine, endLine: endLine + 1 }; // +1 because we want to include the end line
  }
  
  // If no relevant code found, just return the original line for insertion
  return { startLine: nearLine, endLine: nearLine };
};

// Helper: Get indentation level
const getIndentation = (line) => {
  const match = line.match(/^(\s*)/);
  return match ? match[1].length : 0;
};

  // Apply suggested code to main code
  // IMPROVED: Apply suggested code to main code
const applySuggestion = (suggestion) => {
  // Save current code to history for undo
  setChangeHistory((prev) => [...prev, code]);

  const codeLines = code.split("\n");
  
  // Find where to replace code
  const { startLine, endLine } = findReplacementBoundaries(suggestion.type, suggestion.line);
  
  // Parse the suggested code
  const suggestionLines = suggestion.autoFix.split("\n");

  // Remove comment lines for cleaner insertion
  const codeToInsert = suggestionLines
    .filter((line) => !line.trim().startsWith("#"))
    .join("\n");

  // Track what's being added and removed for highlighting
  const addedLines = [];
  const removedLines = [];

  // Create new code with replacement
  const newCodeLines = [...codeLines];
  
  // Record removed lines
  for (let i = startLine; i < endLine; i++) {
    removedLines.push(i);
  }
  
  // Replace lines
  newCodeLines.splice(startLine, endLine - startLine, codeToInsert);
  
  // Track added lines
  for (let i = 0; i < suggestionLines.length; i++) {
    if (!suggestionLines[i].trim().startsWith("#")) {
      addedLines.push(startLine + i);
    }
  }

  const newCode = newCodeLines.join("\n");
  setCode(newCode);

  // Track changes for highlighting
  setLastChanges({
    added: addedLines,
    removed: removedLines,
  });

  // Show diff view
  setShowDiff(true);

  // Update suggestions list
  setSuggestions((prev) => prev.filter((s) => s.title !== suggestion.title));
};

  // Undo last code change
  const undoChange = () => {
    if (changeHistory.length > 0) {
      const previousCode = changeHistory[changeHistory.length - 1];
      setCode(previousCode);
      setChangeHistory((prev) => prev.slice(0, -1));
      setShowDiff(false);
      setLastChanges({ added: [], removed: [] });
    }
  };

  // Reset code to original
  const resetCode = () => {
    setCode(originalCode);
    setChangeHistory([]);
    setShowDiff(false);
    setLastChanges({ added: [], removed: [] });
  };

  // Analyze code with Gemini
  const analyzeCodeWithGemini = async () => {
    if (!code.trim() || code.length < 50) return;

    try {
      setAnalyzing(true);
      setSuggestions([]);

      // Create analysis context from model info
      const analysisContext = {
        code: code,
        framework: modelInfo?.framework || "unknown",
        modelMetrics: {
          accuracy: modelInfo?.accuracy || 0,
          precision: modelInfo?.precision || 0,
          recall: modelInfo?.recall || 0,
          f1: modelInfo?.f1 || 0,
          dataset_size: modelInfo?.dataset_size || 0,
        },
        analysisType: "ml_code_review",
      };

      const response = await fetch("http://localhost:8000/api/analyze-code", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(analysisContext),
      });

      if (response.ok) {
        const result = await response.json();
        setSuggestions(result.suggestions || []);
      } else {
        // Fallback to mock suggestions if API not available
        generateMockSuggestions(code);
      }
    } catch (err) {
      console.error("Bit analysis failed:", err);
      generateMockSuggestions(code);
    } finally {
      setAnalyzing(false);
    }
  };

  // Generate mock suggestions based on code patterns and model metrics
  const generateMockSuggestions = (codeToAnalyze) => {
    const suggestions = [];
    const framework = modelInfo?.framework?.toLowerCase() || "unknown";
    const accuracy = modelInfo?.accuracy || 0;

    // Pattern-based analysis
    const codeLines = codeToAnalyze.toLowerCase().split("\n");

    // Check for common ML issues
    if (accuracy < 0.8) {
      suggestions.push({
        type: "performance",
        severity: "high",
        line: findLineWithPattern(codeLines, ["model =", "class "]),
        title: "Low Model Accuracy Detected",
        message: `Your model accuracy is ${(accuracy * 100).toFixed(1)}%. Consider increasing model complexity or improving data quality.`,
        suggestion:
          framework === "pytorch"
            ? "Try adding more layers: nn.Linear(hidden_size, hidden_size * 2)"
            : framework === "tensorflow"
              ? 'Add more dense layers: tf.keras.layers.Dense(128, activation="relu")'
              : "Try RandomForestClassifier with more estimators",
        autoFix: generateAutoFix("increase_complexity", framework),
      });
    }

    // Check for missing regularization
    if (
      !codeToAnalyze.includes("dropout") &&
      !codeToAnalyze.includes("Dropout") &&
      framework !== "sklearn"
    ) {
      suggestions.push({
        type: "overfitting",
        severity: "medium",
        line: findLineWithPattern(codeLines, [
          "forward",
          "model.add",
          "sequential",
        ]),
        title: "Missing Regularization",
        message: "No dropout layers detected. This may lead to overfitting.",
        suggestion:
          framework === "pytorch"
            ? "Add: self.dropout = nn.Dropout(0.3)"
            : "Add: tf.keras.layers.Dropout(0.3)",
        autoFix: generateAutoFix("add_dropout", framework),
      });
    }

    // Check for hardcoded learning rates
    if (
      codeToAnalyze.includes("lr=0.01") ||
      codeToAnalyze.includes("learning_rate=0.01")
    ) {
      suggestions.push({
        type: "optimization",
        severity: "low",
        line: findLineWithPattern(codeLines, ["optimizer", "adam", "sgd"]),
        title: "Hardcoded Learning Rate",
        message:
          "Hardcoded learning rates may not be optimal for your specific problem.",
        suggestion: "Use adaptive learning rate or scheduler",
        autoFix: generateAutoFix("adaptive_lr", framework),
      });
    }

    // Check for class imbalance handling
    if (
      !codeToAnalyze.includes("class_weight") &&
      !codeToAnalyze.includes("WeightedRandomSampler")
    ) {
      suggestions.push({
        type: "data",
        severity: "medium",
        line: findLineWithPattern(codeLines, ["fit(", "train(", "dataloader"]),
        title: "Class Imbalance Not Addressed",
        message:
          "Consider handling class imbalance with weights or sampling techniques.",
        suggestion: 'Add class_weight="balanced" or use weighted sampling',
        autoFix: generateAutoFix("class_weights", framework),
      });
    }

    setSuggestions(suggestions);
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
        pytorch: `# Add more layers for better model capacity
self.hidden_layer2 = nn.Linear(hidden_size, hidden_size * 2)
self.output_layer = nn.Linear(hidden_size * 2, num_classes)`,
        tensorflow: `# Add more layers for better model capacity
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(64, activation='relu'))`,
        sklearn: `# Increase model complexity
model = RandomForestClassifier(n_estimators=200, max_depth=10)`,
      },
      add_dropout: {
        pytorch: `# Add dropout to prevent overfitting
self.dropout = nn.Dropout(0.3)

# Use in forward pass:
x = self.dropout(x)`,
        tensorflow: `# Add dropout to prevent overfitting
model.add(tf.keras.layers.Dropout(0.3))`,
      },
      adaptive_lr: {
        pytorch: `# Use learning rate scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)`,
        tensorflow: `# Use learning rate scheduler
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5)
model.fit(X_train, y_train, callbacks=[reduce_lr])`,
      },
      class_weights: {
        pytorch: `# Handle class imbalance with weighted sampling
class_counts = [sum(y_train == i) for i in range(num_classes)]
weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
sample_weights = weights[y_train]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights))`,
        tensorflow: `# Handle class imbalance with class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(enumerate(class_weights))
model.fit(X_train, y_train, class_weight=class_weight_dict)`,
      },
    };

    return (
      fixes[fixType]?.[framework] ||
      "# No specific fix available for this framework"
    );
  };

  // Sample model code for demo purposes
  const getSampleModelCode = (framework) => {
    switch (framework.toLowerCase()) {
      case "pytorch":
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

def train_model(model, train_loader, val_loader, epochs=10):
    """Train the PyTorch model."""
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    for epoch in range(epochs):
        model.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                val_loss += criterion(outputs, targets).item()
                pred = outputs.argmax(dim=1)
                correct += pred.eq(targets).sum().item()
                
        accuracy = correct / len(val_loader.dataset)
        print(f'Epoch {epoch+1}: Accuracy = {accuracy:.4f}')`;
      case "tensorflow":
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
      default:
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

def train_model(model, X_train, y_train):
    """Train the scikit-learn model."""
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(report)
    
    return accuracy`;
    }
  };

  // Add this function to generate code for a specific suggestion
  const generateCodeForSuggestion = async (suggestion) => {
    try {
      setAnalyzing(true); // Add loading state while generating code

      console.log("Starting code generation for suggestion:", suggestion);

      // Create request context
      const requestContext = {
        framework: modelInfo?.framework || "pytorch",
        suggestionType: suggestion.type,
        suggestionTitle: suggestion.title,
        currentCode: code,
        modelMetrics: {
          accuracy: modelInfo?.accuracy || 0,
          precision: modelInfo?.precision || 0,
          recall: modelInfo?.recall || 0,
        },
      };

      console.log("Request context prepared:", requestContext);

      // Try to call the Gemini API through the backend using GET
      try {
        console.log("Calling backend API for code generation");
        // Convert suggestion title to category format
        const category = suggestion.title.toLowerCase().replace(/\s+/g, "_");

        // Use GET request with query parameters
        const response = await fetch(
          `http://localhost:8000/api/generate-code-example?framework=${requestContext.framework}&category=${category}`,
        );

        if (response.ok) {
          const result = await response.json();
          console.log("Received code from API:", result);

          // Update the suggestion with the generated code
          const updatedSuggestions = [...suggestions];
          const index = updatedSuggestions.findIndex(
            (s) => s.title === suggestion.title && s.line === suggestion.line,
          );

          if (index !== -1) {
            console.log("Updating suggestion at index:", index);
            updatedSuggestions[index].autoFix = result.code;
            setSuggestions(updatedSuggestions);
            return; // Exit early if API call was successful
          }
        } else {
          console.error("API call failed:", await response.text());
          throw new Error("API call failed");
        }
      } catch (apiError) {
        console.error("Error calling Bit API:", apiError);
        console.log("Falling back to local code generation");
      }

      // If we get here, the API call failed, so use fallback code
      console.log("Using fallback code generation");

      // Add specific case for "Unnecessary Gradient Computation"
      let fallbackCode;
      if (suggestion.title === "Unnecessary Gradient Computation") {
        fallbackCode = `# Prevent unnecessary gradient computation during inference
def inference(model, input_data):
    # Set model to evaluation mode
    model.eval()
    
    # Disable gradient computation for inference
    with torch.no_grad():
        predictions = model(input_data)
        
    return predictions

# Example usage:
# test_predictions = inference(model, test_data)`;
      } else {
        fallbackCode = generateFallbackCode(
          suggestion,
          modelInfo?.framework || "pytorch",
        );
      }

      // Update the suggestion with fallback code
      const updatedSuggestions = [...suggestions];
      const index = updatedSuggestions.findIndex(
        (s) => s.title === suggestion.title && s.line === suggestion.line,
      );

      if (index !== -1) {
        console.log("Updating suggestion at index:", index);
        updatedSuggestions[index].autoFix = fallbackCode;
        setSuggestions(updatedSuggestions);
      } else {
        console.error("Could not find matching suggestion to update");
      }
    } catch (err) {
      console.error("Error in code generation:", err);
    } finally {
      setAnalyzing(false); // Reset loading state
    }
  };

  // Add a fallback code generator function
  const generateFallbackCode = (suggestion, framework) => {
    const { type, title } = suggestion;

    // Generate different code based on suggestion type and framework
    if (type === "data_preprocessing" && title.includes("Data Normalization")) {
      if (framework === "pytorch") {
        return `# Add data normalization for better convergence
from sklearn.preprocessing import StandardScaler
import numpy as np

# For preprocessing the data before creating DataLoader
def normalize_data(X_train, X_val=None):
    """Normalize input features using StandardScaler."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    if X_val is not None:
        # Use the same scaler for validation data
        X_val_scaled = scaler.transform(X_val)
        return X_train_scaled, X_val_scaled, scaler
    
    return X_train_scaled, scaler

# Example usage:
# X_train_scaled, X_val_scaled, scaler = normalize_data(X_train, X_val)
# train_dataset = TensorDataset(torch.FloatTensor(X_train_scaled), torch.LongTensor(y_train))
# val_dataset = TensorDataset(torch.FloatTensor(X_val_scaled), torch.LongTensor(y_val))

# Alternative: Add BatchNorm layers to your model
class NormalizedNeuralNetwork(nn.Module):
    def __init__(self, input_size=10, hidden_size=20, num_classes=2):
        super(NormalizedNeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)  # Add BatchNorm after first layer
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.bn1(out)  # Apply BatchNorm
        out = self.relu(out)
        out = self.layer2(out)
        return F.log_softmax(out, dim=1)`;
      } else if (framework === "tensorflow") {
        return `# Add data normalization for better convergence
from sklearn.preprocessing import StandardScaler
import numpy as np

# Method 1: Use StandardScaler before training
def normalize_data(X_train, X_val=None):
    """Normalize input features using StandardScaler."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    if X_val is not None:
        # Use the same scaler for validation data
        X_val_scaled = scaler.transform(X_val)
        return X_train_scaled, X_val_scaled, scaler
    
    return X_train_scaled, scaler

# Example usage:
# X_train_scaled, X_val_scaled, scaler = normalize_data(X_train, X_val)

# Method 2: Add normalization layer to your model
def create_normalized_model(input_shape, num_classes=2):
    model = tf.keras.Sequential([
        # Add a normalization layer that adapts to the data
        tf.keras.layers.Normalization(axis=-1),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax' if num_classes > 2 else 'sigmoid')
    ])
    
    # Adapt the normalization layer to the training data
    # norm_layer.adapt(X_train)
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy' if num_classes > 2 else 'binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model`;
      } else {
        return `# Add data normalization for better convergence
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# Method 1: Use StandardScaler directly
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model on scaled data
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Method 2: Use Pipeline to combine preprocessing and model
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # First scale the data
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))  # Then apply the classifier
])

# Train the pipeline
pipeline.fit(X_train, y_train)

# Predict using the pipeline (scaling happens automatically)
y_pred = pipeline.predict(X_test)`;
      }
    } else if (type === "performance" && title.includes("Accuracy")) {
      if (framework === "pytorch") {
        return `# Increase model complexity to improve accuracy
class ImprovedNeuralNetwork(nn.Module):
    def __init__(self, input_size=10, hidden_size=64, num_classes=2):
        super(ImprovedNeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn2 = nn.BatchNorm1d(hidden_size // 2)
        self.layer3 = nn.Linear(hidden_size // 2, num_classes)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.layer2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.layer3(out)
        return F.log_softmax(out, dim=1)`;
      } else if (framework === "tensorflow") {
        return `# Increase model complexity to improve accuracy
def create_improved_model(input_shape, num_classes=2):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(input_shape,)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax' if num_classes > 2 else 'sigmoid')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy' if num_classes > 2 else 'binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model`;
      } else {
        return `# Increase model complexity to improve accuracy
def create_improved_model():
    model = RandomForestClassifier(
        n_estimators=200,  # More trees
        max_depth=15,      # Deeper trees
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        bootstrap=True,
        random_state=42
    )
    return model`;
      }
    } else if (type === "overfitting" && title.includes("Regularization")) {
      if (framework === "pytorch") {
        return `# Add dropout regularization to prevent overfitting
class RegularizedNeuralNetwork(nn.Module):
    def __init__(self, input_size=10, hidden_size=20, num_classes=2):
        super(RegularizedNeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.dropout1 = nn.Dropout(0.3)  # Add dropout after first layer
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.relu(out)
        out = self.dropout1(out)  # Apply dropout
        out = self.layer2(out)
        return F.log_softmax(out, dim=1)`;
      } else if (framework === "tensorflow") {
        return `# Add dropout regularization to prevent overfitting
def create_regularized_model(input_shape, num_classes=2):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(input_shape,)),
        tf.keras.layers.Dropout(0.3),  # Add dropout layer
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.3),  # Add dropout layer
        tf.keras.layers.Dense(num_classes, activation='softmax' if num_classes > 2 else 'sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy' if num_classes > 2 else 'binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model`;
      } else {
        return `# Add regularization to prevent overfitting
from sklearn.linear_model import LogisticRegression

def create_regularized_model():
    # Use L2 regularization (Ridge)
    model = LogisticRegression(
        C=0.1,  # Smaller C means stronger regularization
        penalty='l2',
        solver='liblinear',
        random_state=42
    )
    return model`;
      }
    } else if (type === "optimization" && title.includes("Learning Rate")) {
      if (framework === "pytorch") {
        return `# Use adaptive learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Add learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min',
    factor=0.1,
    patience=5,
    verbose=True
)

# In training loop:
for epoch in range(epochs):
    train_loss = train_one_epoch(model, train_loader, optimizer)
    val_loss = validate(model, val_loader)
    
    # Update learning rate based on validation loss
    scheduler.step(val_loss)
    
    # Optional: print current learning rate
    current_lr = optimizer.param_groups[0]['lr']
    print(f'Epoch {epoch+1}, Current LR: {current_lr}')`;
      } else if (framework === "tensorflow") {
        return `# Use adaptive learning rate
initial_learning_rate = 0.001

# Method 1: Learning rate schedule
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=10000,
    decay_rate=0.9,
    staircase=True
)

# Use the schedule in the optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

# Method 2: Use callbacks for learning rate reduction
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.2,
    patience=5, 
    min_lr=0.0001,
    verbose=1
)

# Use in model.fit
model.fit(
    X_train, y_train,
    epochs=epochs,
    validation_data=(X_val, y_val),
    callbacks=[reduce_lr]
)`;
      } else {
        return `# For scikit-learn, use grid search to find optimal hyperparameters
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,
    scoring='accuracy',
    verbose=1
)

grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
print(f"Best parameters: {grid_search.best_params_}")`;
      }
    } else if (type === "data" && title.includes("Class Imbalance")) {
      if (framework === "pytorch") {
        return `# Handle class imbalance with weighted sampling
from torch.utils.data import WeightedRandomSampler

# Calculate class weights
y_train = torch.tensor(y_train)
class_counts = torch.bincount(y_train)
class_weights = 1.0 / class_counts.float()
weights = class_weights[y_train]

# Create weighted sampler
sampler = WeightedRandomSampler(
    weights=weights,
    num_samples=len(y_train),
    replacement=True
)

# Use sampler in DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    sampler=sampler  # Use the weighted sampler
)`;
      } else if (framework === "tensorflow") {
        return `# Handle class imbalance with class weights
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

# Calculate class weights
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = dict(enumerate(class_weights))

# Use class weights in model.fit
model.fit(
    X_train, y_train,
    epochs=epochs,
    batch_size=32,
    validation_data=(X_val, y_val),
    class_weight=class_weight_dict  # Apply class weights
)`;
      } else {
        return `# Handle class imbalance in scikit-learn
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

# Use SMOTE to oversample minority class
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Train model on balanced dataset
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)
model.fit(X_train_resampled, y_train_resampled)

# Alternative: Use class_weight parameter
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    class_weight='balanced',  # Use balanced class weights
    random_state=42
)
model.fit(X_train, y_train)`;
      }
    }

    // Default fallback code
    return `# Generated code for: ${title}
# Framework: ${framework}
# This is a fallback implementation

# Please modify this template based on your specific model structure
def improve_model():
    """
    Implement the suggested improvement: ${suggestion.suggestion}
    """
    # TODO: Implement the specific improvement
    pass`;
  };

  // Helper function to get severity color
// Severity color function
const getSeverityColor = (severity) => {
  switch (severity) {
    case "high":
      return "#D5451B"; // Cinder primary
    case "medium":
      return "#FF9B45"; // Cinder primary-light
    case "low":
      return "#6A9955"; // Green
    default:
      return "#858585"; // Gray
  }
};

{showSearch && (
  <div style={{
    padding: "8px 16px",
    backgroundColor: "#2D2D2D",
    display: "flex",
    alignItems: "center",
    gap: "8px",
    borderBottom: "1px solid #333"
  }}>
    <input
      type="text"
      value={searchText}
      onChange={(e) => setSearchText(e.target.value)}
      placeholder="Search in code..."
      style={{
        backgroundColor: "#1E1E1E",
        color: "#D4D4D4",
        border: "1px solid #555",
        borderRadius: "4px",
        padding: "4px 8px",
        flex: 1
      }}
      onKeyDown={(e) => {
        if (e.key === 'Enter') {
          if (e.shiftKey) {
            findPrev();
          } else {
            searchResults.length > 0 ? findNext() : searchInCode();
          }
        }
        if (e.key === 'Escape') {
          setShowSearch(false);
        }
      }}
    />
    <button 
      onClick={searchInCode}
      style={{
        backgroundColor: "#D5451B",
        color: "white",
        border: "none",
        borderRadius: "4px",
        padding: "4px 8px",
        cursor: "pointer"
      }}
    >
      Find
    </button>
    {searchResults.length > 0 && (
      <>
        <button 
          onClick={findPrev}
          style={{
            backgroundColor: "#2D2D2D",
            color: "#D4D4D4",
            border: "1px solid #555",
            borderRadius: "4px",
            padding: "4px 8px",
            cursor: "pointer"
          }}
        >
          Previous
        </button>
        <button 
          onClick={findNext}
          style={{
            backgroundColor: "#2D2D2D",
            color: "#D4D4D4",
            border: "1px solid #555",
            borderRadius: "4px",
            padding: "4px 8px",
            cursor: "pointer"
          }}
        >
          Next
        </button>
        <span style={{ color: "#D4D4D4", fontSize: "12px" }}>
          {searchIndex + 1} of {searchResults.length}
        </span>
      </>
    )}
    <button 
      onClick={() => setShowSearch(false)}
      style={{
        backgroundColor: "#2D2D2D",
        color: "#D4D4D4",
        border: "1px solid #555",
        borderRadius: "4px",
        padding: "4px 8px",
        cursor: "pointer"
      }}
    >
      Close
    </button>
  </div>
)}

  // Function to copy code to clipboard
  const copyToClipboard = (code, suggestionIndex) => {
    navigator.clipboard.writeText(code);
    setCopiedCode(suggestionIndex);
  };

  if (loading) {
  return (
    <div
      style={{
        padding: "2rem",
        textAlign: "center",
        backgroundColor: "#1E1E1E",
        color: "#D4D4D4",
        minHeight: "400px",
        display: "flex",
        flexDirection: "column",
        justifyContent: "center",
        alignItems: "center",
      }}
    >
      <div style={{ 
        display: "flex", 
        alignItems: "center", 
        gap: "12px", 
        padding: "12px", 
        background: "#2D2D2D", 
        borderRadius: "8px",
        margin: "0 auto",
        width: "fit-content"
      }}>
        <div className="bit-offset" style={{ 
          width: "20px", 
          height: "20px", 
          position: "relative"
        }}>
          <div className="offset-back" style={{ 
            width: "18px", 
            height: "18px", 
            borderRadius: "4px",
            background: "#FF9B45", // Cinder primary-light
            position: "absolute",
            bottom: "0",
            right: "0",
            opacity: "0.6"
          }}></div>
          <div className="offset-front" style={{ 
            width: "18px", 
            height: "18px", 
            borderRadius: "4px",
            background: "#D5451B", // Cinder primary
            position: "absolute",
            top: "0",
            left: "0"
          }}></div>
        </div>
        <span style={{ fontSize: "14px", color: "#D4D4D4" }}>
          Loading model code...
        </span>
      </div>
      // Add this to your component, typically near the end
<style>
  {`
    @keyframes pulse {
      0%, 100% { opacity: 1; }
      50% { opacity: 0.5; }
    }
    
    .bit-offset {
      animation: pulse 2s infinite;
    }
    
    .pulse-animation {
      animation: pulse 2s infinite;
    }
    
    ::-webkit-scrollbar {
      width: 8px;
      height: 8px;
    }
    ::-webkit-scrollbar-track {
      background: #1e1e1e;
    }
    ::-webkit-scrollbar-thumb {
      background: #555;
      border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
      background: #777;
    }
    .suggestions-panel {
      transition: width 0.3s ease;
    }
    .resize-handle-visible {
      background-color: rgba(231, 76, 50, 0.1);
    }
  `}
</style>
    </div>
  );
}

  return (
    <div
  className="code-editor-wrapper"
  ref={containerRef}
  style={{
    backgroundColor: "#1E1E1E", // Dark background
    color: "#D4D4D4", // Light text
    minHeight: "100vh",
    display: "flex",
    position: "relative",
    overflow: "hidden",
  }}

  
>
      {/* Main Code Viewer - make sure it can shrink */}
      <div
        style={{
          flex: "1 1 auto", // Allow shrinking
          minWidth: "300px", // Ensure minimum width
          display: "flex",
          flexDirection: "column",
        }}
      >
        {/* Header */}
        <div
  style={{
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    padding: "1rem 1.5rem",
    backgroundColor: "#252526", // Darker background
    borderBottom: "1px solid #333333",
    boxShadow: "0 1px 3px rgba(0, 0, 0, 0.2)",
    flexWrap: "wrap",
    gap: "1rem",
  }}
>
          <div>
            <h3
              style={{
                margin: 0,
                fontSize: "1.1rem",
                fontWeight: "600",
                color: "white",
              }}
            >
              Bit's Hyperparameter Tuning Recommendations
              {analyzing && (
                <span
                  style={{
                    marginLeft: "1rem",
                    padding: "0.2rem 0.5rem",
                    backgroundColor: "#D5451B",
                    borderRadius: "0.25rem",
                    fontSize: "0.7rem",
                    fontWeight: "500",
                    color: "white",
                  }}
                >
                  Analyzing
                </span>
              )}
            </h3>
            <p
              style={{
                margin: "0.25rem 0 0 0",
                fontSize: "0.8rem",
                color: "white",
              }}
            >
              Machine Learning Model Analysis
              {modelInfo?.framework && (
                <span
                  style={{
                    marginLeft: "0.5rem",
                    padding: "0.2rem 0.5rem",
                    backgroundColor: "#D5451B",
                    borderRadius: "0.25rem",
                    fontSize: "0.7rem",
                    fontWeight: "500",
                    color: "white",
                  }}
                >
                  {modelInfo.framework}
                </span>
              )}
            </p>
          </div>

          <div style={{ display: "flex", gap: "0.5rem" }}>
            <button
              onClick={analyzeCodeWithGemini}
              disabled={analyzing}
              style={{
                padding: "0.5rem 1rem",
                fontSize: "0.8rem",
                backgroundColor: "#D5451B", // Cinder primary
                color: "white",
                border: "none",
                borderRadius: "0.25rem",
                cursor: analyzing ? "not-allowed" : "pointer",
                display: "flex",
                alignItems: "center",
                gap: "0.5rem",
                fontWeight: "500",
              }}
            >
              <div style={{ 
                width: "16px", 
                height: "16px", 
                position: "relative",
                flexShrink: 0
              }}>
                <div style={{ 
                  width: "14px", 
                  height: "14px", 
                  background: "rgba(255,255,255,0.3)", 
                  borderRadius: "3px", 
                  position: "absolute", 
                  bottom: "0", 
                  right: "0"
                }}></div>
                <div style={{ 
                  width: "14px", 
                  height: "14px", 
                  background: "rgba(255,255,255,0.9)", 
                  borderRadius: "3px", 
                  position: "absolute", 
                  top: "0", 
                  left: "0" 
                }}></div>
              </div>
              {analyzing ? "Analyzing..." : "Bit Analyze"}
            </button>
            <div style={{ display: "flex", gap: "0.5rem" }}>
              {changeHistory.length > 0 && (
                <button
                  onClick={undoChange}
                  style={{
                    padding: "0.5rem 1rem",
                    fontSize: "0.8rem",
                    backgroundColor: "#f8fafc",
                    color: "#333333",
                    border: "1px solid #e1e1e1",
                    borderRadius: "0.25rem",
                    cursor: "pointer",
                    display: "flex",
                    alignItems: "center",
                    gap: "0.5rem",
                    fontWeight: "500",
                  }}
                >
                  Undo Change
                </button>
              )}

              {code !== originalCode && (
                <button
                  onClick={resetCode}
                  style={{
                    padding: "0.5rem 1rem",
                    fontSize: "0.8rem",
                    backgroundColor: "#f8fafc",
                    color: "#333333",
                    border: "1px solid #e1e1e1",
                    borderRadius: "0.25rem",
                    cursor: "pointer",
                    display: "flex",
                    alignItems: "center",
                    gap: "0.5rem",
                    fontWeight: "500",
                  }}
                >
                  Reset Code
                </button>
              )}
            </div>
          </div>
        </div>

{/* Status Bar */}
<div
  style={{
    padding: "0.4rem 1.5rem",
    backgroundColor: "#252526", // Darker background
    color: "#D4D4D4", // Light text
    fontFamily: "'JetBrains Mono', 'Consolas', 'Monaco', monospace",
    fontSize: "0.75rem",
    fontWeight: "500",
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
  }}
>
  <span>model_code.py</span>
  <div>
    <span>{code.split("\n").length} lines</span>
    {showDiff && (
      <>
        <span style={{ marginLeft: "1rem", color: "#10b981" }}>
          {lastChanges.added.length > 0 &&
            `+${lastChanges.added.length} lines added`}
        </span>
        <span style={{ marginLeft: "1rem", color: "#ef4444" }}>
          {lastChanges.removed.length > 0 &&
            `-${lastChanges.removed.length} lines removed`}
        </span>
      </>
    )}
    {suggestions.length > 0 && (
      <span style={{ marginLeft: "1rem" }}>
        {suggestions.length} suggestions
      </span>
    )}
  </div>
</div>

        {/* Code Viewer (Read-only) */}
        <div
          style={{
            height: "calc(100vh - 137px)",
            overflow: "auto",
            position: "relative",
            flex: 1,
            backgroundColor: "#1e1e1e",  // Dark background
          }}
        >
          <SyntaxHighlighter
            language="python"
            style={cinderTheme}
            showLineNumbers={true}
            lineNumberStyle={{
              minWidth: "3em",
              paddingRight: "1em",
              textAlign: "right",
              color: "#858585", // Lighter grey for line numbers
              borderRight: "1px solid #333333",
              marginRight: "1em",
              userSelect: "none",
            }}
            customStyle={{
              margin: 0,
              padding: "1rem",
              backgroundColor: "#1E1E1E", // Dark background
              fontSize: "14px",
              lineHeight: "1.5",
              fontFamily: "'JetBrains Mono', 'Consolas', 'Monaco', monospace",
              overflow: "visible",
            }}
            codeTagProps={{
              style: {
                fontFamily: "'JetBrains Mono', 'Consolas', 'Monaco', monospace",
              },
            }}
            wrapLines={true}
            lineProps={(lineNumber) => {
              // Highlight lines with suggestions
              const hasSuggestion = suggestions.some(
                (s) => s.line === lineNumber,
              );

              const hasSearchResults = searchResults.some(result => result.lineIndex === lineNumber - 1);


              // Highlight added lines
              const isAdded = lastChanges.added.includes(lineNumber);
              
              // Highlight removed lines
              const isRemoved = lastChanges.removed.includes(lineNumber);

              // Decide on style based on highlights
              let style = { display: "block" };
              if (hasSearchResults && showSearch) {
                style.backgroundColor = style.backgroundColor || "rgba(213, 69, 27, 0.1)";
                // Add an indicator
                style.position = "relative";
              }
              if (isAdded) {
                style.backgroundColor = "rgba(16, 185, 129, 0.2)"; // Darker green for added
                style.borderLeft = "3px solid #10b981";
                style.paddingLeft = "1rem";
              } else if (isRemoved && showDiff) {
                style.backgroundColor = "rgba(239, 68, 68, 0.2)"; // Darker red for removed
                style.borderLeft = "3px solid #ef4444";
                style.paddingLeft = "1rem";
                style.textDecoration = "line-through";
                style.opacity = "0.6";
              } else if (hasSuggestion) {
                style.backgroundColor = "rgba(213, 69, 27, 0.2)"; // Darker Cinder primary for suggestions
                style.borderLeft = "3px solid #D5451B";
                style.paddingLeft = "1rem";
              }

              return { style };
            }}
          >
            {code}
          </SyntaxHighlighter>
        </div>
      </div>

      
      

      {/* Suggestions Panel */}
      {showSuggestions && (
        <div
  className="suggestions-panel"
  style={{
    width: `${panelWidth}px`,
    backgroundColor: "white", // Darker background for panel
    borderLeft: "1px solid #333333",
    display: "flex",
    flexDirection: "column",
    height: "100vh",
    overflow: "hidden",
    color: "#D4D4D4", // Light text
    position: "relative",
    boxShadow: "-2px 0 10px rgba(0, 0, 0, 0.2)",
  }}
>
          {/* Resize Handle */}
          <div
            style={{
              position: "absolute",
              left: "-5px", // Position it slightly outside the panel for easier grabbing
              top: 0,
              width: "10px", // Make it wider for easier grabbing
              height: "100%",
              cursor: "col-resize",
              zIndex: 10,
              backgroundColor: isResizing
                ? "rgba(231, 76, 50, 0.3)"
                : "transparent",
              transition: "background-color 0.2s",
              // Add a visible indicator
              "&::after": {
                content: '""',
                position: "absolute",
                left: "5px",
                top: 0,
                width: "2px",
                height: "100%",
                backgroundColor: "#e1e1e1",
              },
            }}
            onMouseDown={startResize}
            onMouseOver={(e) =>
              (e.currentTarget.style.backgroundColor = "rgba(231, 76, 50, 0.1)")
            }
            onMouseOut={(e) =>
              (e.currentTarget.style.backgroundColor = "transparent")
            }
          />

          {/* Suggestions Header */}
          <div
            style={{
              padding: "0.75rem 1.25rem",
              marginTop: "1rem",
              marginLeft: "1rem",
              marginRight: "1rem",
              backgroundColor: "#fafafb", // Slightly lighter dark background
              borderRadius: "9999px",
              display: "flex",
              alignItems: "center",
              gap: "0.75rem",
              boxShadow: "0 1px 3px rgba(0, 0, 0, 0.2)",
            }}
          >
            <div
              style={{
                width: "0.75rem",
                height: "0.75rem",
                backgroundColor: "#D5451B", // Cinder primary
                borderRadius: "50%",
              }}
            ></div>

            <h4
              style={{
                margin: 0,
                fontSize: "1rem",
                fontWeight: "400",
                color: "black",
                fontFamily:
                  '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif',
              }}
            >
              Bit's Suggestions
            </h4>
          </div>

          {/* Suggestions count as a separate element */}
          <div
            style={{
              padding: "0.5rem 1.25rem",
              display: "flex",
              alignItems: "center",
            }}
          >
            <p
              style={{
                margin: 0,
                fontSize: "0.9rem",
                color: "#666",
                fontFamily:
                  '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif',
              }}
            >
              {analyzing
                ? "Analyzing your code..."
                : suggestions.length > 0
                  ? `${suggestions.length} suggestions found`
                  : ""}
            </p>
          </div>

          {/* Model Performance Context */}
          {modelInfo && (
            <div
              style={{
                margin: "1rem",
                backgroundColor: "#ffffff",
                borderRadius: "0.75rem",
                boxShadow: "0 1px 3px rgba(0, 0, 0, 0.05)",
                overflow: "hidden",
                fontFamily:
                  'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace',
              }}
            >
              {/* Section Header */}
              <div
                style={{
                  padding: "0.75rem 1.25rem",
                  backgroundColor: "#f8f9fa",
                  borderBottom: "1px solid #f0f0f0",
                  display: "flex",
                  alignItems: "center",
                  gap: "0.75rem",
                }}
              >
                <h5
                  style={{
                    margin: 0,
                    fontSize: "1rem",
                    fontWeight: "300",
                    color: "#333",
                  }}
                >
                  Current Model Performance
                </h5>
              </div>

              {/* Section Content */}
              <div
                style={{
                  padding: "1rem 1.25rem",
                  fontSize: "0.9rem",
                  color: "#666",
                }}
              >
                {/* Metrics displayed in a clean, modern style */}
                <div
                  style={{
                    display: "grid",
                    gridTemplateColumns: "repeat(2, 1fr)",
                    gap: "1rem",
                  }}
                >
                  {/* Accuracy */}
                  <div
                    style={{
                      padding: "0.75rem",
                      backgroundColor: "#f8f9fa",
                      borderRadius: "0.5rem",
                    }}
                  >
                    <div
                      style={{
                        fontSize: "0.8rem",
                        color: "#666",
                        marginBottom: "0.25rem",
                      }}
                    >
                      Accuracy
                    </div>
                    <div
                      style={{
                        fontSize: "1.25rem",
                        fontWeight: "500",
                        color: "#333",
                      }}
                    >
                      {((modelInfo.accuracy || 0) * 100).toFixed(1)}%
                    </div>
                  </div>

                  {/* Precision */}
                  {modelInfo.precision && (
                    <div
                      style={{
                        padding: "0.75rem",
                        backgroundColor: "#f8f9fa",
                        borderRadius: "0.5rem",
                      }}
                    >
                      <div
                        style={{
                          fontSize: "0.8rem",
                          color: "#666",
                          marginBottom: "0.25rem",
                        }}
                      >
                        Precision
                      </div>
                      <div
                        style={{
                          fontSize: "1.25rem",
                          fontWeight: "500",
                          color: "#333",
                        }}
                      >
                        {(modelInfo.precision * 100).toFixed(1)}%
                      </div>
                    </div>
                  )}

                  {/* Recall */}
                  {modelInfo.recall && (
                    <div
                      style={{
                        padding: "0.75rem",
                        backgroundColor: "#f8f9fa",
                        borderRadius: "0.5rem",
                      }}
                    >
                      <div
                        style={{
                          fontSize: "0.8rem",
                          color: "#666",
                          marginBottom: "0.25rem",
                        }}
                      >
                        Recall
                      </div>
                      <div
                        style={{
                          fontSize: "1.25rem",
                          fontWeight: "500",
                          color: "#333",
                        }}
                      >
                        {(modelInfo.recall * 100).toFixed(1)}%
                      </div>
                    </div>
                  )}

                  {/* Framework */}
                  <div
                    style={{
                      padding: "0.75rem",
                      backgroundColor: "#f8f9fa",
                      borderRadius: "0.5rem",
                    }}
                  >
                    <div
                      style={{
                        fontSize: "0.8rem",
                        color: "#666",
                        marginBottom: "0.25rem",
                      }}
                    >
                      Framework
                    </div>
                    <div
                      style={{
                        fontSize: "1.25rem",
                        fontWeight: "300",
                        color: "#333",
                      }}
                    >
                      {modelInfo.framework || "Unknown"}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Suggestions List */}
          <div
            style={{
              flex: 1,
              overflow: "auto",
              padding: "0.75rem",
            }}
          >
            {analyzing && (
  <div
    style={{
      padding: "2rem",
      textAlign: "center",
      color: "#D4D4D4",
    }}
  >
    <div style={{ 
      display: "flex", 
      alignItems: "center", 
      gap: "12px", 
      padding: "12px", 
      background: "#fafafb", 
      borderRadius: "8px",
      margin: "0 auto",
      width: "fit-content"
    }}>
      <div className="bit-offset" style={{ 
        width: "20px", 
        height: "20px", 
        position: "relative"
      }}>
        <div className="offset-back" style={{ 
          width: "18px", 
          height: "18px", 
          borderRadius: "4px",
          background: "#FF9B45", // Cinder primary-light
          position: "absolute",
          bottom: "0",
          right: "0",
          opacity: "0.6"
        }}></div>
        <div className="offset-front" style={{ 
          width: "18px", 
          height: "18px", 
          borderRadius: "4px",
          background: "#D5451B", // Cinder primary
          position: "absolute",
          top: "0",
          left: "0"
        }}></div>
      </div>
      <span style={{ fontSize: "14px", color: "black" }}>
        Bit is analyzing your code...
      </span>
    </div>
  </div>
)}

            {!analyzing && suggestions.length === 0 && (
              <div
                style={{
                  padding: "3rem",
                  textAlign: "center",
                  color: "#666",
                }}
              >
                <div
                  style={{
                    fontSize: "7.5rem",
                    marginBottom: "1.5rem",
                    color: "#D5451B",
                  }}
                >
                </div>
                <p style={{ fontSize: "1rem", lineHeight: "1.5" }}></p>
              </div>
            )}

            {suggestions.map((suggestion, index) => (
              <div
                key={index}
                style={{
                  backgroundColor: "#fff",
                  border: "1px solid #e1e1e1",
                  borderRadius: "0.5rem",
                  margin: "0 0 1rem 0",
                  overflow: "hidden",
                  boxShadow: "0 1px 3px rgba(0, 0, 0, 0.1)",
                  borderLeft: `4px solid ${getSeverityColor(suggestion.severity)}`,
                }}
              >
                {/* Suggestion Header */}
                <div
                  onClick={() =>
                    setSelectedSuggestion(
                      selectedSuggestion === index ? null : index,
                    )
                  }
                  style={{
                    padding: "1rem",
                    borderBottom:
                      selectedSuggestion === index
                        ? "1px solid #e1e1e1"
                        : "none",
                    cursor: "pointer",
                    backgroundColor:
                      selectedSuggestion === index ? "#f9fafb" : "#fff",
                  }}
                >
                  <div
                    style={{
                      display: "flex",
                      justifyContent: "space-between",
                      alignItems: "center",
                    }}
                  >
                    <div
                      style={{
                        display: "flex",
                        alignItems: "center",
                        gap: "0.75rem",
                      }}
                    >
                      <span
                        style={{
                          width: "0.75rem",
                          height: "0.75rem",
                          borderRadius: "50%",
                          backgroundColor: getSeverityColor(
                            suggestion.severity,
                          ),
                        }}
                      ></span>
                      <h5
                        style={{
                          margin: 0,
                          fontSize: "1rem",
                          fontWeight: "500",
                          color: "#333",
                        }}
                      >
                        {suggestion.title}
                      </h5>
                    </div>
                    <span
                      style={{
                        fontSize: "0.8rem",
                        padding: "0.2rem 0.5rem",
                        backgroundColor: "#f1f5f9",
                        borderRadius: "0.25rem",
                        color: "#64748b",
                      }}
                    >
                      Line {suggestion.line}
                    </span>
                  </div>
                  <p
                    style={{
                      margin: "0.75rem 0 0 0",
                      fontSize: "0.9rem",
                      color: "#666",
                      lineHeight: 1.5,
                    }}
                  >
                    {suggestion.message}
                  </p>
                </div>

                {/* Expanded Content */}
                {selectedSuggestion === index && (
                  <div style={{ padding: "1rem", backgroundColor: "#f9fafb" }}>
                    {/* Suggestion Details */}
                    <div style={{ marginBottom: "1rem" }}>
                      <h6
                        style={{
                          margin: "0 0 0.75rem 0",
                          fontSize: "0.9rem",
                          fontWeight: "600",
                          color: "#333",
                        }}
                      >
                        Recommended Solution
                      </h6>
                      <p
                        style={{
                          margin: 0,
                          fontSize: "0.9rem",
                          color: "#4b5563",
                          lineHeight: 1.5,
                        }}
                      >
                        {suggestion.suggestion}
                      </p>
                    </div>

                    {/* Auto-fix Code */}
                    {suggestion.autoFix && (
                      <div style={{ marginBottom: "1rem" }}>
                        <h6
                          style={{
                            margin: "0 0 0.75rem 0",
                            fontSize: "0.9rem",
                            fontWeight: "600",
                            color: "#333",
                          }}
                        >
                          Suggested Code
                        </h6>
                        <div
                          style={{
                            backgroundColor: "#1e1e1e",
                            borderRadius: "0.25rem",
                            overflow: "hidden",
                          }}
                        >
                          <SyntaxHighlighter
                            language="python"
                            style={vscDarkPlus}
                            customStyle={{
                              margin: 0,
                              fontSize: "0.85rem",
                              backgroundColor: "#1e1e1e",
                              padding: "1rem",
                            }}
                          >
                            {suggestion.autoFix}
                          </SyntaxHighlighter>

                          {/* Code footer with copy button */}
                          <div
                            style={{
                              display: "flex",
                              justifyContent: "space-between",
                              alignItems: "center",
                              padding: "0.5rem 1rem",
                              backgroundColor: "#252526",
                              borderTop: "1px solid #333",
                            }}
                          >
                            <span
                              style={{
                                color: "#aaa",
                                fontSize: "0.75rem",
                                display: "flex",
                                alignItems: "center",
                                gap: "0.5rem",
                              }}
                            >
                              <span
                                style={{ color: "#D5451B", fontWeight: "bold" }}
                              >
                                B
                              </span>
                              Created by Bit
                            </span>
                            <button
                              onClick={() =>
                                copyToClipboard(suggestion.autoFix, index)
                              }
                              style={{
                                background: "none",
                                border: "1px solid #555",
                                borderRadius: "0.25rem",
                                padding: "0.25rem 0.5rem",
                                color: "#eee",
                                fontSize: "0.75rem",
                                cursor: "pointer",
                              }}
                            >
                              {copiedCode === index ? "Copied!" : "Copy Code"}
                            </button>
                          </div>
                        </div>
                      </div>
                    )}

                    {/* Action Buttons */}
                    <div
                      style={{
                        display: "flex",
                        justifyContent: "space-between",
                        gap: "1rem",
                      }}
                    >
                      <button
                        onClick={() => {
                          console.log(
                            "Generate button clicked for suggestion:",
                            suggestion,
                          );
                          // Generate code with Gemini for this specific suggestion
                          generateCodeForSuggestion(suggestion);
                        }}
                        style={{
                          padding: "0.5rem 1rem",
                          fontSize: "0.85rem", 
                          backgroundColor: "#f0f0f0",
                          fontWeight: '300',
                          fontFamily: 'Consolas, "Courier New", monospace',
                          color: "black",
                          border: "none",
                          borderRadius: "0.25rem",
                          cursor: "pointer",
                          flex: 1,
                          display: "flex",
                          alignItems: "center",
                          justifyContent: "center",
                          gap: "8px"
                        }}
                      >
                        <div style={{ 
                          width: "16px", 
                          height: "16px", 
                          position: "relative",
                          flexShrink: 0
                        }}>
                          <div style={{ 
                            width: "14px", 
                            height: "14px", 
                            background: "#FF9B45", 
                            borderRadius: "3px", 
                            position: "absolute", 
                            bottom: "0", 
                            right: "0", 
                            opacity: "0.6" 
                          }}></div>
                          <div style={{ 
                            width: "14px", 
                            height: "14px", 
                            background: "#D5451B", 
                            borderRadius: "3px", 
                            position: "absolute", 
                            top: "0", 
                            left: "0" 
                          }}></div>
                        </div>
                        <span>Bit Code</span>
                      </button>
                      <button
                        onClick={() => {
                          applySuggestion(suggestion);
                          setSelectedSuggestion(null);
                        }}
                        style={{
                          padding: "0.5rem 1rem",
                          fontSize: "0.85rem",
                          backgroundColor: "#D5451B", // Different color
                          fontWeight: '500',
                          color: "white",
                          border: "none",
                          borderRadius: "0.25rem",
                          cursor: "pointer",
                          flex: 1,
                        }}
                      >
                        Apply to Code
                      </button>
                      <button
                        onClick={() => {
                          // Scroll to the line in the code view
                          const lineElements = document.querySelectorAll(
                            ".react-syntax-highlighter-line-number",
                          );
                          if (
                            lineElements &&
                            lineElements[suggestion.line - 1]
                          ) {
                            lineElements[suggestion.line - 1].scrollIntoView({
                              behavior: "smooth",
                              block: "center",
                            });
                          }
                        }}
                        style={{
                          padding: "0.5rem 1rem",
                          fontSize: "0.85rem",
                          backgroundColor: "#fff",
                          backgroundColor: "#dbdbdb", // Different color
                          fontWeight: '500',
                          fontFamily: 'Consolas, "Courier New", monospace',
                          color: "#333",
                          border: "1px solid #e1e1e1",
                          borderRadius: "0.25rem",
                          cursor: "pointer",
                          flex: 1,
                        }}
                      >
                        Go to Line
                      </button>
                    </div>
                  </div>
                )}
              </div>
            ))}
            {/* Diff View Information Banner (only when showing diffs) */}
{showDiff && lastChanges.added.length > 0 && (
  <div style={{ margin: "1rem", padding: "1rem", backgroundColor: "#f9fafb", borderRadius: "0.5rem", border: "1px solid #f9fafb", fontSize: "0.9rem", color: "black"}}>
    <div style={{ fontWeight: "600", marginBottom: "0.5rem" }}>Code Changes Applied</div>
    <div>
      <p style={{ margin: "0 0 0.5rem 0" }}>Added {lastChanges.added.length} new lines to the code</p>
      <p style={{ margin: "0" }}>Lines highlighted in green show the newly added code</p>
    </div>
  </div>
)}
          </div>

          {/* Custom scrollbar and animation styles */}
          <style>
            {`
              ::-webkit-scrollbar {
                width: 8px;
                height: 8px;
              }
              ::-webkit-scrollbar-track {
                background: #f1f1f1;
              }
              ::-webkit-scrollbar-thumb {
                background: #ccc;
                border-radius: 4px;
              }
              ::-webkit-scrollbar-thumb:hover {
                background: #aaa;
              }
              @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
              }
              .suggestions-panel {
                transition: width 0.3s ease;
              }
              .resize-handle-visible {
                background-color: rgba(231, 76, 50, 0.1);
              }
            `}
          </style>
        </div>
      )}
    </div>
  );
};

export default CodeEditor;
