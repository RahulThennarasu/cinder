// BitAutoOptimizer.js
import React, { useState, useEffect, useRef } from 'react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';
import './BitAutoOptimizer.css';
import BitOptimizerStateManager from './BitOptimizerStateManager';

const BitAutoOptimizer = ({ modelInfo, inSidePanel = true }) => {
  // State for code and chat
  const [originalCode, setOriginalCode] = useState('');
  const [currentCode, setCurrentCode] = useState('');
  const [messages, setMessages] = useState([]);
  const [userInput, setUserInput] = useState('');
  const [isConnected, setIsConnected] = useState(false);
  const [isOptimizing, setIsOptimizing] = useState(false);
  const [optimizationProgress, setOptimizationProgress] = useState(0);
  const [totalSteps, setTotalSteps] = useState(5);
  const [currentStep, setCurrentStep] = useState(0);
  const [currentOptimization, setCurrentOptimization] = useState('');
  const [codeHighlights, setCodeHighlights] = useState({ added: [], removed: [] });
  const [showDiff, setShowDiff] = useState(false);
  const [typingMessage, setTypingMessage] = useState(null);
  const [stateLoaded, setStateLoaded] = useState(false);
  
  // References
  const websocketRef = useRef(null);
  const chatEndRef = useRef(null);
  
  // Load saved state or initial code
  useEffect(() => {
    if (!stateLoaded) {
      loadSavedState();
    } else if (!currentCode && modelInfo) {
      fetchModelCode();
    }
  }, [modelInfo, stateLoaded]);

  // Periodically save state while component is mounted
  useEffect(() => {
    // Only save state if we have content to save
    if (stateLoaded && (currentCode || messages.length > 0)) {
      saveCurrentState();
      
      // Set up auto-save interval
      const saveInterval = setInterval(() => {
        saveCurrentState();
      }, 10000); // Save every 10 seconds
      
      return () => {
        clearInterval(saveInterval);
        // Save once more when unmounting
        saveCurrentState();
      };
    }
  }, [currentCode, messages, stateLoaded]);
  
  // Auto-scroll chat to bottom when new messages arrive
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, typingMessage]);
  
  // Handle WebSocket connection
  useEffect(() => {
    return () => {
      // Clean up WebSocket on component unmount
      if (websocketRef.current) {
        websocketRef.current.close();
      }
    };
  }, []);
  
  // Load saved state from storage
  const loadSavedState = () => {
    const savedState = BitOptimizerStateManager.loadState();
    
    if (savedState) {
      // Check if saved state matches current model
      const isSameModel = 
        savedState.modelInfo && 
        modelInfo && 
        savedState.modelInfo.id === modelInfo.id;
      
      if (isSameModel) {
        // Restore state if it's for the same model
        if (savedState.currentCode) {
          setCurrentCode(savedState.currentCode);
          setOriginalCode(savedState.currentCode);
        }
        
        if (savedState.messages && savedState.messages.length > 0) {
          setMessages(savedState.messages);
        }
        
        // Add a system message about restored state
        if (savedState.lastUpdated) {
          const lastUpdated = new Date(savedState.lastUpdated);
          const timeString = lastUpdated.toLocaleTimeString();
          const dateString = lastUpdated.toLocaleDateString();
          
          setMessages(prevMessages => [
            ...prevMessages,
            {
              role: 'system',
              content: `Restored previous session from ${timeString} on ${dateString}.`,
              timestamp: new Date()
            }
          ]);
        }
      } else {
        // Different model, fetch fresh code
        fetchModelCode();
      }
    } else {
      // No saved state, fetch fresh code
      fetchModelCode();
    }
    
    setStateLoaded(true);
  };
  
  // Save current state to storage
  const saveCurrentState = () => {
    BitOptimizerStateManager.saveState({
      currentCode,
      messages,
      modelInfo
    });
  };
  
  // Reset state
  const resetState = () => {
    BitOptimizerStateManager.clearState();
    fetchModelCode();
    setMessages([]);
    setTypingMessage(null);
    setIsOptimizing(false);
    setShowDiff(false);
    setCodeHighlights({ added: [], removed: [] });
  };
  
  // Fetch initial model code
  const fetchModelCode = async () => {
    try {
      const response = await fetch('/api/model-code');
      
      if (response.ok) {
        const data = await response.json();
        const code = data.code || '# No model code available';
        setOriginalCode(code);
        setCurrentCode(code);
        
        // Connect to WebSocket and wait for greeting
        connectWebSocket();
      } else {
        console.error('Error fetching model code');
      }
    } catch (error) {
      console.error('Error:', error);
    }
  };
  
  // Connect to WebSocket
  const connectWebSocket = () => {
  // Important: Use the correct WebSocket URL without '/api' prefix
  const wsUrl = `ws://${window.location.host}/ws/bit-optimizer`;
  websocketRef.current = new WebSocket(wsUrl);
  
  websocketRef.current.onopen = () => {
    console.log('WebSocket connection opened');
    setIsConnected(true);
    
    // Send initial message with API key for authentication
    websocketRef.current.send(JSON.stringify({
      action: "connect",
      api_key: getApiKey()
    }));
  };
  
  websocketRef.current.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      handleWebSocketMessage(data);
    } catch (error) {
      console.error("Error parsing message:", error);
    }
  };
  
  websocketRef.current.onerror = (error) => {
    console.error('WebSocket error:', error);
    simulateTyping("I encountered an error connecting to the optimization service. Please check your connection and try again.", 'assistant');
  };
  
  websocketRef.current.onclose = () => {
    console.log('WebSocket connection closed');
    setIsConnected(false);
    if (isOptimizing) {
      simulateTyping("The optimization process was interrupted. Would you like to resume?", 'assistant');
      setIsOptimizing(false);
    }
  };
};

const getApiKey = () => {
  // Try multiple sources for the API key
  
  // 1. Try window object (injected by server)
  if (window._api_key) {
    console.log("Found API key in window object");
    return window._api_key;
  }
  
  // 2. Try meta tag
  const metaApiKey = document.querySelector('meta[name="api-key"]');
  if (metaApiKey && metaApiKey.content) {
    console.log("Found API key in meta tag");
    return metaApiKey.content;
  }
  
  // 3. Try URL parameters
  const urlParams = new URLSearchParams(window.location.search);
  const urlApiKey = urlParams.get('api_key');
  if (urlApiKey) {
    console.log("Found API key in URL parameter");
    return urlApiKey;
  }
  
};

  
  // Simulate typing animation for messages
  const simulateTyping = (content, role, extras = {}) => {
    setTypingMessage({ role, content: '', fullContent: content, extras, charIndex: 0 });
    
    const typingInterval = setInterval(() => {
      setTypingMessage(prev => {
        if (!prev) return null;
        
        const newCharIndex = prev.charIndex + 2; // Increase speed by processing 2 chars at once
        if (newCharIndex > prev.fullContent.length) {
          clearInterval(typingInterval);
          
          // Add the complete message to the messages array
          setMessages(msgs => [...msgs, {
            role: prev.role,
            content: prev.fullContent,
            timestamp: new Date(),
            ...prev.extras
          }]);
          
          return null;
        }
        
        return {
          ...prev,
          content: prev.fullContent.substring(0, newCharIndex),
          charIndex: newCharIndex
        };
      });
    }, 10); // Typing speed - adjust as needed
  };
  
  // Generate a simple diff between two code strings
  const generateDiff = (oldCode, newCode) => {
    const oldLines = oldCode.split('\n');
    const newLines = newCode.split('\n');
    let diff = '';
    
    // Generate diff by comparing lines
    const maxLines = Math.max(oldLines.length, newLines.length);
    const changedLines = {
      added: [],
      removed: []
    };
    
    for (let i = 0; i < maxLines; i++) {
      if (i >= oldLines.length) {
        // Line was added
        diff += `+ ${newLines[i]}\n`;
        changedLines.added.push(i);
      } else if (i >= newLines.length) {
        // Line was removed
        diff += `- ${oldLines[i]}\n`;
        changedLines.removed.push(i);
      } else if (oldLines[i] !== newLines[i]) {
        // Line was changed
        diff += `- ${oldLines[i]}\n+ ${newLines[i]}\n`;
        changedLines.removed.push(i);
        changedLines.added.push(i);
      }
    }
    
    // Update code highlights for visualization
    setCodeHighlights(changedLines);
    setShowDiff(true);
    
    return diff || "No visible changes detected.";
  };
  
  // Handle user input submission
  const handleSendMessage = () => {
  if (!userInput.trim()) return;
  
  // Add user message to chat
  setMessages(prev => [...prev, {
    role: 'user',
    content: userInput,
    timestamp: new Date()
  }]);
  
  // Check if we're already connected
  if (!websocketRef.current || websocketRef.current.readyState !== WebSocket.OPEN) {
    connectWebSocket();
    
    // Wait for connection to open before sending
    setTimeout(() => {
      if (websocketRef.current && websocketRef.current.readyState === WebSocket.OPEN) {
        sendMessageToWebSocket();
      } else {
        console.log("WebSocket not connected yet, waiting longer...");
        // Try again in 1 second
        setTimeout(() => {
          if (websocketRef.current && websocketRef.current.readyState === WebSocket.OPEN) {
            sendMessageToWebSocket();
          } else {
            simulateTyping("Connection to the optimization service failed. Please try again later.", 'assistant');
          }
        }, 1000);
      }
    }, 500);
  } else {
    // WebSocket already connected, send immediately
    sendMessageToWebSocket();
  }
  
  // Helper function to send the message
  function sendMessageToWebSocket() {
    websocketRef.current.send(JSON.stringify({
      action: "chat",
      query: userInput,
      code: currentCode,
      framework: (modelInfo && modelInfo.framework) || 'pytorch',
      api_key: getApiKey() // Add API key to request
    }));
    setIsOptimizing(true);
  }
  
  // Clear input
  setUserInput('');
  
  // Save state after sending message
  setTimeout(saveCurrentState, 500);
};

// Modified startOptimization function
const startOptimization = () => {
  if (isOptimizing) return;
  
  // Add message
  setMessages(prev => [...prev, {
    role: 'user',
    content: 'Start auto optimization',
    timestamp: new Date()
  }]);
  
  // Initialize WebSocket if not already connected
  if (!websocketRef.current || websocketRef.current.readyState !== WebSocket.OPEN) {
    connectWebSocket();
  }
  
  // Set optimization state
  setIsOptimizing(true);
  
  // Initialize progress tracking
  setTotalSteps(5);
  setCurrentStep(0);
  setOptimizationProgress(0);
  
  // Send optimization request with API key
  if (websocketRef.current && websocketRef.current.readyState === WebSocket.OPEN) {
    websocketRef.current.send(JSON.stringify({
      action: "optimize",
      code: currentCode,
      framework: (modelInfo && modelInfo.framework) || 'pytorch',
      api_key: getApiKey() // Add API key to request
    }));
  }
  
  // Save state after starting optimization
  setTimeout(saveCurrentState, 500);
};
    
  // Handle different types of WebSocket messages
  const handleWebSocketMessage = (data) => {
    console.log("Processing message type:", data.type);
    
    // Extract optimization name for status messages
    if (data.type === 'status' && data.message && data.message.includes('Generating code changes for')) {
      const optimizationName = data.message.replace('Generating code changes for ', '').replace('...', '');
      setCurrentOptimization(optimizationName);
      
      // Update progress counter for each new optimization
      const newStep = currentStep + 1;
      setCurrentStep(newStep);
      setOptimizationProgress((newStep / totalSteps) * 100);
    }
    
    // Process message based on type
    switch (data.type) {
      case 'greeting':
        simulateTyping(data.message, 'assistant');
        break;
        
      case 'status':
        simulateTyping(data.message, 'assistant');
        break;
        
      case 'optimization':
        console.log("Received optimization data:", data);
        
        // Display optimization title and description
        if (data.optimization) {
          simulateTyping(`**${data.optimization.title}**\n\n${data.optimization.description}`, 'assistant');
        }
        
        // Apply code changes if available
        if (data.changes && data.changes.updated_code) {
          const oldCode = currentCode;
          setCurrentCode(data.changes.updated_code);
          
          // Generate diff for visualization
          const diff = generateDiff(oldCode, data.changes.updated_code);
          
          // Create message about changes
          let changeMessage = `I've applied this optimization to your code.`;
          if (data.changes.changes_summary) {
            changeMessage += `\n\n**Changes made:** ${data.changes.changes_summary}`;
          }
          
          simulateTyping(changeMessage, 'assistant', {
            codeChange: {
              oldCode: oldCode,
              newCode: data.changes.updated_code,
              diff: diff
            }
          });
          
          // Save state after code update
          setTimeout(saveCurrentState, 500);
        }
        break;
        
      case 'explanation':
        simulateTyping(`**Explanation:**\n${data.message}`, 'assistant');
        break;
        
      case 'complete':
        setIsOptimizing(false);
        simulateTyping(data.message || "Optimization process completed successfully!", 'assistant');
        // Save state after completion
        setTimeout(saveCurrentState, 500);
        break;
        
      case 'error':
        setIsOptimizing(false);
        simulateTyping(`Error: ${data.message}`, 'assistant');
        break;
        
      default:
        console.log('Unknown message type:', data.type);
    }
  };

  // Render the code panel with line highlighting
  const renderCodePanel = () => {
    return (
      <div className="code-panel">
        <div className="code-header">
          <h3>model_code</h3>
          {showDiff && (
            <div className="diff-indicator">
              <span className="added-indicator">+{codeHighlights.added.length} lines added</span>
              <span className="removed-indicator">-{codeHighlights.removed.length} lines removed</span>
            </div>
          )}
          <div className="code-actions">
            <button 
              className="reset-button"
              onClick={resetState}
              disabled={isOptimizing}
              title="Reset to original code"
            >
              Reset
            </button>
          </div>
        </div>
        <div className="code-container">
          <SyntaxHighlighter
            language="python"
            style={vscDarkPlus}
            showLineNumbers={true}
            wrapLines={true}
            customStyle={{
              margin: 0,
              padding: 0,
              backgroundColor: '#1e1e1e',
              overflow: 'auto'
            }}
            codeTagProps={{
              style: {
                display: 'block',
                paddingBottom: 0,
                marginBottom: 0
              }
            }}
            lineProps={lineNumber => {
              // Highlight added/removed lines
              const style = {};
              
              if (showDiff) {
                if (codeHighlights.added.includes(lineNumber - 1)) {
                  style.backgroundColor = 'rgba(0, 255, 0, 0.1)';
                  style.borderLeft = '3px solid #00cc00';
                } else if (codeHighlights.removed.includes(lineNumber - 1)) {
                  style.backgroundColor = 'rgba(255, 0, 0, 0.1)';
                  style.borderLeft = '3px solid #cc0000';
                }
              }
              
              return { style };
            }}
          >
            {currentCode}
          </SyntaxHighlighter>
        </div>
      </div>
    );
  };
  
  // Render the chat panel
  const renderChatPanel = () => {
    return (
      <div className="chat-panel">
        <div className="chat-header">
          <h3></h3>
          {isOptimizing && (
            <div className="optimization-progress">
              <div className="progress-info">
                <span className="progress-step">
                  Step {currentStep} of {totalSteps}: {currentOptimization || 'Analyzing'}
                </span>
                <span className="progress-percentage">
                  {Math.round(optimizationProgress)}%
                </span>
              </div>
              <div className="progress-bar">
                <div 
                  className="progress-fill"
                  style={{ width: `${optimizationProgress}%` }}
                ></div>
              </div>
            </div>
          )}
        </div>
        
        <div className="chat-messages">
          {messages.map((message, index) => (
            <div 
              key={index} 
              className={`message ${message.role === 'user' ? 'user-message' : message.role === 'system' ? 'system-message' : 'assistant-message'}`}
            >
              <div className="message-content">
                {/* Parse message content for markdown-like formatting */}
                {message.content.split('\n').map((line, i) => {
                  // Handle bold text
                  if (line.startsWith('**') && line.endsWith('**')) {
                    return <h4 key={i} className="message-heading">{line.slice(2, -2)}</h4>;
                  }
                  
                  // Handle bold within text
                  const boldPattern = /\*\*(.*?)\*\*/g;
                  if (boldPattern.test(line)) {
                    const parts = line.split(boldPattern);
                    const elements = [];
                    
                    let i = 0;
                    for (let j = 0; j < parts.length; j++) {
                      if (j % 2 === 0) {
                        // Regular text
                        if (parts[j]) elements.push(<span key={`${i}-${j}`}>{parts[j]}</span>);
                      } else {
                        // Bold text
                        elements.push(<strong key={`${i}-${j}`}>{parts[j]}</strong>);
                      }
                    }
                    
                    return <p key={i}>{elements}</p>;
                  }
                  
                  // Regular paragraph
                  return <p key={i}>{line}</p>;
                })}
                
                {message.codeChange && (
                  <div className="code-change">
                    <div className="code-diff">
                      <SyntaxHighlighter
                        language="diff"
                        style={vscDarkPlus}
                        showLineNumbers={false}
                      >
                        {message.codeChange.diff}
                      </SyntaxHighlighter>
                    </div>
                  </div>
                )}
              </div>
              <div className="message-time">
                {message.timestamp && typeof message.timestamp.toLocaleTimeString === 'function' 
                  ? message.timestamp.toLocaleTimeString() 
                  : new Date().toLocaleTimeString()}
              </div>
            </div>
          ))}
          
          {/* Typing animation message */}
          {typingMessage && (
            <div className={`message ${typingMessage.role === 'user' ? 'user-message' : typingMessage.role === 'system' ? 'system-message' : 'assistant-message'}`}>
              <div className="message-content">
                {typingMessage.content.split('\n').map((line, i) => {
                  // Handle bold text
                  if (line.startsWith('**') && line.endsWith('**')) {
                    return <h4 key={i} className="message-heading">{line.slice(2, -2)}</h4>;
                  }
                  
                  // Handle bold within text
                  const boldPattern = /\*\*(.*?)\*\*/g;
                  if (boldPattern.test(line)) {
                    const parts = line.split(boldPattern);
                    const elements = [];
                    
                    let i = 0;
                    for (let j = 0; j < parts.length; j++) {
                      if (j % 2 === 0) {
                        // Regular text
                        if (parts[j]) elements.push(<span key={`${i}-${j}`}>{parts[j]}</span>);
                      } else {
                        // Bold text
                        elements.push(<strong key={`${i}-${j}`}>{parts[j]}</strong>);
                      }
                    }
                    
                    return <p key={i}>{elements}</p>;
                  }
                  
                  // Regular paragraph
                  return <p key={i}>{line}</p>;
                })}
                <span className="typing-cursor"></span>
              </div>
            </div>
          )}
          
          <div ref={chatEndRef} />
        </div>
        
        <div className="chat-input">
          <input
            type="text"
            value={userInput}
            onChange={(e) => setUserInput(e.target.value)}
            onKeyPress={(e) => {
              if (e.key === 'Enter') {
                handleSendMessage();
              }
            }}
            placeholder="Ask about improving your model..."
            className="chat-input-field"
            disabled={isOptimizing}
          />
          <button 
            className="send-button"
            onClick={handleSendMessage}
            disabled={isOptimizing || !userInput.trim()}
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M22 2L11 13M22 2L15 22L11 13M11 13L2 9L22 2"></path>
            </svg>
          </button>
        </div>
      </div>
    );
  };
  
  // Render based on view mode
  return (
    <div className="bit-auto-optimizer split-view">
      <div className="split-container">
        <div className="split-code">
          {renderCodePanel()}
        </div>
        <div className="split-chat">
          {renderChatPanel()}
        </div>
      </div>
    </div>
  );
};

export default BitAutoOptimizer;