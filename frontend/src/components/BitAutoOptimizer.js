// Updated BitAutoOptimizer.js - Improved UI with better code display and progress tracking

import React, { useState, useEffect, useRef } from 'react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';
import './BitAutoOptimizer.css';

const BitAutoOptimizer = ({ modelInfo, inSidePanel = true }) => {
  // State for code and chat
  const [originalCode, setOriginalCode] = useState('');
  const [currentCode, setCurrentCode] = useState('');
  const [messages, setMessages] = useState([]);
  const [isConnected, setIsConnected] = useState(false);
  const [isOptimizing, setIsOptimizing] = useState(false);
  const [optimizationProgress, setOptimizationProgress] = useState(0);
  const [totalSteps, setTotalSteps] = useState(5); // Default to 5 steps
  const [currentStep, setCurrentStep] = useState(0);
  const [currentOptimization, setCurrentOptimization] = useState('');
  const [viewMode, setViewMode] = useState('split'); // 'split', 'code', 'chat'
  const [typingMessage, setTypingMessage] = useState(null);
  const [codeHighlights, setCodeHighlights] = useState({ added: [], removed: [] });
  const [showDiff, setShowDiff] = useState(false);
  
  // References
  const websocketRef = useRef(null);
  const chatEndRef = useRef(null);
  
  // Load initial code
  useEffect(() => {
    fetchModelCode();
  }, [modelInfo]);
  
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
  
  // Fetch initial model code
  const fetchModelCode = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/model-code');
      
      if (response.ok) {
        const data = await response.json();
        const code = data.code || '# No model code available';
        setOriginalCode(code);
        setCurrentCode(code);
        
        // Add initial message with typing animation
        simulateTyping("Hello! I'm Bit, your ML optimization assistant powered by Gemini. I've loaded your model code and I'm ready to help you improve it with AI-driven optimizations. Would you like me to start analyzing your model?", 'assistant');
      } else {
        console.error('Error fetching model code');
      }
    } catch (error) {
      console.error('Error:', error);
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
  
  // Start the optimization process
  const startOptimization = () => {
    if (isOptimizing) return;
    
    // Add message
    addMessage('user', 'Yes, please start optimizing my model automatically.');
    
    // Connect to WebSocket
    const wsUrl = `ws://localhost:8000/ws/bit-optimizer`;
    websocketRef.current = new WebSocket(wsUrl);
    
    // Set up event handlers
    websocketRef.current.onopen = () => {
      console.log('WebSocket connection opened');
      setIsConnected(true);
      setIsOptimizing(true);
      
      // Initialize progress tracking
      setTotalSteps(5); // Assuming 5 optimizations as default
      setCurrentStep(0);
      setOptimizationProgress(0);
      
      // Send initial data
      websocketRef.current.send(JSON.stringify({
        action: "optimize",
        code: currentCode,
        framework: (modelInfo && modelInfo.framework) || 'pytorch'
      }));
      
      simulateTyping("I'll analyze your model using the Gemini API and suggest optimizations. I'll explain each improvement as I implement it.", 'assistant');
    };
    
    websocketRef.current.onmessage = (event) => {
      console.log("Received message:", event.data);
      
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
      setIsOptimizing(false);
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
        }
        
        // Add explanation if available
        if (data.explanation) {
          simulateTyping(`**Explanation:**\n${data.explanation}`, 'assistant');
        }
        break;
        
      case 'complete':
        setIsOptimizing(false);
        simulateTyping(data.message || "Optimization process completed successfully!", 'assistant');
        break;
        
      case 'error':
        setIsOptimizing(false);
        simulateTyping(`Error: ${data.message}`, 'assistant');
        break;
        
      default:
        console.log('Unknown message type:', data.type);
    }
  };
  
  // Add a message to the chat (without typing animation)
  const addMessage = (role, content, extras = {}) => {
    setMessages(prev => [
      ...prev,
      {
        role,
        content,
        timestamp: new Date(),
        ...extras
      }
    ]);
  };
  
  // Render the code panel with line highlighting
  const renderCodePanel = () => {
    return (
      <div className="code-panel">
        <div className="code-header">
          <h3>Model Code</h3>
          {showDiff && (
            <div className="diff-indicator">
              <span className="added-indicator">+{codeHighlights.added.length} lines added</span>
              <span className="removed-indicator">-{codeHighlights.removed.length} lines removed</span>
            </div>
          )}
          <div className="code-actions">
            <button 
              className={`view-button ${viewMode === 'split' ? 'active' : ''}`}
              onClick={() => setViewMode('split')}
            >
              Split View
            </button>
            <button 
              className={`view-button ${viewMode === 'code' ? 'active' : ''}`}
              onClick={() => setViewMode('code')}
            >
              Code Only
            </button>
            <button 
              className={`view-button ${viewMode === 'chat' ? 'active' : ''}`}
              onClick={() => setViewMode('chat')}
            >
              Chat Only
            </button>
          </div>
        </div>
        <div className="code-container">
          <SyntaxHighlighter
            language="python"
            style={vscDarkPlus}
            showLineNumbers={true}
            wrapLines={true}
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
          <h3>Bit AI Optimizer</h3>
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
              className={`message ${message.role === 'user' ? 'user-message' : 'assistant-message'}`}
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
                {message.timestamp.toLocaleTimeString()}
              </div>
            </div>
          ))}
          
          {/* Typing animation message */}
          {typingMessage && (
            <div className={`message ${typingMessage.role === 'user' ? 'user-message' : 'assistant-message'}`}>
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
          {!isOptimizing ? (
            <button 
              className="start-button"
              onClick={startOptimization}
              disabled={isOptimizing}
            >
              Start Auto Optimization
            </button>
          ) : (
            <button 
              className="stop-button"
              onClick={() => {
                if (websocketRef.current) {
                  websocketRef.current.close();
                }
                setIsOptimizing(false);
                simulateTyping("Optimization process paused. Would you like to continue?", 'assistant');
              }}
            >
              Pause Optimization
            </button>
          )}
        </div>
      </div>
    );
  };
  
  // Render based on view mode
  if (viewMode === 'code') {
    return (
      <div className="bit-auto-optimizer code-only">
        {renderCodePanel()}
      </div>
    );
  } else if (viewMode === 'chat') {
    return (
      <div className="bit-auto-optimizer chat-only">
        {renderChatPanel()}
      </div>
    );
  } else {
    // Split view
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
  }
};

export default BitAutoOptimizer;