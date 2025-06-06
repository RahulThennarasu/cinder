// Updated BitAutoOptimizer.js - removing mock data and using real API exclusively

import React, { useState, useEffect, useRef } from 'react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';
import './BitAutoOptimizer.css';

const BitAutoOptimizer = ({ modelInfo }) => {
  // State for code and chat
  const [originalCode, setOriginalCode] = useState('');
  const [currentCode, setCurrentCode] = useState('');
  const [messages, setMessages] = useState([]);
  const [isConnected, setIsConnected] = useState(false);
  const [isOptimizing, setIsOptimizing] = useState(false);
  const [optimizationProgress, setOptimizationProgress] = useState(0);
  const [totalSteps, setTotalSteps] = useState(0);
  const [currentStep, setCurrentStep] = useState(0);
  const [viewMode, setViewMode] = useState('split'); // 'split', 'code', 'chat'
  const [typingMessage, setTypingMessage] = useState(null);
  
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
        
        const newCharIndex = prev.charIndex + 1;
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
      setIsConnected(true);
      setIsOptimizing(true);
      
      // Send initial data
      websocketRef.current.send(JSON.stringify({
        code: currentCode,
        modelInfo: modelInfo || { framework: 'pytorch' }
      }));
      
      simulateTyping("I'll analyze your model using the Gemini API and suggest optimizations. I'll explain each improvement as I implement it.", 'assistant');
    };
    
    websocketRef.current.onmessage = (event) => {
      const data = JSON.parse(event.data);
      handleWebSocketMessage(data);
    };
    
    websocketRef.current.onerror = (error) => {
      console.error('WebSocket error:', error);
      simulateTyping("I encountered an error connecting to the optimization service. Please check your connection and try again.", 'assistant');
      setIsOptimizing(false);
    };
    
    websocketRef.current.onclose = () => {
      setIsConnected(false);
      if (isOptimizing) {
        simulateTyping("The optimization process was interrupted. Would you like to resume?", 'assistant');
        setIsOptimizing(false);
      }
    };
  };
  
  // Handle different types of WebSocket messages
  const handleWebSocketMessage = (data) => {
    const messageType = data.type;
    
    switch (messageType) {
      case 'status':
        simulateTyping(data.message, 'assistant');
        break;
        
      case 'plan':
        setTotalSteps(data.steps);
        simulateTyping(data.message, 'assistant');
        break;
        
      case 'progress':
        setOptimizationProgress(data.percentage);
        setCurrentStep(data.current);
        break;
        
      case 'analyzing':
        simulateTyping(data.message, 'assistant');
        break;
        
      case 'explanation':
        let explanationText = `**${data.title}**\n\n${data.description}`;
        if (data.benefit) {
          explanationText += `\n\n**Expected benefit:** ${data.benefit}`;
        }
        simulateTyping(explanationText, 'assistant');
        break;
        
      case 'code_change':
        setCurrentCode(data.newCode);
        
        let changeMessage = "I've applied this optimization. Here's what changed:";
        if (data.changesSummary) {
          changeMessage += `\n\n**Changes made:** ${data.changesSummary}`;
        }
        
        simulateTyping(changeMessage, 'assistant', {
          codeChange: {
            oldCode: data.oldCode,
            newCode: data.newCode,
            diff: data.diff
          }
        });
        break;
        
      case 'insights':
        simulateTyping(data.message, 'assistant');
        break;
        
      case 'completed':
        setIsOptimizing(false);
        simulateTyping(data.message, 'assistant');
        break;
        
      case 'error':
        setIsOptimizing(false);
        simulateTyping(`Error: ${data.message}`, 'assistant');
        break;
        
      default:
        console.log('Unknown message type:', messageType);
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
  
  // Render the code panel
  const renderCodePanel = () => {
    return (
      <div className="code-panel">
        <div className="code-header">
          <h3>Model Code</h3>
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
              <div className="progress-bar">
                <div 
                  className="progress-fill"
                  style={{ width: `${optimizationProgress}%` }}
                ></div>
              </div>
              <div className="progress-text">
                Step {currentStep} of {totalSteps}
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
                {message.content.split('\n').map((line, i) => (
                  <p key={i}>{line}</p>
                ))}
                
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
                {typingMessage.content.split('\n').map((line, i) => (
                  <p key={i}>{line}</p>
                ))}
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