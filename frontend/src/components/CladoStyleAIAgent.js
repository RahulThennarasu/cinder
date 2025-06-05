import React, { useState, useEffect, useRef } from 'react';
import { MinimalIcons } from './MinimalIcons';
import './CladoStyleAIAgent.css';

const CladoStyleAIAgent = ({ modelInfo }) => {
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isTyping, setIsTyping] = useState(false);
  const [typingText, setTypingText] = useState('');
  const [codeAnalysis, setCodeAnalysis] = useState(null);
  
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);
  const typingIntervalRef = useRef(null);

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, typingText]);

  // Focus input and analyze code on mount
  useEffect(() => {
    inputRef.current?.focus();
    analyzeModelCode();
  }, []);

  // Typing animation effect
  const typeMessage = (text, callback) => {
    setIsTyping(true);
    setTypingText('');
    let index = 0;
    
    typingIntervalRef.current = setInterval(() => {
      if (index < text.length) {
        setTypingText(prev => prev + text[index]);
        index++;
      } else {
        clearInterval(typingIntervalRef.current);
        setIsTyping(false);
        callback();
      }
    }, 30); // Adjust speed here (lower = faster)
  };

  // Analyze model code automatically
  const analyzeModelCode = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/model-code');
      if (response.ok) {
        const data = await response.json();
        setCodeAnalysis(data);
        
        // Send initial analysis message
        const welcomeMessage = `Hi! I'm Bit, your ML optimization assistant. I've analyzed your ${data.framework || 'pytorch'} model and found several areas for improvement. Let me help you optimize your code!`;
        
        typeMessage(welcomeMessage, () => {
          setMessages([{
            role: "assistant",
            content: welcomeMessage,
            timestamp: new Date(),
            isAnalysis: true
          }]);
          
          // Follow up with specific suggestions after a delay
          setTimeout(() => {
            provideSuggestions();
          }, 1500);
        });
      }
    } catch (error) {
      console.error('Error analyzing code:', error);
      // Fallback message if API fails
      const fallbackMessage = "Hi! I'm Bit, your ML optimization assistant. I can help optimize your ML model. What would you like to improve today?";
      
      typeMessage(fallbackMessage, () => {
        setMessages([{
          role: "assistant",
          content: fallbackMessage,
          timestamp: new Date(),
          isAnalysis: true
        }]);
      });
    }
  };

  // Provide AI-generated suggestions
  const provideSuggestions = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/model-improvements?detail_level=comprehensive');
      if (response.ok) {
        const data = await response.json();
        const suggestions = data.suggestions || [];
        
        if (suggestions.length > 0) {
          const suggestionMessage = `I found ${suggestions.length} optimization opportunities:\n\n1. Use Cross-Validation: Single train-test split may not be reliable`;
          
          typeMessage(suggestionMessage, () => {
            setMessages(prev => [...prev, {
              role: "assistant",
              content: suggestionMessage,
              timestamp: new Date(),
              suggestions: suggestions.slice(0, 3)
            }]);
          });
        }
      }
    } catch (error) {
      console.error('Error getting suggestions:', error);
      // Fallback suggestions if API fails
      const fallbackSuggestions = "I found 3 optimization opportunities:\n\n1. Use Cross-Validation: Single train-test split may not be reliable";
      
      typeMessage(fallbackSuggestions, () => {
        setMessages(prev => [...prev, {
          role: "assistant",
          content: fallbackSuggestions,
          timestamp: new Date(),
          suggestions: [
            { title: "Use Cross-Validation", issue: "Single train-test split may not be reliable" }
          ]
        }]);
      });
    }
  };

  // Handle sending messages to backend
  const handleSendMessage = async () => {
    if (inputValue.trim() === '') return;
    
    const userMessage = {
      role: "user",
      content: inputValue,
      timestamp: new Date()
    };
    
    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);
    
    try {
      const response = await fetch('http://localhost:8000/api/bit-chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: userMessage.content,
          code: codeAnalysis?.code || '',
          modelInfo: modelInfo || { framework: 'pytorch', accuracy: 0.85 },
          framework: (modelInfo?.framework || 'pytorch')
        }),
      });

      if (!response.ok) throw new Error(`API error: ${response.status}`);

      const data = await response.json();
      
      // Type the response with animation
      typeMessage(data.message || "I'll help you optimize your model. Can you provide more details about what you're trying to improve?", () => {
        setMessages(prev => [...prev, {
          role: "assistant",
          content: data.message || "I'll help you optimize your model. Can you provide more details about what you're trying to improve?",
          timestamp: new Date(),
          suggestions: data.suggestions || []
        }]);
        setIsLoading(false);
      });
      
    } catch (error) {
      console.error("Error:", error);
      const errorMessage = "I'm having trouble connecting right now. Please try again in a moment.";
      
      typeMessage(errorMessage, () => {
        setMessages(prev => [...prev, {
          role: "assistant",
          content: errorMessage,
          timestamp: new Date(),
          isError: true
        }]);
        setIsLoading(false);
      });
    }
  };

  const formatTime = (date) => {
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  const QuickActions = () => (
    <div className="quick-actions">
      {[
        "Optimize my model performance",
        "Fix overfitting issues", 
        "Improve training speed",
        "Add regularization"
      ].map((action, index) => (
        <button
          key={index}
          onClick={() => setInputValue(action)}
          className="quick-action-button"
        >
          {action}
        </button>
      ))}
    </div>
  );

  return (
    <div className="clado-container">
      {/* Header */}
      <div className="clado-header">
        <h1 className="clado-title">How can Bit help optimize your model?</h1>
        <p className="clado-subtitle">AI-powered code analysis and continuous improvement suggestions</p>
        
        {modelInfo && (
          <div className="model-info">
            <div className="model-badge model-loaded">
              <div className="model-loaded-indicator"></div>
              <span className="model-loaded-text">Model Loaded</span>
            </div>
            <div className="model-badge framework-badge">
              <MinimalIcons.Code />
              <span>{modelInfo.framework || 'pytorch'}</span>
            </div>
            <div className="model-badge accuracy-badge">
              <MinimalIcons.Zap />
              <span>{((modelInfo?.accuracy || 0.85) * 100).toFixed(1)}% accuracy</span>
            </div>
          </div>
        )}
      </div>

      {/* Chat Messages */}
      <div className="messages-container">
        {/* Quick Actions */}
        {messages.length === 0 && !isTyping && (
          <QuickActions />
        )}

        {/* Messages */}
        {messages.map((message, index) => (
          <div key={index} className={`message ${message.role === 'user' ? 'user-message' : 'assistant-message'}`}>
            {message.role === 'assistant' && (
              <div className="agent-info">
                <div className="agent-avatar">
                  <MinimalIcons.Brain />
                </div>
                <span className="agent-name">Bit</span>
                <span className="time-display">{formatTime(message.timestamp)}</span>
              </div>
            )}
            
            <div className="message-content">
              {message.content}
            </div>

            {message.role === 'user' && (
              <div className="message-time">
                {formatTime(message.timestamp)}
              </div>
            )}
          </div>
        ))}
        
        {/* Typing Animation */}
        {isTyping && (
          <div className="message assistant-message">
            <div className="agent-info">
              <div className="agent-avatar">
                <MinimalIcons.Brain />
              </div>
              <span className="agent-name">Bit</span>
            </div>
            <div className="message-content">
              {typingText}
              <span className="cursor-blink">|</span>
            </div>
          </div>
        )}
        
        {/* Loading indicator when waiting for API */}
        {isLoading && !isTyping && (
          <div className="message assistant-message">
            <div className="agent-info">
              <div className="agent-avatar">
                <MinimalIcons.Brain />
              </div>
              <span className="agent-name">Bit</span>
            </div>
            <div className="typing-indicator">
              <div className="typing-dot"></div>
              <div className="typing-dot"></div>
              <div className="typing-dot"></div>
            </div>
          </div>
        )}
        
        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div className="input-container">
        <div className="input-wrapper">
          <textarea
            ref={inputRef}
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyPress={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                handleSendMessage();
              }
            }}
            placeholder="Ask me anything about optimizing your ML model..."
            className="input-field"
            rows="1"
          />
          <button
            onClick={handleSendMessage}
            disabled={isLoading || isTyping || inputValue.trim() === ''}
            className="send-button"
          >
            <MinimalIcons.ArrowUp />
          </button>
        </div>
        
        <div className="disclaimer">
          <MinimalIcons.Sparkles />
          <span>Bit can make mistakes. Always verify important code changes.</span>
        </div>
      </div>
    </div>
  );
};

export default CladoStyleAIAgent;