import React, { useState, useRef, useEffect } from 'react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';

const BitChatInterface = ({ 
  code, 
  modelInfo, 
  onApplyFix, 
  messages,
  setMessages,
  inputValue,
  setInputValue,
  isLoading,
  setIsLoading 
}) => {
  
  const [copiedCode, setCopiedCode] = useState(null);  // Add this state for tracking copied code
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  const cleanCodeForDisplay = (code) => {
  if (!code) return '';
  
  // Remove markdown code blocks and language indicators
  let cleanedCode = code.replace(/```python\n?/g, '').replace(/```\n?/g, '');
  
  // Check if code still contains escaped newlines
  if (cleanedCode.includes('\\n')) {
    // Replace escaped newlines with actual newlines
    cleanedCode = cleanedCode.replace(/\\n/g, '\n');
  }
  
  return cleanedCode.trim();
};
  
  // Automatically scroll to the bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  useEffect(() => {
  // Focus the input field and scroll to bottom when the chat view becomes active
  inputRef.current?.focus();
  messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
}, []); // Empty dependency array means this runs once when component mounts

  // Focus input field when component mounts
  useEffect(() => {
    inputRef.current?.focus();
  }, []);
  
  const handleSendMessage = async () => {
    if (inputValue.trim() === '') return;
    
    // Add user message
    const userMessage = {
      role: "user",
      content: inputValue,
      timestamp: new Date()
    };
    
    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);
    
    try {
      // Call your API endpoint
      const response = await fetch('http://localhost:8000/api/bit-chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: userMessage.content,
          code: code,
          modelInfo: modelInfo,
          framework: modelInfo?.framework || 'pytorch'
        }),
      });

      if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
      }

      const data = await response.json();
      
      // Add assistant message with response data
      setMessages(prev => [...prev, {
        role: "assistant",
        content: data.message,
        timestamp: new Date(),
        suggestions: data.suggestions || []
      }]);
      
    } catch (error) {
      console.error("Error getting Bit response:", error);
      // Add error message
      setMessages(prev => [...prev, {
        role: "assistant",
        content: "Sorry, I encountered an error analyzing your code. Please try again.",
        timestamp: new Date(),
        isError: true
      }]);
    } finally {
      setIsLoading(false);
    }
  };
  
  // Function to copy code to clipboard
  const copyToClipboard = (code, suggestionIndex) => {
    navigator.clipboard.writeText(code);
    setCopiedCode(suggestionIndex);
    
    // Reset after 2 seconds
    setTimeout(() => {
      setCopiedCode(null);
    }, 2000);
  };
  
  // Format timestamp
  const formatTime = (date) => {
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };
  
  return (
    <div className="bit-chat-container" style={{
      display: 'flex',
      flexDirection: 'column',
      height: '100%',
      backgroundColor: '#f9fafb',
      borderRadius: '8px',
      overflow: 'hidden'
    }}>
      {/* Chat header */}
      <div style={{
        padding: '12px 16px',
        borderBottom: '1px solid #e5e7eb',
        backgroundColor: 'white',
        display: 'flex',
        alignItems: 'center',
        gap: '12px'
      }}>
        <div style={{
          width: '24px',
          height: '24px',
          position: 'relative'
        }}>
          <div style={{
            width: '22px',
            height: '22px',
            backgroundColor: '#FF9B45',
            borderRadius: '6px',
            position: 'absolute',
            bottom: '0',
            right: '0',
            opacity: '0.6'
          }}></div>
          <div style={{
            width: '22px',
            height: '22px',
            backgroundColor: '#D5451B',
            borderRadius: '6px',
            position: 'absolute',
            top: '0',
            left: '0'
          }}></div>
        </div>
        <h3 style={{
          margin: '0',
          fontSize: '16px',
          fontWeight: '600',
          color: '#111827'
        }}>Chat with Bit</h3>
      </div>
      
      {/* Messages container */}
      <div style={{
        flex: '1',
        overflowY: 'auto',
        padding: '16px',
        display: 'flex',
        flexDirection: 'column',
        gap: '16px'
      }}>
        {messages.map((message, index) => (
          <div key={index} style={{
            display: 'flex',
            flexDirection: 'column',
            alignSelf: message.role === 'user' ? 'flex-end' : 'flex-start',
            maxWidth: '80%'
          }}>
            <div style={{
              padding: '12px 16px',
              borderRadius: '12px',
              backgroundColor: message.role === 'user' ? '#D5451B' : 'white',
              color: message.role === 'user' ? 'white' : '#111827',
              boxShadow: '0 1px 2px rgba(0, 0, 0, 0.05)',
              border: message.role === 'assistant' ? '1px solid #e5e7eb' : 'none',
              whiteSpace: 'pre-wrap'
            }}>
              {message.content}
            </div>
            
            {/* Timestamp */}
            <span style={{
              fontSize: '12px',
              color: '#6b7280',
              marginTop: '4px',
              alignSelf: message.role === 'user' ? 'flex-end' : 'flex-start'
            }}>
              {formatTime(message.timestamp)}
            </span>
            
            {/* Suggestions if any */}
            {message.role === 'assistant' && message.suggestions && message.suggestions.length > 0 && (
              <div style={{
                marginTop: '12px',
                display: 'flex',
                flexDirection: 'column',
                gap: '12px'
              }}>
                {message.suggestions.map((suggestion, idx) => (
                  <div key={idx} style={{
                    backgroundColor: 'white',
                    border: '1px solid #e5e7eb',
                    borderRadius: '8px',
                    padding: '16px',
                    boxShadow: '0 1px 2px rgba(0, 0, 0, 0.05)'
                  }}>
                    <div style={{
                      display: 'flex',
                      justifyContent: 'space-between',
                      alignItems: 'center',
                      marginBottom: '8px'
                    }}>
                      <h4 style={{
                        margin: '0',
                        fontSize: '14px',
                        fontWeight: '600',
                        color: '#111827'
                      }}>{suggestion.title}</h4>
                      <span style={{
                        fontSize: '12px',
                        color: '#6b7280',
                        backgroundColor: '#f3f4f6',
                        padding: '2px 6px',
                        borderRadius: '4px'
                      }}>Line {suggestion.lineNumber}</span>
                    </div>
                    
                    <p style={{
                      margin: '0 0 12px 0',
                      fontSize: '13px',
                      color: '#4b5563'
                    }}>{suggestion.description}</p>
                    
                    {/* Format code properly with syntax highlighting */}
                    <div style={{
                      backgroundColor: '#1e1e1e',
                      borderRadius: '6px',
                      overflow: 'hidden',
                      marginBottom: '12px'
                    }}>
                      <SyntaxHighlighter
                        language="python"
                        style={vscDarkPlus}
                        customStyle={{
                          margin: 0,
                          padding: '12px',
                          fontSize: '13px',
                          lineHeight: '1.4'
                        }}
                      >
                        {/* Clean up code by removing markdown code block syntax */}
                          {cleanCodeForDisplay(suggestion.code)}
                      </SyntaxHighlighter>
                      
                      {/* Code actions bar */}
                      <div style={{
                        display: 'flex',
                        justifyContent: 'space-between',
                        alignItems: 'center',
                        padding: '8px 12px',
                        backgroundColor: '#252526',
                        borderTop: '1px solid #333'
                      }}>
                        <span style={{
                          color: '#aaa',
                          fontSize: '12px',
                          display: 'flex',
                          alignItems: 'center',
                          gap: '6px'
                        }}>
                          <span style={{ color: '#D5451B', fontWeight: 'bold' }}>B</span>
                          Code suggestion
                        </span>
                        <button 
                          onClick={() => copyToClipboard(cleanCodeForDisplay(suggestion.code), idx)}
                          style={{
                            background: 'none',
                            border: '1px solid #555',
                            borderRadius: '4px',
                            padding: '4px 8px',
                            color: '#eee',
                            fontSize: '12px',
                            cursor: 'pointer'
                          }}
                        >
                          {copiedCode === idx ? 'Copied!' : 'Copy'}
                        </button>
                      </div>
                    </div>
                    
                    {/* Action button */}
                    <button 
                      onClick={() => onApplyFix(suggestion)}
                      style={{
                        backgroundColor: '#D5451B',
                        color: 'white',
                        border: 'none',
                        borderRadius: '4px',
                        padding: '8px 12px',
                        fontSize: '13px',
                        fontWeight: '500',
                        cursor: 'pointer',
                        width: '100%'
                      }}
                    >
                      Apply Fix
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>
        ))}
        
        {/* Loading indicator */}
        {isLoading && (
          <div style={{
            alignSelf: 'flex-start',
            padding: '12px 16px',
            borderRadius: '12px',
            backgroundColor: 'white',
            color: '#6b7280',
            boxShadow: '0 1px 2px rgba(0, 0, 0, 0.05)',
            border: '1px solid #e5e7eb',
            display: 'flex',
            alignItems: 'center',
            gap: '8px'
          }}>
            <div style={{
              width: '8px',
              height: '8px',
              borderRadius: '50%',
              backgroundColor: '#D5451B',
              animation: 'pulse 1.5s infinite ease-in-out'
            }}></div>
            <div style={{
              width: '8px',
              height: '8px',
              borderRadius: '50%',
              backgroundColor: '#FF9B45',
              animation: 'pulse 1.5s infinite ease-in-out',
              animationDelay: '0.2s'
            }}></div>
            <div style={{
              width: '8px',
              height: '8px',
              borderRadius: '50%',
              backgroundColor: '#FFD166',
              animation: 'pulse 1.5s infinite ease-in-out',
              animationDelay: '0.4s'
            }}></div>
            <span>Bit is thinking...</span>
          </div>
        )}
        
        {/* Invisible element to scroll to */}
        <div ref={messagesEndRef} />
      </div>
      
      {/* Input area */}
      <div style={{
        padding: '16px',
        borderTop: '1px solid #e5e7eb',
        backgroundColor: 'white',
        display: 'flex',
        gap: '8px'
      }}>
        <input
          ref={inputRef}
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
          placeholder="Ask Bit about your code..."
          style={{
            flex: '1',
            padding: '10px 14px',
            borderRadius: '6px',
            border: '1px solid #e5e7eb',
            fontSize: '14px',
            outline: 'none'
          }}
        />
        <button
          onClick={handleSendMessage}
          disabled={isLoading || inputValue.trim() === ''}
          style={{
            padding: '10px 16px',
            backgroundColor: '#D5451B',
            color: 'white',
            border: 'none',
            borderRadius: '6px',
            fontSize: '14px',
            fontWeight: '500',
            cursor: isLoading || inputValue.trim() === '' ? 'not-allowed' : 'pointer',
            opacity: isLoading || inputValue.trim() === '' ? '0.7' : '1'
          }}
        >
          Send
        </button>
      </div>
      
      <style>{`
        @keyframes pulse {
          0%, 100% { opacity: 0.5; transform: scale(0.8); }
          50% { opacity: 1; transform: scale(1); }
        }
      `}</style>
    </div>
  );
};

export default BitChatInterface;