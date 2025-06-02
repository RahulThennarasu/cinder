import React, { useState, useEffect, useRef } from 'react';
import { 
  Send,
  Sparkles,
  Code,
  Zap,
  CheckCircle,
  ArrowUp,
  FileText,
  Brain
} from 'lucide-react';

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
        const welcomeMessage = `👋 Hi! I'm Bit, your ML optimization assistant. I've analyzed your ${data.framework} model and found several areas for improvement. Let me help you optimize your code!`;
        
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
          const suggestionMessage = `I found ${suggestions.length} optimization opportunities:\n\n${suggestions.slice(0, 3).map((s, i) => `${i + 1}. **${s.title}**: ${s.issue}`).join('\n')}`;
          
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
    }
  };

  // Handle sending messages to Gemini
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
          modelInfo: modelInfo,
          framework: modelInfo?.framework || 'pytorch'
        }),
      });

      if (!response.ok) throw new Error(`API error: ${response.status}`);

      const data = await response.json();
      
      // Type the response with animation
      typeMessage(data.message, () => {
        setMessages(prev => [...prev, {
          role: "assistant",
          content: data.message,
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

  const CodeSuggestionCard = ({ suggestion, index }) => (
    <div className="mt-4 p-4 bg-gray-50 rounded-2xl border border-gray-200">
      <div className="flex items-start justify-between mb-3">
        <div className="flex items-center space-x-2">
          <div className="w-6 h-6 bg-orange-500 rounded-full flex items-center justify-center text-white text-xs font-semibold">
            {index + 1}
          </div>
          <h4 className="font-medium text-gray-900">{suggestion.title}</h4>
        </div>
        <span className="text-xs text-gray-500 bg-gray-200 px-2 py-1 rounded-full">
          {suggestion.category}
        </span>
      </div>
      
      <p className="text-sm text-gray-600 mb-3 leading-relaxed">
        {suggestion.suggestion}
      </p>
      
      {suggestion.code && (
        <div className="bg-gray-900 rounded-xl p-4 mb-3">
          <code className="text-sm text-green-400 font-mono block whitespace-pre-wrap">
            {suggestion.code}
          </code>
        </div>
      )}
      
      <div className="flex space-x-2">
        <button className="flex-1 bg-orange-500 hover:bg-orange-600 text-white text-sm font-medium py-2 px-4 rounded-xl transition-colors flex items-center justify-center space-x-2">
          <CheckCircle className="w-4 h-4" />
          <span>Apply Fix</span>
        </button>
        <button className="px-4 py-2 text-sm font-medium text-gray-600 hover:text-gray-800 transition-colors">
          Explain More
        </button>
      </div>
    </div>
  );

  const QuickActions = () => (
    <div className="flex flex-wrap gap-2 mb-6">
      {[
        "Optimize my model performance",
        "Fix overfitting issues", 
        "Improve training speed",
        "Add regularization"
      ].map((action, index) => (
        <button
          key={index}
          onClick={() => setInputValue(action)}
          className="px-4 py-2 bg-gray-100 hover:bg-gray-200 text-gray-700 text-sm rounded-full transition-colors"
        >
          {action}
        </button>
      ))}
    </div>
  );

  return (
    <div className="h-full bg-white flex flex-col">
      {/* Header */}
      <div className="p-6 border-b border-gray-100">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-semibold text-gray-900 mb-1">
              How can Bit help optimize your model?
            </h1>
            <p className="text-gray-600">
              AI-powered code analysis and continuous improvement suggestions
            </p>
          </div>
          
          {modelInfo && (
            <div className="flex items-center space-x-4 text-sm">
              <div className="flex items-center space-x-2 px-3 py-1 bg-green-50 rounded-full">
                <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                <span className="text-green-700">Model Loaded</span>
              </div>
              <div className="flex items-center space-x-2 px-3 py-1 bg-blue-50 rounded-full">
                <Code className="w-3 h-3 text-blue-600" />
                <span className="text-blue-700">{modelInfo.framework}</span>
              </div>
              <div className="flex items-center space-x-2 px-3 py-1 bg-purple-50 rounded-full">
                <Zap className="w-3 h-3 text-purple-600" />
                <span className="text-purple-700">{(modelInfo.accuracy * 100).toFixed(1)}% accuracy</span>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Chat Messages */}
      <div className="flex-1 overflow-y-auto">
        <div className="max-w-4xl mx-auto p-6">
          {/* Quick Actions */}
          {messages.length === 0 && !isTyping && (
            <QuickActions />
          )}

          {/* Messages */}
          <div className="space-y-6">
            {messages.map((message, index) => (
              <div key={index} className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                <div className={`max-w-3xl ${message.role === 'user' ? 'order-2' : 'order-1'}`}>
                  {message.role === 'assistant' && (
                    <div className="flex items-center space-x-2 mb-2">
                      <div className="w-8 h-8 bg-gradient-to-r from-orange-500 to-red-500 rounded-full flex items-center justify-center">
                        <Brain className="w-4 h-4 text-white" />
                      </div>
                      <span className="text-sm font-medium text-gray-700">Bit</span>
                      <span className="text-xs text-gray-500">{formatTime(message.timestamp)}</span>
                    </div>
                  )}
                  
                  <div className={`p-4 rounded-2xl ${
                    message.role === 'user' 
                      ? 'bg-orange-500 text-white ml-12' 
                      : 'bg-gray-50 border border-gray-200'
                  }`}>
                    <div className={`text-sm leading-relaxed whitespace-pre-wrap ${
                      message.role === 'user' ? 'text-white' : 'text-gray-800'
                    }`}>
                      {message.content}
                    </div>
                  </div>

                  {message.role === 'user' && (
                    <div className="flex items-center justify-end space-x-2 mt-2">
                      <span className="text-xs text-gray-500">{formatTime(message.timestamp)}</span>
                    </div>
                  )}

                  {/* Suggestions */}
                  {message.suggestions && message.suggestions.length > 0 && (
                    <div className="space-y-3">
                      {message.suggestions.map((suggestion, idx) => (
                        <CodeSuggestionCard key={idx} suggestion={suggestion} index={idx} />
                      ))}
                    </div>
                  )}
                </div>
              </div>
            ))}
            
            {/* Typing Animation */}
            {isTyping && (
              <div className="flex justify-start">
                <div className="max-w-3xl">
                  <div className="flex items-center space-x-2 mb-2">
                    <div className="w-8 h-8 bg-gradient-to-r from-orange-500 to-red-500 rounded-full flex items-center justify-center">
                      <Brain className="w-4 h-4 text-white" />
                    </div>
                    <span className="text-sm font-medium text-gray-700">Bit</span>
                    <div className="flex space-x-1">
                      <div className="w-1 h-1 bg-orange-500 rounded-full animate-pulse"></div>
                      <div className="w-1 h-1 bg-orange-500 rounded-full animate-pulse" style={{animationDelay: '0.2s'}}></div>
                      <div className="w-1 h-1 bg-orange-500 rounded-full animate-pulse" style={{animationDelay: '0.4s'}}></div>
                    </div>
                  </div>
                  <div className="bg-gray-50 border border-gray-200 p-4 rounded-2xl">
                    <div className="text-sm leading-relaxed text-gray-800 whitespace-pre-wrap">
                      {typingText}
                      <span className="animate-pulse">|</span>
                    </div>
                  </div>
                </div>
              </div>
            )}
            
            {/* Loading indicator when waiting for API */}
            {isLoading && !isTyping && (
              <div className="flex justify-start">
                <div className="max-w-3xl">
                  <div className="flex items-center space-x-2 mb-2">
                    <div className="w-8 h-8 bg-gradient-to-r from-orange-500 to-red-500 rounded-full flex items-center justify-center">
                      <Brain className="w-4 h-4 text-white" />
                    </div>
                    <span className="text-sm font-medium text-gray-700">Bit</span>
                  </div>
                  <div className="bg-gray-50 border border-gray-200 p-4 rounded-2xl">
                    <div className="flex items-center space-x-2">
                      <div className="flex space-x-1">
                        <div className="w-2 h-2 bg-orange-500 rounded-full animate-bounce"></div>
                        <div className="w-2 h-2 bg-orange-500 rounded-full animate-bounce" style={{animationDelay: '0.1s'}}></div>
                        <div className="w-2 h-2 bg-orange-500 rounded-full animate-bounce" style={{animationDelay: '0.2s'}}></div>
                      </div>
                      <span className="text-sm text-gray-500">Analyzing your code...</span>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
          
          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Input Area */}
      <div className="border-t border-gray-100 p-6">
        <div className="max-w-4xl mx-auto">
          <div className="relative">
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
              className="w-full p-4 pr-16 border border-gray-300 rounded-2xl resize-none focus:ring-2 focus:ring-orange-500 focus:border-transparent text-gray-900 placeholder-gray-500"
              rows="1"
              style={{ minHeight: '56px', maxHeight: '120px' }}
            />
            <button
              onClick={handleSendMessage}
              disabled={isLoading || isTyping || inputValue.trim() === ''}
              className="absolute right-3 bottom-3 p-2 bg-orange-500 hover:bg-orange-600 disabled:opacity-50 disabled:cursor-not-allowed text-white rounded-xl transition-colors"
            >
              <ArrowUp className="w-5 h-5" />
            </button>
          </div>
          
          <div className="mt-3 flex items-center justify-center">
            <div className="flex items-center space-x-1 text-xs text-gray-500">
              <Sparkles className="w-3 h-3" />
              <span>Bit can make mistakes. Always verify important code changes.</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default CladoStyleAIAgent;