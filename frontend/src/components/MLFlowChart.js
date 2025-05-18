import React, { useState, useRef, useEffect } from 'react';

const MLFlowChart = () => {
  const [selectedNode, setSelectedNode] = useState(null);
  const [zoom, setZoom] = useState(1);
  const [position, setPosition] = useState({ x: 0, y: 0 });
  const svgRef = useRef(null);
  
  // Node types with better colors and descriptions
  const nodeTypes = {
    data: {
      color: '#5fb3e5',
      gradient: 'url(#dataGradient)',
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M22 12H2M16 6H8M19 18H5"/>
        </svg>
      ),
      description: 'Stages related to data handling, including collection, cleaning, and preparation.'
    },
    model: {
      color: '#5fe5b9',
      gradient: 'url(#modelGradient)',
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/>
        </svg>
      ),
      description: 'Model architecture selection and configuration stages.'
    },
    training: {
      color: '#d8ce4f',
      gradient: 'url(#trainingGradient)',
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"/>
        </svg>
      ),
      description: 'Training process and optimization of model parameters.'
    },
    evaluation: {
      color: '#a7ce4f',
      gradient: 'url(#evaluationGradient)',
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M22 11.08V12a10 10 0 11-5.93-9.14"/>
          <polyline points="22 4 12 14.01 9 11.01"/>
        </svg>
      ),
      description: 'Testing model performance and analyzing results.'
    },
    deployment: {
      color: '#ce4fd8',
      gradient: 'url(#deploymentGradient)',
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M12 2L2 7l10 5 10-5-10-5z"/>
          <path d="M2 17l10 5 10-5"/>
          <path d="M2 12l10 5 10-5"/>
        </svg>
      ),
      description: 'Taking the model to production environments.'
    },
    feedback: {
      color: '#ce754f',
      gradient: 'url(#feedbackGradient)',
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M21 11.5a8.38 8.38 0 01-.9 3.8 8.5 8.5 0 01-7.6 4.7 8.38 8.38 0 01-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 01-.9-3.8 8.5 8.5 0 014.7-7.6 8.38 8.38 0 013.8-.9h.5a8.48 8.48 0 018 8v.5z"/>
        </svg>
      ),
      description: 'Collecting and incorporating user feedback to improve the model.'
    }
  };

  // Improved pipeline data with better positioning and meaningful connections
  const flowchartData = {
    nodes: [
      { 
        id: 'data-collection', 
        type: 'data', 
        label: 'Data Collection',
        x: 150, 
        y: 100,
        description: 'Gather data from various sources including databases, APIs, files, or sensors.'
      },
      { 
        id: 'data-cleaning', 
        type: 'data', 
        label: 'Data Cleaning',
        x: 150, 
        y: 200,
        description: 'Handle missing values, remove duplicates, and correct inconsistencies.'
      },
      { 
        id: 'feature-engineering', 
        type: 'data', 
        label: 'Feature Engineering',
        x: 150, 
        y: 300,
        description: 'Create new features, transform existing ones, and select relevant attributes.'
      },
      { 
        id: 'data-splitting', 
        type: 'data', 
        label: 'Train/Test Split',
        x: 150, 
        y: 400,
        description: 'Divide dataset into training, validation, and test sets.'
      },
      { 
        id: 'model-selection', 
        type: 'model', 
        label: 'Model Selection',
        x: 350, 
        y: 250,
        description: 'Choose appropriate algorithm type (regression, classification, clustering, etc.).'
      },
      { 
        id: 'hyperparameter-tuning', 
        type: 'model', 
        label: 'Hyperparameter Tuning',
        x: 350, 
        y: 350, 
        description: 'Optimize model configuration to improve performance.'
      },
      { 
        id: 'model-training', 
        type: 'training', 
        label: 'Model Training',
        x: 550, 
        y: 300,
        description: 'Train model on data, optimize parameters through iterations.'
      },
      { 
        id: 'model-validation', 
        type: 'evaluation', 
        label: 'Model Validation',
        x: 750, 
        y: 200,
        description: 'Assess model performance on validation set to prevent overfitting.'
      },
      { 
        id: 'model-evaluation', 
        type: 'evaluation', 
        label: 'Model Evaluation',
        x: 750, 
        y: 300,
        description: 'Measure final performance on test set using metrics like accuracy, F1, etc.'
      },
      { 
        id: 'model-deployment', 
        type: 'deployment', 
        label: 'Model Deployment',
        x: 750, 
        y: 400,
        description: 'Integrate model into production systems with appropriate APIs.'
      },
      {
        id: 'model-monitoring', 
        type: 'feedback', 
        label: 'Model Monitoring',
        x: 550, 
        y: 500,
        description: 'Track model performance in production and detect drift or issues.'
      },
      { 
        id: 'feedback-collection', 
        type: 'feedback', 
        label: 'Feedback Collection',
        x: 350, 
        y: 500,
        description: 'Gather user feedback and production data to improve future versions.'
      }
    ],
    
    connections: [
      { 
        from: 'data-collection', 
        to: 'data-cleaning',
        description: 'Raw data flows to cleaning process'
      },
      { 
        from: 'data-cleaning', 
        to: 'feature-engineering',
        description: 'Clean data enables effective feature engineering'
      },
      { 
        from: 'feature-engineering', 
        to: 'data-splitting',
        description: 'Prepared features are split for model training'
      },
      { 
        from: 'data-splitting', 
        to: 'model-training',
        description: 'Training data is fed to the model'
      },
      { 
        from: 'model-selection', 
        to: 'hyperparameter-tuning',
        description: 'Selected model architecture undergoes tuning'
      },
      { 
        from: 'model-selection', 
        to: 'model-training',
        description: 'Model architecture is used in training'
      },
      { 
        from: 'hyperparameter-tuning', 
        to: 'model-training',
        description: 'Optimized parameters improve training'
      },
      { 
        from: 'model-training', 
        to: 'model-validation',
        description: 'Trained model is validated to prevent overfitting'
      },
      { 
        from: 'model-validation', 
        to: 'model-training',
        label: 'Iterate',
        description: 'Validation results guide further training refinements'
      },
      { 
        from: 'model-validation', 
        to: 'model-evaluation',
        description: 'Validated model undergoes final evaluation'
      },
      { 
        from: 'model-evaluation', 
        to: 'model-deployment',
        description: 'Evaluated model moves to production'
      },
      { 
        from: 'model-deployment', 
        to: 'model-monitoring',
        description: 'Deployed model is continuously monitored'
      },
      { 
        from: 'model-monitoring', 
        to: 'feedback-collection',
        description: 'Monitoring insights inform feedback collection'
      },
      { 
        from: 'feedback-collection', 
        to: 'data-collection',
        label: 'Cycle Repeats',
        description: 'Feedback guides new data collection for model improvement'
      }
    ]
  };
  
  // Handle mouse wheel for zooming
  const handleWheel = (e) => {
    e.preventDefault();
    const delta = e.deltaY < 0 ? 0.1 : -0.1;
    setZoom(Math.max(0.5, Math.min(2, zoom + delta)));
  };
  
  // Dragging functionality
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  
  const handleMouseDown = (e) => {
    if (e.target.closest('.node') === null) {
      setIsDragging(true);
      setDragStart({ x: e.clientX, y: e.clientY });
    }
  };
  
  const handleMouseMove = (e) => {
    if (isDragging) {
      const dx = e.clientX - dragStart.x;
      const dy = e.clientY - dragStart.y;
      setPosition({
        x: position.x + dx,
        y: position.y + dy
      });
      setDragStart({ x: e.clientX, y: e.clientY });
    }
  };
  
  const handleMouseUp = () => {
    setIsDragging(false);
  };
  
  // Event listeners for dragging
  useEffect(() => {
    const svg = svgRef.current;
    
    if (svg) {
      svg.addEventListener('wheel', handleWheel);
      window.addEventListener('mouseup', handleMouseUp);
      window.addEventListener('mousemove', handleMouseMove);
      
      return () => {
        svg.removeEventListener('wheel', handleWheel);
        window.removeEventListener('mouseup', handleMouseUp);
        window.removeEventListener('mousemove', handleMouseMove);
      };
    }
  }, [isDragging, dragStart, position]);
  
  // Generate path between two nodes with a quadratic curve
  const generatePath = (fromNode, toNode, label) => {
    const fromX = fromNode.x * zoom + position.x;
    const fromY = fromNode.y * zoom + position.y;
    const toX = toNode.x * zoom + position.x;
    const toY = toNode.y * zoom + position.y;
    
    // Calculate control point for the curve
    const dx = toX - fromX;
    const dy = toY - fromY;
    const distance = Math.sqrt(dx * dx + dy * dy);
    
    // More curve for longer distances or feedback loops
    const isFeedback = label === "Cycle Repeats";
    const curveIntensity = isFeedback ? 0.7 : 0.2;
    
    // For feedback loops going up, create a more pronounced curve
    let controlPointX, controlPointY;
    
    if (isFeedback) {
      // For feedback loops, create a larger curve to the right
      controlPointX = (fromX + toX) / 2 - distance * 0.6;
      controlPointY = Math.min(fromY, toY) - distance * 0.4;
    } else {
      // For normal connections
      controlPointX = (fromX + toX) / 2;
      controlPointY = (fromY + toY) / 2 - distance * curveIntensity;
    }
    
    return {
      path: `M ${fromX} ${fromY} Q ${controlPointX} ${controlPointY} ${toX} ${toY}`,
      labelPos: {
        x: controlPointX,
        y: controlPointY - 10
      }
    };
  };
  
  // Center the flowchart in the viewport initially
  useEffect(() => {
    if (svgRef.current) {
      const bbox = svgRef.current.getBBox();
      const svgWidth = svgRef.current.clientWidth;
      const svgHeight = svgRef.current.clientHeight;
      
      setPosition({
        x: (svgWidth - bbox.width) / 2 - bbox.x,
        y: (svgHeight - bbox.height) / 2 - bbox.y
      });
    }
  }, []);
  
  return (
    <div className="relative w-full h-full overflow-hidden bg-claude-bg rounded-lg">
      {/* Controls */}
      <div className="absolute top-4 right-4 z-10 flex bg-claude-card rounded-lg shadow-lg overflow-hidden">
        <button 
          className="p-2 text-claude-text-secondary hover:bg-claude-hover"
          onClick={() => setZoom(Math.min(2, zoom + 0.1))}
        >
          <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
            <path fillRule="evenodd" d="M10 5a1 1 0 011 1v3h3a1 1 0 110 2h-3v3a1 1 0 11-2 0v-3H6a1 1 0 110-2h3V6a1 1 0 011-1z" clipRule="evenodd" />
          </svg>
        </button>
        <button 
          className="p-2 text-claude-text-secondary hover:bg-claude-hover"
          onClick={() => setZoom(Math.max(0.5, zoom - 0.1))}
        >
          <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
            <path fillRule="evenodd" d="M5 10a1 1 0 011-1h8a1 1 0 110 2H6a1 1 0 01-1-1z" clipRule="evenodd" />
          </svg>
        </button>
        <button 
          className="p-2 text-claude-text-secondary hover:bg-claude-hover"
          onClick={() => {
            setPosition({ x: 0, y: 0 });
            setZoom(1);
          }}
        >
          <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
            <path fillRule="evenodd" d="M4 2a2 2 0 00-2 2v11a3 3 0 106 0V4a2 2 0 00-2-2H4zm1 14a1 1 0 100-2 1 1 0 000 2zm5-1.757l4.9-4.9a2 2 0 000-2.828L13.485 5.1a2 2 0 00-2.828 0L10 5.757v8.486zM16 18H9.071l6-6H16a2 2 0 012 2v2a2 2 0 01-2 2z" clipRule="evenodd" />
          </svg>
        </button>
      </div>
      
      {/* SVG Canvas */}
      <svg 
        ref={svgRef}
        className="w-full h-full cursor-grab"
        onMouseDown={handleMouseDown}
        style={{ cursor: isDragging ? 'grabbing' : 'grab' }}
      >
        {/* Gradient definitions */}
        <defs>
          {/* Flow path markers */}
          <marker 
            id="arrowhead" 
            markerWidth="10" 
            markerHeight="7" 
            refX="9" 
            refY="3.5" 
            orient="auto"
          >
            <polygon points="0 0, 10 3.5, 0 7" fill="#cccccc" />
          </marker>
          
          {/* Node gradients */}
          <linearGradient id="dataGradient" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stopColor="#7ac3ff" />
            <stop offset="100%" stopColor="#4d8fc7" />
          </linearGradient>
          
          <linearGradient id="modelGradient" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stopColor="#6ff2d0" />
            <stop offset="100%" stopColor="#3bc4a3" />
          </linearGradient>
          
          <linearGradient id="trainingGradient" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stopColor="#f4df5b" />
            <stop offset="100%" stopColor="#c9b641" />
          </linearGradient>
          
          <linearGradient id="evaluationGradient" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stopColor="#bdee55" />
            <stop offset="100%" stopColor="#a7ce4f" />
          </linearGradient>
          
          <linearGradient id="deploymentGradient" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stopColor="#e27bff" />
            <stop offset="100%" stopColor="#ce4fd8" />
          </linearGradient>
          
          <linearGradient id="feedbackGradient" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stopColor="#ff9e6c" />
            <stop offset="100%" stopColor="#ce754f" />
          </linearGradient>
          
          {/* Shadow filter */}
          <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">
            <feDropShadow dx="2" dy="2" stdDeviation="3" floodOpacity="0.3" />
          </filter>
          
          {/* Bevel filter for 3D effect */}
          <filter id="bevel" x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur in="SourceAlpha" stdDeviation="2" result="blur" />
            <feOffset in="blur" dx="4" dy="4" result="offsetBlur" />
            <feSpecularLighting in="blur" surfaceScale="5" specularConstant="1" specularExponent="20" lightingColor="#ffffff" result="specOut">
              <fePointLight x="-5000" y="-10000" z="20000" />
            </feSpecularLighting>
            <feComposite in="specOut" in2="SourceAlpha" operator="in" result="specOut" />
            <feComposite in="SourceGraphic" in2="specOut" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="litPaint" />
            <feMerge>
              <feMergeNode in="offsetBlur" />
              <feMergeNode in="litPaint" />
            </feMerge>
          </filter>
        </defs>
        
        {/* Connections */}
        <g className="connections">
          {flowchartData.connections.map((conn, index) => {
            const fromNode = flowchartData.nodes.find(n => n.id === conn.from);
            const toNode = flowchartData.nodes.find(n => n.id === conn.to);
            
            if (!fromNode || !toNode) return null;
            
            const { path, labelPos } = generatePath(fromNode, toNode, conn.label);
            
            return (
              <g key={`conn-${index}`} className="connection">
                <path 
                  d={path} 
                  stroke="#9a9a9a" 
                  strokeWidth="2"
                  fill="none"
                  markerEnd="url(#arrowhead)"
                  className="transition-all duration-300"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeDasharray={conn.label === "Cycle Repeats" ? "5,5" : "none"}
                />
                
                {conn.label && (
                  <g transform={`translate(${labelPos.x}, ${labelPos.y})`}>
                    <rect
                      x="-30"
                      y="-10"
                      width="60"
                      height="20"
                      rx="5"
                      ry="5"
                      fill="rgba(50, 50, 50, 0.8)"
                      filter="url(#shadow)"
                    />
                    <text
                      textAnchor="middle"
                      dominantBaseline="middle"
                      fill="#ffffff"
                      fontSize="11"
                      fontWeight="bold"
                    >
                      {conn.label}
                    </text>
                  </g>
                )}
              </g>
            );
          })}
        </g>
        
        {/* Nodes */}
        <g className="nodes">
          {flowchartData.nodes.map((node) => {
            const nodeType = nodeTypes[node.type];
            const isSelected = selectedNode === node.id;
            
            return (
              <g
                key={node.id}
                className={`node transition-transform duration-300 ${isSelected ? 'scale-110' : ''}`}
                transform={`translate(${node.x * zoom + position.x}, ${node.y * zoom + position.y})`}
                onClick={(e) => {
                  e.stopPropagation();
                  setSelectedNode(isSelected ? null : node.id);
                }}
              >
                {/* Node background with bevel effect */}
                <rect
                  x="-75"
                  y="-30"
                  width="150"
                  height="60"
                  rx="10"
                  ry="10"
                  fill={nodeType.gradient}
                  filter="url(#bevel)"
                  className={`transition-all duration-300 ${isSelected ? 'stroke-[3px] stroke-white' : 'stroke-[1px] stroke-gray-700'}`}
                />
                
                {/* Node icon */}
                <g transform="translate(-55, 0)" className="text-white">
                  <circle
                    r="15"
                    fill="rgba(255, 255, 255, 0.2)"
                    stroke="rgba(255, 255, 255, 0.5)"
                    strokeWidth="1"
                  />
                  <g transform="scale(0.04) translate(-150, -150)" fill="white" stroke="white">
                    {nodeType.icon}
                  </g>
                </g>
                
                {/* Node label */}
                <text
                  dominantBaseline="middle"
                  textAnchor="middle"
                  fill="white"
                  fontSize="13"
                  fontWeight="bold"
                  filter="drop-shadow(1px 1px 1px rgba(0,0,0,0.5))"
                >
                  {node.label}
                </text>
              </g>
            );
          })}
        </g>
      </svg>
      
      {/* Details panel when node is selected */}
      {selectedNode && (
        <div className="absolute bottom-4 left-4 bg-claude-card border border-claude-border rounded-lg shadow-lg p-4 max-w-md">
          {(() => {
            const node = flowchartData.nodes.find(n => n.id === selectedNode);
            const nodeType = nodeTypes[node.type];
            
            const incomingConnections = flowchartData.connections.filter(conn => conn.to === selectedNode);
            const outgoingConnections = flowchartData.connections.filter(conn => conn.from === selectedNode);
            
            return (
              <>
                <div className="flex items-center mb-3">
                  <div className="mr-3 p-2 rounded-full" style={{ backgroundColor: nodeType.color }}>
                    <div className="w-6 h-6 text-white">
                      {nodeType.icon}
                    </div>
                  </div>
                  <div>
                    <h3 className="font-bold text-claude-text-primary">{node.label}</h3>
                    <p className="text-xs text-claude-text-secondary capitalize">{node.type} Stage</p>
                  </div>
                </div>
                
                <p className="text-sm text-claude-text-primary mb-4">{node.description}</p>
                
                {/* Input and Output Connections */}
                <div className="flex flex-col gap-3">
                  {incomingConnections.length > 0 && (
                    <div>
                      <h4 className="text-xs font-bold text-claude-text-secondary mb-1">Input From:</h4>
                      <div className="flex flex-wrap gap-1">
                        {incomingConnections.map((conn, i) => {
                          const fromNode = flowchartData.nodes.find(n => n.id === conn.from);
                          const fromNodeType = nodeTypes[fromNode.type];
                          
                          return (
                            <div key={i} className="flex flex-col">
                              <span 
                                className="px-2 py-1 rounded text-xs text-white"
                                style={{ backgroundColor: fromNodeType.color }}
                              >
                                {fromNode.label}
                              </span>
                              <span className="text-xs text-claude-text-secondary mt-1">
                                {conn.description}
                              </span>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  )}
                  
                  {outgoingConnections.length > 0 && (
                    <div>
                      <h4 className="text-xs font-bold text-claude-text-secondary mb-1">Output To:</h4>
                      <div className="flex flex-wrap gap-1">
                        {outgoingConnections.map((conn, i) => {
                          const toNode = flowchartData.nodes.find(n => n.id === conn.to);
                          const toNodeType = nodeTypes[toNode.type];
                          
                          return (
                            <div key={i} className="flex flex-col">
                              <span 
                                className="px-2 py-1 rounded text-xs text-white"
                                style={{ backgroundColor: toNodeType.color }}
                              >
                                {toNode.label} {conn.label ? `(${conn.label})` : ''}
                              </span>
                              <span className="text-xs text-claude-text-secondary mt-1">
                                {conn.description}
                              </span>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  )}
                </div>
              </>
            );
          })()}
        </div>
      )}
      
      {/* Legend */}
      <div className="absolute bottom-4 right-4 bg-claude-card border border-claude-border rounded-lg shadow-lg p-3">
        <h4 className="text-xs font-bold text-claude-text-secondary mb-2">Node Types</h4>
        <div className="flex flex-col gap-2">
          {Object.entries(nodeTypes).map(([key, type]) => (
            <div key={key} className="flex items-center text-xs">
              <div 
                className="w-3 h-3 mr-2 rounded-sm" 
                style={{ backgroundImage: type.gradient }}
              ></div>
              <span className="text-claude-text-primary capitalize">{key}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default MLFlowChart;