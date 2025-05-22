import React, { useState } from 'react';
import { 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  Cell, ReferenceLine, LineChart, Line
} from 'recharts';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';

const EnhancedPredictionDistribution = ({ 
  predictionDistribution, 
  errorAnalysis, 
  modelInfo,
  confidenceAnalysis,
  featureImportance,
  trainingHistory
}) => {
  const [selectedClass, setSelectedClass] = useState(null);
  const [expandedSection, setExpandedSection] = useState(null);
  const [activeChartTab, setActiveChartTab] = useState('distribution');

  // Add proper null checks
  if (!predictionDistribution || !Array.isArray(predictionDistribution) || predictionDistribution.length === 0) {
    return (
      <div className="enhanced-chart-container">
        <h3 className="chart-title">Model Analysis</h3>
        <div className="empty-state">Analysis data unavailable</div>
      </div>
    );
  }

  // Get framework for code examples
  const getFramework = () => {
    return modelInfo && modelInfo.framework ? modelInfo.framework.toLowerCase() : 'sklearn';
  };
  
  const framework = getFramework();
  
  // Handle bar click
  const handleBarClick = (data) => {
    setSelectedClass(selectedClass === data.class_name ? null : data.class_name);
    setExpandedSection(null);
  };

  // Calculate expected distribution (ideally uniform)
  const calculateExpectedDistribution = () => {
    if (!predictionDistribution || predictionDistribution.length === 0) return 0;
    
    const totalPredictions = predictionDistribution.reduce((sum, item) => sum + item.count, 0);
    return totalPredictions / predictionDistribution.length;
  };

  const calculateImbalanceThreshold = () => {
    // If you have many classes, a smaller deviation might be significant
    if (predictionDistribution && predictionDistribution.length > 5) {
        return 0.2; // 20% deviation threshold for many classes
    } else {
        return 0.3; // 30% deviation threshold for few classes
    }
  };

  const expectedCount = calculateExpectedDistribution();
  const imbalanceThreshold = calculateImbalanceThreshold();

  // Get class-specific error info
  const getClassErrors = (className) => {
    if (!errorAnalysis || !errorAnalysis.error_types) return null;
    
    const classId = parseInt(className.replace('Class ', ''));
    
    // Find errors related to this class
    const relevantErrors = errorAnalysis.error_types.filter(error => 
      error.true_class === classId || error.predicted_class === classId
    );
    
    // Add confidence values from confidence analysis if available
    if (confidenceAnalysis && confidenceAnalysis.overconfident_examples) {
      // Find overconfident errors for this class
      const overconfidentExamples = confidenceAnalysis.overconfident_examples.indices
        .filter(idx => {
          // Check if this index relates to the current class
          const errorSample = errorAnalysis.error_indices.includes(idx);
          return errorSample; // Add more conditions if needed
        });
        
      // Add this info to your error analysis
      if (overconfidentExamples.length > 0) {
        relevantErrors.overconfident = overconfidentExamples;
      }
    }
    
    return relevantErrors;
  };
  
  // Determine if this class has imbalance issues
  const hasImbalanceIssue = (classData) => {
    return Math.abs(classData.count - expectedCount) > (expectedCount * imbalanceThreshold);
  };
  
  // Get code example for fixing class-specific issues
  const getCodeExample = (className, issue) => {
    // Extract class index from the class name
    const classIdx = parseInt(className.replace('Class ', ''));
    
    // Get model-specific parameters
    const numClasses = predictionDistribution ? predictionDistribution.length : 2;
    const batchSize = modelInfo?.batch_size || 32;
    const learningRate = 0.001; // Default value
    
    // Class-specific data
    const classData = predictionDistribution.find(d => d.class_name === className);
    const totalSamples = predictionDistribution.reduce((sum, item) => sum + item.count, 0);
    const classPercentage = classData ? (classData.count / totalSamples) * 100 : 0;
    const expectedPercentage = 100 / numClasses;
    const weightMultiplier = expectedPercentage / classPercentage;
    
    // Round to 2 decimal places for cleaner code
    const roundedWeight = Math.round(weightMultiplier * 100) / 100;
    
    if (issue === 'imbalance') {
      // Generate framework-specific code with actual model parameters
      if (framework === 'pytorch') {
        return `# PyTorch: Fix class imbalance for ${className}
# Current distribution: ${classPercentage.toFixed(1)}% (expected ${expectedPercentage.toFixed(1)}%)

# Method 1: Use weighted loss function
class_weights = torch.ones(${numClasses})
class_weights[${classIdx}] = ${roundedWeight}  # Weight based on class distribution

# Apply to loss function
criterion = nn.CrossEntropyLoss(weight=class_weights)

# Method 2: Use weighted sampling
from torch.utils.data import WeightedRandomSampler

# Calculate weights for all samples
sample_weights = []
for label in y_train:
    if label == ${classIdx}:
        sample_weights.append(${roundedWeight})
    else:
        sample_weights.append(1.0)

# Create sampler
sampler = WeightedRandomSampler(
    sample_weights, 
    len(sample_weights)
)

# Create dataloader with your batch size
train_loader = DataLoader(
    dataset, 
    batch_size=${batchSize}, 
    sampler=sampler
)`;
      } else if (framework === 'tensorflow') {
        return `# TensorFlow: Fix class imbalance for ${className}
# Current distribution: ${classPercentage.toFixed(1)}% (expected ${expectedPercentage.toFixed(1)}%)

# Create class weights dictionary
class_weights = {}
for i in range(${numClasses}):
    if i == ${classIdx}:
        class_weights[i] = ${roundedWeight}  # Adjust weight for ${className}
    else:
        class_weights[i] = 1.0

# Use in model.fit()
model.compile(
    optimizer=tf.keras.optimizers.Adam(${learningRate}),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    X_train, y_train,
    class_weight=class_weights,  # Apply the weights
    batch_size=${batchSize},
    epochs=10,
    validation_split=0.2
)

# Alternative: Use SMOTE oversampling
from imblearn.over_sampling import SMOTE

smote = SMOTE(sampling_strategy={
    ${classIdx}: int(np.sum(y_train != ${classIdx}) / (${numClasses} - 1))  # Balance with other classes
})
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Verify new class distribution
new_dist = np.bincount(y_resampled)
print(f"New distribution: {new_dist}")`;
      } else {
        // Default to sklearn
        return `# Scikit-learn: Fix class imbalance for ${className}
# Current distribution: ${classPercentage.toFixed(1)}% (expected ${expectedPercentage.toFixed(1)}%)

# Method 1: Use class weights
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Calculate class weights
class_weight = 'balanced'  # Automatic option
# Or manual weights:
manual_weights = {
    ${classIdx}: ${roundedWeight}  # Adjust weight for ${className}
}

# Create model with weights
model = RandomForestClassifier(
    n_estimators=100,
    class_weight=manual_weights,  # Use 'balanced' or manual weights
    random_state=42
)

# Method 2: Use SMOTE for targeted oversampling
from imblearn.over_sampling import SMOTE

# Calculate target number for ${className}
target_count = int(np.sum(y_train != ${classIdx}) / (${numClasses} - 1))

# Create sampling strategy
strategy = {${classIdx}: target_count}

# Apply SMOTE
smote = SMOTE(sampling_strategy=strategy, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Train on balanced dataset
model.fit(X_resampled, y_resampled)

# Check new class distribution
print(f"New distribution: {np.bincount(y_resampled)}")`;
      }
    } else if (issue === 'errors') {
      // Get class-specific error data if available
      const errorCount = errorAnalysis?.error_count || 0;
      const classErrors = getClassErrors(className);
      const errorTypes = classErrors ? classErrors.length : 0;
      
      return `# Analyze errors for ${className}
import numpy as np
import matplotlib.pyplot as plt
${framework === 'pytorch' ? 'import torch' : ''}

# Find ${className} error samples
class_idx = ${classIdx}

# Error statistics for this class:
# - Total errors in dataset: ${errorCount}
# - Error types involving this class: ${errorTypes}

# Get error indices
${framework === 'pytorch' ? `
# PyTorch implementation
model.eval()
error_indices = []

with torch.no_grad():
    for batch_idx, (data, target) in enumerate(test_loader):
        output = model(data)
        pred = output.argmax(dim=1)
        
        # Find errors related to class ${classIdx}
        class_errors = ((pred == ${classIdx}) & (target != ${classIdx})) | ((pred != ${classIdx}) & (target == ${classIdx}))
        
        if class_errors.any():
            # Get the indices in the original dataset
            batch_error_indices = torch.where(class_errors)[0].cpu() + batch_idx * test_loader.batch_size
            error_indices.extend(batch_error_indices.tolist())` 
      : 
      `# TensorFlow/sklearn implementation
# Where true class is ${classIdx} but prediction is wrong
true_class_errors = np.where((y_test == ${classIdx}) & (y_pred != ${classIdx}))[0]

# Where predicted class is ${classIdx} but true class is different
pred_class_errors = np.where((y_pred == ${classIdx}) & (y_test != ${classIdx}))[0]

# Combine error indices
error_indices = np.concatenate([true_class_errors, pred_class_errors])`}

# Visualize the errors
if len(error_indices) > 0:
    # Extract error features
    X_errors = X_test[error_indices]
    y_errors = y_test[error_indices]
    y_pred_errors = y_pred[error_indices]
    
    # Analyze feature patterns in errors
    feature_means = X_errors.mean(axis=0)
    feature_stds = X_errors.std(axis=0)
    
    # Get top 5 distinctive features by variance
    feature_importance = feature_stds / (X_test.std(axis=0) + 1e-10)
    top_features = np.argsort(feature_importance)[-5:]
    
    # Plot feature importance for these errors
    plt.figure(figsize=(10, 6))
    plt.bar(top_features, feature_importance[top_features])
    plt.xlabel('Feature Index')
    plt.ylabel('Relative Importance in Errors')
    plt.title(f'Top Features Contributing to {className} Errors')
    plt.tight_layout()
    plt.show()
    
    print(f"Found {len(error_indices)} errors involving {className}")
    print(f"Most common misclassification: {className} → Class {y_pred_errors[0]}")`;
    }
    
    return '';
  };

  // Render the details panel for a selected class
  const renderClassDetails = () => {
    if (!selectedClass) return null;
    
    const classData = predictionDistribution.find(d => d.class_name === selectedClass);
    const totalPredictions = predictionDistribution.reduce((sum, item) => sum + item.count, 0);
    const percentage = ((classData.count / totalPredictions) * 100).toFixed(1);
    const classErrors = getClassErrors(selectedClass);
    const hasImbalance = hasImbalanceIssue(classData);
    
    return (
      <div className="class-details-panel">
        <h4 className="details-title">{selectedClass} Analysis</h4>
        
        <div className="details-stats">
          <div className="stat-item">
            <span className="stat-label">Count</span>
            <span className="stat-value">{classData.count}</span>
          </div>
          <div className="stat-item">
            <span className="stat-label">Percentage</span>
            <span className="stat-value">{percentage}%</span>
          </div>
          <div className="stat-item">
            <span className="stat-label">Expected</span>
            <span className="stat-value">{Math.round(expectedCount)}</span>
          </div>
          <div className="stat-item">
            <span className="stat-label">Deviation</span>
            <span className={`stat-value ${Math.abs(classData.count - expectedCount) > (expectedCount * 0.2) ? 'warning' : ''}`}>
              {((classData.count - expectedCount) / expectedCount * 100).toFixed(1)}%
            </span>
          </div>
        </div>
        
        <div className="details-sections">
          {hasImbalance && (
            <div className="details-section">
              <div 
                className="section-header"
                onClick={() => setExpandedSection(expandedSection === 'imbalance' ? null : 'imbalance')}
              >
                <span className="section-icon warning">⚠️</span>
                <h5 className="section-title">Class Imbalance Detected</h5>
                <span className="section-toggle">{expandedSection === 'imbalance' ? '−' : '+'}</span>
              </div>
              
              {expandedSection === 'imbalance' && (
                <div className="section-content">
                  <p className="section-description">
                    {classData.count > expectedCount 
                      ? `This class is overrepresented (${percentage}% of predictions vs expected ${(expectedCount/totalPredictions*100).toFixed(1)}%).`
                      : `This class is underrepresented (${percentage}% of predictions vs expected ${(expectedCount/totalPredictions*100).toFixed(1)}%).`
                    }
                  </p>
                  
                  <div className="code-section">
                    <h6 className="code-title">Fix Class Imbalance</h6>
                    <SyntaxHighlighter 
                      language="python"
                      style={vscDarkPlus}
                      customStyle={{ borderRadius: '6px' }}
                    >
                      {getCodeExample(selectedClass, 'imbalance')}
                    </SyntaxHighlighter>
                  </div>
                </div>
              )}
            </div>
          )}
          
          {classErrors && classErrors.length > 0 && (
            <div className="details-section">
              <div 
                className="section-header"
                onClick={() => setExpandedSection(expandedSection === 'errors' ? null : 'errors')}
              >
                <span className="section-icon error"></span>
                <h5 className="section-title">Error Analysis</h5>
                <span className="section-toggle">{expandedSection === 'errors' ? '−' : '+'}</span>
              </div>
              
              {expandedSection === 'errors' && (
                <div className="section-content">
                  <div className="error-table">
                    <div className="table-header">
                      <div>Error Type</div>
                      <div>Count</div>
                    </div>
                    {classErrors.map((error, idx) => (
                      <div className="table-row" key={idx}>
                        <div>{`Class ${error.true_class} → ${error.predicted_class}`}</div>
                        <div>{error.count}</div>
                      </div>
                    ))}
                  </div>
                  
                  <div className="code-section">
                    <h6 className="code-title">Analyze Class Errors</h6>
                    <SyntaxHighlighter 
                      language="python"
                      style={vscDarkPlus}
                      customStyle={{ borderRadius: '6px' }}
                    >
                      {getCodeExample(selectedClass, 'errors')}
                    </SyntaxHighlighter>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    );
  };

  // Render chart tabs
  const renderChartTabs = () => {
    const tabs = [
      { id: 'distribution', label: 'Class Distribution' },
      { id: 'training', label: 'Training History' },
      { id: 'learning_rate', label: 'Learning Rate' }
    ];

    return (
      <div style={{ 
        display: 'flex', 
        gap: '8px', 
        marginBottom: '20px', 
        borderBottom: '2px solid #e5e7eb', 
        paddingBottom: '8px' 
      }}>
        {tabs.map(tab => (
          <button
            key={tab.id}
            onClick={() => setActiveChartTab(tab.id)}
            style={{
              padding: '10px 16px',
              border: 'none',
              background: activeChartTab === tab.id ? '#e74c32' : 'none',
              cursor: 'pointer',
              borderRadius: activeChartTab === tab.id ? '8px 8px 0 0' : '8px',
              fontWeight: '500',
              transition: 'all 0.2s',
              color: activeChartTab === tab.id ? 'white' : '#6b7280',
              transform: activeChartTab === tab.id ? 'translateY(-2px)' : 'none'
            }}
            onMouseOver={(e) => {
              if (activeChartTab !== tab.id) {
                e.target.style.backgroundColor = '#f3f4f6';
                e.target.style.color = '#374151';
              }
            }}
            onMouseOut={(e) => {
              if (activeChartTab !== tab.id) {
                e.target.style.backgroundColor = 'transparent';
                e.target.style.color = '#6b7280';
              }
            }}
          >
            {tab.label}
          </button>
        ))}
      </div>
    );
  };

  // Render chart content based on active tab
  const renderChartContent = () => {
    switch (activeChartTab) {
      case 'distribution':
        return (
          <div className="chart-wrapper">
            <ResponsiveContainer width="100%" height={300}>
              <BarChart 
                data={predictionDistribution} 
                margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="class_name" />
                <YAxis />
                <Tooltip 
                  formatter={(value, name, props) => [`${value} samples`, `Count`]}
                  labelFormatter={(label) => `${label}`}
                />
                <ReferenceLine y={expectedCount} stroke="#666" strokeDasharray="3 3" />
                <Bar dataKey="count" onClick={handleBarClick}>
                  {predictionDistribution.map((entry, index) => {
                    const isSelected = entry.class_name === selectedClass;
                    const hasImbalance = Math.abs(entry.count - expectedCount) > (expectedCount * 0.3);
                    
                    let fillColor = '#e74c32';
                    if (isSelected) fillColor = '#e74c32';
                    else if (hasImbalance) fillColor = entry.count > expectedCount ? '#4e42f5' : '#ffba66';
                    
                    return (
                      <Cell 
                        key={`cell-${index}`} 
                        fill={fillColor}
                        stroke={isSelected ? '#333' : 'none'}
                        strokeWidth={isSelected ? 2 : 0}
                        cursor="pointer"
                      />
                    );
                  })}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
            
            <div className="chart-legend">
              <div className="legend-item">
                <span className="legend-color" style={{ backgroundColor: '#e74c32' }}></span>
                <span className="legend-label">Balanced Class</span>
              </div>
              <div className="legend-item">
                <span className="legend-color" style={{ backgroundColor: '#4e42f5' }}></span>
                <span className="legend-label">Overrepresented</span>
              </div>
              <div className="legend-item">
                <span className="legend-color" style={{ backgroundColor: '#ffba66' }}></span>
                <span className="legend-label">Underrepresented</span>
              </div>
              <div className="legend-item">
                <span className="legend-line"></span>
                <span className="legend-label">Expected Distribution</span>
              </div>
            </div>
          </div>
        );

      case 'training':
        return (
          <div className="chart-wrapper">
            {trainingHistory && trainingHistory.length > 0 ? (
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={trainingHistory}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="iteration" />
                  <YAxis yAxisId="left" label={{ value: 'Accuracy', angle: -90, position: 'insideLeft' }} />
                  <YAxis yAxisId="right" orientation="right" label={{ value: 'Loss', angle: 90, position: 'insideRight' }} />
                  <Tooltip formatter={(value, name) => [value.toFixed(4), name]} />
                  <Legend />
                  <Line yAxisId="left" type="monotone" dataKey="accuracy" stroke="#e74c32" name="Accuracy" strokeWidth={2} />
                  <Line yAxisId="right" type="monotone" dataKey="loss" stroke="#7dd3fc" name="Loss" strokeWidth={2} />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <div className="empty-state" style={{ height: '300px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                No training history available
              </div>
            )}
          </div>
        );

      case 'learning_rate':
        return (
          <div className="chart-wrapper">
            {trainingHistory && trainingHistory.length > 0 && trainingHistory[0].learning_rate ? (
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={trainingHistory}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="iteration" />
                  <YAxis />
                  <Tooltip formatter={(value) => [value.toExponential(4), 'Learning Rate']} />
                  <Line type="monotone" dataKey="learning_rate" stroke="#e74c32" name="Learning Rate" strokeWidth={2} />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <div className="empty-state" style={{ height: '300px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                No learning rate data available
              </div>
            )}
          </div>
        );

      default:
        return null;
    }
  };
  
  return (
    <div className="enhanced-chart-container">
      <h3 className="chart-title">Model Analysis Dashboard</h3>
      
      {renderChartTabs()}
      
      {renderChartContent()}
      
      {activeChartTab === 'distribution' && renderClassDetails()}
      
      {activeChartTab === 'distribution' && (
        <div className="chart-tips">
          <div className="tip-icon"></div>
          <div className="tip-text">Click on any bar to see detailed analysis and code samples for that class</div>
        </div>
      )}

      {activeChartTab === 'training' && (
        <div className="chart-tips">
          <div className="tip-icon"></div>
          <div className="tip-text">Track your model's learning progress over training iterations</div>
        </div>
      )}

      {activeChartTab === 'learning_rate' && (
        <div className="chart-tips">
          <div className="tip-icon"></div>
          <div className="tip-text">Monitor how the learning rate changes during training</div>
        </div>
      )}
    </div>
  );
};

export default EnhancedPredictionDistribution;