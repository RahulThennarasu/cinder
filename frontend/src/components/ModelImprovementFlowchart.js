import React, { useState } from 'react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';

const ModelImprovementFlowchart = ({ modelInfo, errorAnalysis, confidenceAnalysis }) => {
  const [expandedNodes, setExpandedNodes] = useState({});

  // Add this framework detection function
  const getFramework = () => {
    return modelInfo && modelInfo.framework 
      ? modelInfo.framework.toLowerCase() 
      : 'sklearn';
  };

  // Store the framework for use in code generation
  const framework = getFramework();
  // Toggle node expansion
  const toggleNode = (nodeId) => {
    setExpandedNodes(prev => ({
      ...prev,
      [nodeId]: !prev[nodeId]
    }));
  };

  // Helper to determine if we have high bias (underfitting)
  const hasHighBias = () => {
    return modelInfo && modelInfo.accuracy < 0.8;
  };

  // Helper to determine if we have high variance (overfitting)
  const hasHighVariance = () => {
    // If we have training history, we could compare training vs validation accuracy
    // For now, we'll use a heuristic based on confidence analysis
    return confidenceAnalysis && 
           confidenceAnalysis.avg_correct_confidence > 0.9 && 
           confidenceAnalysis.avg_incorrect_confidence > 0.7;
  };

  // Helper to check for class imbalance
  const hasClassImbalance = () => {
    return errorAnalysis && 
           errorAnalysis.error_types && 
           errorAnalysis.error_types.some(type => 
             type.count > (errorAnalysis.error_count * 0.7));
  };

  // Determine what flowchart branch to highlight based on model issues
  const determineModelIssue = () => {
    if (hasHighBias()) return 'high_bias';
    if (hasHighVariance()) return 'high_variance';
    if (hasClassImbalance()) return 'class_imbalance';
    return 'optimize';
  };

  const activeIssue = determineModelIssue();
  // Code generators for different nodes
  const getModelEvaluationCode = () => {
    if (framework === 'pytorch') {
      return `# PyTorch Model Evaluation
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    print(f'Test Accuracy: {accuracy:.4f}')
    
    # For detailed metrics
    all_preds = []
    all_labels = []
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    from sklearn.metrics import classification_report, confusion_matrix
    print("Classification Report:")
    print(classification_report(all_labels, all_preds))
    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))`;
    } else if (framework === 'tensorflow') {
      return `# TensorFlow Model Evaluation
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=2)
print(f"Test accuracy: {test_accuracy:.4f}")

# For more detailed metrics
y_pred = np.argmax(model.predict(X_test), axis=1)
from sklearn.metrics import classification_report, confusion_matrix
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))`;
    } else {
      // Default sklearn
      return `# Scikit-learn Model Evaluation
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Evaluate model on test data
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(report)
print("Confusion Matrix:")
print(conf_matrix)`;
    }
  };

  const getIncreaseComplexityCode = () => {
    if (framework === 'pytorch') {
      return `# Increase model complexity - PyTorch
import torch
import torch.nn as nn

class ImprovedModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_classes=10):
        super(ImprovedModel, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn2 = nn.BatchNorm1d(hidden_size // 2)
        self.layer3 = nn.Linear(hidden_size // 2, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer3(x)
        return x
        
# For CNN models, add more convolutional layers
class ImprovedCNN(nn.Module):
    def __init__(self):
        super(ImprovedCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 10)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(-1, 128 * 4 * 4)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x`;
    } else if (framework === 'tensorflow') {
      return `# Increase model complexity - TensorFlow
import tensorflow as tf

# For standard neural networks
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(input_shape,)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# For CNN models
cnn_model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# Compile model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)`;
    } else {
      return `# Increase model complexity - Scikit-learn
# For tree-based models
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(
    n_estimators=200,  # Increase from default 100
    max_depth=15,      # Allow deeper trees
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42
)

# For gradient boosting
from sklearn.ensemble import GradientBoostingClassifier
gb_model = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)

# For neural networks
from sklearn.neural_network import MLPClassifier
mlp_model = MLPClassifier(
    hidden_layer_sizes=(128, 64, 32),  # Add more layers and neurons
    activation='relu',
    solver='adam',
    max_iter=1000,
    random_state=42
)

# Train the model
model.fit(X_train, y_train)`;
    }
  };

  const getRegularizationCode = () => {
    if (framework === 'pytorch') {
      return `# Add regularization - PyTorch
import torch.nn as nn
import torch.optim as optim

# 1. Add dropout to model
model = nn.Sequential(
    nn.Linear(input_size, hidden_size),
    nn.ReLU(),
    nn.Dropout(0.5),  # Add dropout with 0.5 probability
    nn.Linear(hidden_size, hidden_size // 2),
    nn.ReLU(),
    nn.Dropout(0.3),  # Add dropout with 0.3 probability
    nn.Linear(hidden_size // 2, num_classes)
)

# 2. Add L2 regularization (weight decay)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# 3. Implement early stopping
best_val_loss = float('inf')
patience = 5
patience_counter = 0
best_model_state = None

for epoch in range(epochs):
    # Training
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
    
    # Validation
    val_loss = validate(model, val_loader, criterion)
    
    # Early stopping logic
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        best_model_state = model.state_dict().copy()
    else:
        patience_counter += 1
        
    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch}")
        # Restore best model
        model.load_state_dict(best_model_state)
        break`;
    } else if (framework === 'tensorflow') {
      return `# Add regularization - TensorFlow
import tensorflow as tf

# 1. Add L1 or L2 regularization
model = tf.keras.Sequential([
    tf.keras.layers.Dense(
        hidden_size, 
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(0.001),  # L2 regularization
        input_shape=(input_shape,)
    ),
    tf.keras.layers.Dropout(0.5),  # Add dropout
    tf.keras.layers.Dense(
        hidden_size // 2,
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(0.001)
    ),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# 2. Create callbacks for early stopping and model checkpointing
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    'best_model.h5',
    monitor='val_loss',
    save_best_only=True
)

# 3. Use callbacks in training
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping, model_checkpoint],
    verbose=1
)`;
    } else {
      return `# Add regularization - Scikit-learn
# For linear models
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(
    C=0.1,  # Lower C = stronger regularization (inverse of regularization strength)
    penalty='l2',  # L2 regularization (Ridge)
    solver='liblinear',
    random_state=42
)

# For tree models - limit tree growth
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(
    max_depth=6,            # Limit tree depth
    min_samples_split=5,    # Require more samples to split
    min_samples_leaf=2,     # Require more samples in leaves
    max_features='sqrt',    # Use subset of features
    random_state=42
)

# For neural networks
from sklearn.neural_network import MLPClassifier
mlp_model = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    alpha=0.01,             # L2 penalty (regularization term)
    early_stopping=True,    # Use early stopping
    validation_fraction=0.2,
    n_iter_no_change=10,    # Patience parameter
    random_state=42
)

# Train with regularization
model.fit(X_train, y_train)`;
    }
  };

  const getClassImbalanceCode = () => {
    if (framework === 'pytorch') {
      return `# Handle class imbalance - PyTorch
import torch
import numpy as np
from torch.utils.data import WeightedRandomSampler
from imblearn.over_sampling import SMOTE

# Method 1: Use weighted loss function
# Calculate class weights inversely proportional to frequency
class_counts = [sum(y_train == c) for c in range(num_classes)]
class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
class_weights = class_weights / class_weights.sum() * num_classes

# Use weighted loss
criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

# Method 2: Use weighted sampling
sample_weights = [class_weights[y] for y in y_train]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=32,
    sampler=sampler  # Use the weighted sampler
)

# Method 3: Use SMOTE for oversampling (before creating PyTorch dataset)
# Convert PyTorch dataset to numpy arrays first
X_np = X_train.numpy()  # Assuming X_train is a tensor
y_np = y_train.numpy()  # Assuming y_train is a tensor

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_np, y_np)

# Convert back to tensors
X_resampled_tensor = torch.FloatTensor(X_resampled)
y_resampled_tensor = torch.LongTensor(y_resampled)

# Create new dataset and loader with balanced data
balanced_dataset = torch.utils.data.TensorDataset(X_resampled_tensor, y_resampled_tensor)
balanced_loader = torch.utils.data.DataLoader(balanced_dataset, batch_size=32, shuffle=True)`;
    } else if (framework === 'tensorflow') {
      return `# Handle class imbalance - TensorFlow
import numpy as np
import tensorflow as tf
from imblearn.over_sampling import SMOTE

# Method 1: Class weights in model training
# Calculate class weights
class_counts = np.bincount(y_train)
total_samples = len(y_train)
class_weights = {i: total_samples / (len(class_counts) * count) for i, count in enumerate(class_counts)}

# Use weights in training
model.fit(
    X_train, y_train,
    class_weight=class_weights,
    epochs=10,
    batch_size=32,
    validation_split=0.2
)

# Method 2: SMOTE for oversampling
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Train on balanced dataset
model.fit(
    X_resampled, y_resampled,
    epochs=10,
    batch_size=32,
    validation_split=0.2
)

# Method 3: Use Focal Loss for extreme imbalance
def focal_loss(gamma=2., alpha=4.):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = 1e-9
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * tf.math.pow(1. - y_pred, gamma)
        loss = weight * cross_entropy
        return tf.reduce_mean(tf.reduce_sum(loss, axis=1))
    return focal_loss_fixed

# Use focal loss in model
model.compile(
    optimizer='adam',
    loss=focal_loss(gamma=2, alpha=4),
    metrics=['accuracy']
)`;
    } else {
      return `# Handle class imbalance - Scikit-learn
import numpy as np
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
from sklearn.ensemble import RandomForestClassifier

# Method 1: Use class weights in model
# Calculate class weights
class_weights = 'balanced'  # Automatically adjust weights inversely proportional to frequencies

# Use weights in model
model = RandomForestClassifier(
    n_estimators=100,
    class_weight=class_weights,
    random_state=42
)

# Method 2: Use SMOTE for oversampling
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Check new class distribution
unique, counts = np.unique(y_resampled, return_counts=True)
print(f"Class distribution after SMOTE: {dict(zip(unique, counts))}")

# Train on balanced dataset
model.fit(X_resampled, y_resampled)

# Method 3: SMOTETomek (combination of over and under sampling)
smt = SMOTETomek(random_state=42)
X_smt, y_smt = smt.fit_resample(X_train, y_train)
print(f"Class distribution after SMOTETomek: {dict(zip(*np.unique(y_smt, return_counts=True)))}")`;
    }
  };
  // Render a code box
  const renderCodeBox = (code, language = 'python') => (
    <div className="flowchart-code-box">
      <SyntaxHighlighter 
        language={language}
        style={vscDarkPlus}
        showLineNumbers={true}
      >
        {code}
      </SyntaxHighlighter>
    </div>
  );

  // Render a flowchart node
  const renderNode = (nodeId, title, content, codeExample = null, isActive = false) => {
    const isExpanded = expandedNodes[nodeId];
    
    return (
      <div className={`flowchart-node ${isActive ? 'flowchart-node-active' : ''}`}>
        <div 
          className="flowchart-node-header"
          onClick={() => toggleNode(nodeId)}
        >
          <div className="flowchart-node-title">{title}</div>
          <div className="flowchart-node-toggle">{isExpanded ? '−' : '+'}</div>
        </div>
        
        {isExpanded && (
          <div className="flowchart-node-content">
            <div className="flowchart-node-description">
              {content}
            </div>
            
            {codeExample && renderCodeBox(codeExample)}
          </div>
        )}
      </div>
    );
  };

  // Render the connector lines between nodes
  const renderConnector = (direction = 'vertical') => (
    <div className={`flowchart-connector ${direction}`}>
      <div className="flowchart-connector-line"></div>
      {direction === 'branch' && (
        <>
          <div className="flowchart-connector-line horizontal left"></div>
          <div className="flowchart-connector-line horizontal right"></div>
        </>
      )}
    </div>
  );

  return (
    <div className="flowchart-container">
      <h3 className="card-title">Model Improvement Flowchart</h3>
      
      {/* Root node */}
      {renderNode(
        'start',
        'Evaluate Model Performance',
        <div>
          <p>Start by evaluating your model's performance metrics:</p>
          <ul>
            <li>Accuracy: {modelInfo ? `${(modelInfo.accuracy * 100).toFixed(2)}%` : 'N/A'}</li>
            <li>Error Rate: {errorAnalysis ? `${(errorAnalysis.error_rate * 100).toFixed(2)}%` : 'N/A'}</li>
            <li>Bias/Variance Status: {activeIssue === 'high_bias' ? 'High Bias' : 
                                     activeIssue === 'high_variance' ? 'High Variance' : 'Balanced'}</li>
          </ul>
        </div>,
        getModelEvaluationCode(),
        true
      )}

      {renderConnector('branch')}
      
      <div className="flowchart-branches">
        {/* Left branch - High Bias */}
        <div className="flowchart-branch">
          {renderNode(
            'high_bias',
            'High Bias (Underfitting)',
            <div>
              <p>Your model seems to have high bias (underfitting). This means it's too simple to capture the underlying patterns.</p>
              <p><strong>Symptoms:</strong></p>
              <ul>
                <li>Low training and validation accuracy</li>
                <li>Similar performance on training and validation sets</li>
              </ul>
            </div>,
            null,
            activeIssue === 'high_bias'
          )}
          
          {renderConnector()}
          
          {renderNode(
            'increase_complexity',
            'Increase Model Complexity',
            <div>
              <p>Try increasing your model's complexity:</p>
              <ul>
                <li>Add more layers or neurons</li>
                <li>Use a more powerful model architecture</li>
                <li>Reduce regularization if it's too strong</li>
              </ul>
            </div>,
            getIncreaseComplexityCode(),
            activeIssue === 'high_bias'
          )}
          
          {renderConnector()}
          
          {renderNode(
            'feature_engineering',
            'Feature Engineering',
            <div>
              <p>Create better features to help your model learn:</p>
              <ul>
                <li>Add polynomial features</li>
                <li>Create interaction terms</li>
                <li>Apply domain-specific transformations</li>
                <li>Use automated feature engineering tools</li>
              </ul>
            </div>,
            `# Feature engineering for better representation
from sklearn.preprocessing import PolynomialFeatures

# Add polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Train model on enhanced features
model.fit(X_train_poly, y_train)

# For domain-specific features, create custom transformations
def create_custom_features(X):
    # Example: ratio of feature 0 to feature 1
    X_new = X.copy()
    X_new['ratio_0_1'] = X['feature_0'] / (X['feature_1'] + 1e-6)
    # Add more domain-specific features
    return X_new`,
            activeIssue === 'high_bias'
          )}
        </div>
        
        {/* Middle branch - High Variance */}
        <div className="flowchart-branch">
          {renderNode(
            'high_variance',
            'High Variance (Overfitting)',
            <div>
              <p>Your model seems to have high variance (overfitting). This means it's too complex and capturing noise.</p>
              <p><strong>Symptoms:</strong></p>
              <ul>
                <li>High training accuracy but lower validation accuracy</li>
                <li>Large gap between training and validation performance</li>
              </ul>
            </div>,
            null,
            activeIssue === 'high_variance'
          )}
          
          {renderConnector()}
          
          {renderNode(
            'add_regularization',
            'Add Regularization',
            <div>
              <p>Apply regularization techniques to reduce overfitting:</p>
              <ul>
                <li>L1/L2 regularization</li>
                <li>Dropout</li>
                <li>Early stopping</li>
                <li>Batch normalization</li>
              </ul>
            </div>,
            getRegularizationCode(),
            activeIssue === 'high_variance'
          )}
          
          {renderConnector()}
          
          {renderNode(
            'more_data',
            'Get More Training Data',
            <div>
              <p>Increase your training data to help the model generalize:</p>
              <ul>
                <li>Collect more real data</li>
                <li>Use data augmentation techniques</li>
                <li>Generate synthetic samples</li>
                <li>Use transfer learning from related tasks</li>
              </ul>
            </div>,
            `# Data augmentation for image data (PyTorch)
from torchvision import transforms

transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomAffine(0, translate=(0.1, 0.1)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# For tabular data, use SMOTE for synthetic samples
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)`,
            activeIssue === 'high_variance'
          )}
        </div>
        
        {/* Right branch - Class Imbalance */}
        <div className="flowchart-branch">
          {renderNode(
            'class_imbalance',
            'Class Imbalance',
            <div>
              <p>Your model may be affected by class imbalance. This means some classes have many more samples than others.</p>
              <p><strong>Symptoms:</strong></p>
              <ul>
                <li>Good accuracy but poor precision/recall for minority classes</li>
                <li>Most errors occur in underrepresented classes</li>
              </ul>
            </div>,
            null,
            activeIssue === 'class_imbalance'
          )}
          
          {renderConnector()}
          
          {renderNode(
            'sampling_techniques',
            'Sampling Techniques',
            <div>
              <p>Balance your dataset using sampling techniques:</p>
              <ul>
                <li>Oversampling minority classes</li>
                <li>Undersampling majority classes</li>
                <li>SMOTE (Synthetic Minority Over-sampling Technique)</li>
                <li>Hybrid approaches</li>
              </ul>
            </div>,
            `# Handle class imbalance with sampling techniques
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek

# Oversample minority classes
oversampler = RandomOverSampler(random_state=42)
X_over, y_over = oversampler.fit_resample(X_train, y_train)

# Use SMOTE to generate synthetic samples
smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X_train, y_train)

# Combine over and under sampling (SMOTETomek)
combined = SMOTETomek(random_state=42)
X_combined, y_combined = combined.fit_resample(X_train, y_train)

# Choose the best approach based on validation performance`,
            activeIssue === 'class_imbalance'
          )}
          
          {renderConnector()}
          
          {renderNode(
            'class_weights',
            'Class Weights & Metrics',
            <div>
              <p>Adjust the learning process to handle imbalance:</p>
              <ul>
                <li>Use class weights in loss function</li>
                <li>Use F1-score or AUC-ROC instead of accuracy</li>
                <li>Adjust decision threshold</li>
                <li>Use focal loss for extreme imbalance</li>
              </ul>
            </div>,
            getClassImbalanceCode(),
            activeIssue === 'class_imbalance'
          )}
        </div>
      </div>
      
      {renderConnector()}
      
      {/* Common optimization node */}
      {renderNode(
        'optimize',
        'Fine-tune & Optimize',
        <div>
          <p>Once you've addressed the main issues, optimize your model:</p>
          <ul>
            <li>Tune hyperparameters systematically</li>
            <li>Try ensemble methods</li>
            <li>Implement cross-validation</li>
            <li>Consider advanced architectures</li>
          </ul>
        </div>,
        `# Hyperparameter tuning
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# Define parameter grid
param_grid = {
    'learning_rate': [0.001, 0.01, 0.1],
    'hidden_size': [64, 128, 256],
    'dropout': [0.3, 0.5, 0.7],
    # Add model-specific parameters
}

# Randomized search (for large parameter spaces)
search = RandomizedSearchCV(
    model,
    param_distributions=param_grid,
    n_iter=20,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42
)
search.fit(X_train, y_train)
best_params = search.best_params_

# Create ensemble of models
from sklearn.ensemble import VotingClassifier
ensemble = VotingClassifier(
    estimators=[
        ('model1', model1),
        ('model2', model2),
        ('model3', model3)
    ],
    voting='soft'  # Use predicted probabilities
)
ensemble.fit(X_train, y_train)`,
        activeIssue === 'optimize'
      )}
    </div>
  );
};

export default ModelImprovementFlowchart;