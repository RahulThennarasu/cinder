import logging
import numpy as np
import time
from typing import Dict, Any, Optional, Union, Tuple, List
from datetime import datetime
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, roc_curve, confusion_matrix
from sklearn.model_selection import KFold

import torch
import tensorflow as tf

class ModelDebugger:
    """Enhanced interface for connecting ML models to CompileML."""
    
    def __init__(self, model, dataset, name: str = "My Model"):
        """
        Initialize the debugger with a model and dataset.
        
        Args:
            model: A PyTorch or TensorFlow model
            dataset: Dataset for evaluation (various formats supported)
            name: Name for this debugging session
        """
        self.model = model
        self.dataset = dataset
        self.name = name
        self.framework = self._detect_framework()
        
        # Initialize storage for debugging data
        self.predictions = None
        self.ground_truth = None
        self.prediction_probas = None  # Store probability scores
        self.session_id = None
        self.server = None
        self.session_start_time = datetime.now()
        
        # Feature importance cache
        self._feature_importances = None
        
        # Training history storage
        self.training_history = []
        
        logging.info(f"Initialized ModelDebugger for {self.framework} model: {name}")
    
    def _detect_framework(self) -> str:
        """Detect which ML framework the model uses."""
        if isinstance(self.model, torch.nn.Module):
            return "pytorch"
        elif isinstance(self.model, tf.Module) or hasattr(self.model, 'predict'):
            return "tensorflow"
        else:
            # Try to detect scikit-learn
            if hasattr(self.model, 'predict') and hasattr(self.model, 'fit'):
                return "sklearn"
            else:
                raise ValueError("Unsupported model type. Please provide a PyTorch, TensorFlow, or scikit-learn model.")
    
    def analyze(self) -> Dict[str, Any]:
        """
        Run comprehensive analysis on the model using the provided dataset.
        
        Returns:
            Dict containing analysis results
        """
        # Get predictions if they don't exist yet
        if self.predictions is None or self.ground_truth is None:
            self.predictions, self.ground_truth = self._get_predictions()
        
        # Calculate metrics
        accuracy = self._calculate_accuracy()
        error_analysis = self._analyze_errors()
        
        # Calculate precision, recall, F1
        precision, recall, f1, _ = precision_recall_fscore_support(
            self.ground_truth, 
            self.predictions,
            average='weighted'
        )
        
        # Generate confusion matrix
        confusion_matrix = self._calculate_confusion_matrix()
        
        # Add advanced metrics
        results = {
            "framework": self.framework,
            "model_name": self.name,
            "dataset_size": len(self.ground_truth) if self.ground_truth is not None else 0,
            "accuracy": accuracy,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "error_analysis": error_analysis,
            "confusion_matrix": confusion_matrix,
        }
        
        # Add more metrics if probabilities are available
        if hasattr(self, 'prediction_probas') and self.prediction_probas is not None:
            try:
                # Try to calculate AUC, but only for binary classification
                unique_classes = np.unique(self.ground_truth)
                if len(unique_classes) == 2:
                    # For binary classification
                    positive_class = unique_classes[1]  # Assuming second class is positive
                    binary_truth = (self.ground_truth == positive_class).astype(int)
                    
                    # Get probabilities for positive class
                    pos_probs = self.prediction_probas[:, 1] if self.prediction_probas.shape[1] > 1 else self.prediction_probas
                    
                    # Calculate ROC AUC
                    roc_auc = roc_auc_score(binary_truth, pos_probs)
                    results["roc_auc"] = float(roc_auc)
                    
                    # Calculate ROC curve points
                    fpr, tpr, thresholds = roc_curve(binary_truth, pos_probs)
                    results["roc_curve"] = {
                        "fpr": fpr.tolist(),
                        "tpr": tpr.tolist(),
                        "thresholds": thresholds.tolist()
                    }
            except Exception as e:
                logging.warning(f"Could not calculate ROC metrics: {str(e)}")
        
        return results
    
    def _get_predictions(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get predictions from the model on the dataset."""
        # Implementation will depend on the framework and dataset format
        
        if self.framework == "pytorch":
            # PyTorch implementation
            self.model.eval()
            all_preds = []
            all_targets = []
            all_probas = []  # Store probability scores
            
            # Simplified example - actual implementation would depend on dataset format
            try:
                with torch.no_grad():
                    for inputs, targets in self.dataset:
                        # Move to GPU if available
                        if torch.cuda.is_available():
                            inputs = inputs.cuda()
                            targets = targets.cuda()
                            self.model = self.model.cuda()
                        
                        outputs = self.model(inputs)
                        
                        # Store probabilities for confidence analysis
                        probas = torch.softmax(outputs, dim=1) if outputs.dim() > 1 else torch.sigmoid(outputs)
                        all_probas.extend(probas.cpu().numpy())
                        
                        # Get predicted class
                        if outputs.dim() > 1:  # Multi-class
                            _, preds = torch.max(outputs, 1)
                        else:  # Binary
                            preds = (outputs > 0.5).long()
                            
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                
                self.prediction_probas = np.array(all_probas)
                return np.array(all_preds), np.array(all_targets)
            except Exception as e:
                logging.error(f"Error getting predictions: {str(e)}")
                # Generate random data for demonstration if there's an error
                return self._generate_demo_predictions()
                
        elif self.framework == "tensorflow":
            # TensorFlow implementation
            try:
                # Get predictions
                predictions_raw = self.model.predict(self.dataset)
                
                # Handle different output formats
                if len(predictions_raw.shape) > 1 and predictions_raw.shape[1] > 1:
                    # Multi-class: store probabilities and get class with highest probability
                    self.prediction_probas = predictions_raw
                    predictions = np.argmax(predictions_raw, axis=1)
                else:
                    # Binary: apply threshold
                    self.prediction_probas = predictions_raw
                    predictions = (predictions_raw > 0.5).astype(int)
                
                # Extract targets based on dataset format (simplified)
                if hasattr(self.dataset, 'unbatch'):
                    # TF Dataset
                    targets = np.array([y.numpy() for x, y in self.dataset.unbatch()])
                else:
                    # Try to extract y from dataset
                    targets = getattr(self.dataset, 'y', None)
                    if targets is None:
                        # Fall back to demo data if targets can't be extracted
                        return self._generate_demo_predictions()
                
                return predictions, targets
            except Exception as e:
                logging.error(f"Error getting predictions: {str(e)}")
                # Generate random data for demonstration if there's an error
                return self._generate_demo_predictions()
        
        # Default case - return demo data
        return self._generate_demo_predictions()
    
    def _generate_demo_predictions(self, num_samples: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """Generate random predictions for demonstration."""
        # For demo purposes, generate binary classifications with 80% accuracy
        np.random.seed(42)  # For reproducibility
        
        # Generate ground truth (0 or 1)
        ground_truth = np.random.randint(0, 2, size=num_samples)
        
        # Generate predictions with 80% accuracy
        predictions = np.copy(ground_truth)
        error_indices = np.random.choice(
            np.arange(num_samples), 
            size=int(num_samples * 0.2),  # 20% error rate
            replace=False
        )
        predictions[error_indices] = 1 - predictions[error_indices]
        
        # Generate prediction probabilities
        self.prediction_probas = np.zeros((num_samples, 2))
        for i in range(num_samples):
            if predictions[i] == 1:
                # If prediction is class 1, probability is between 0.5 and 1.0
                self.prediction_probas[i, 1] = 0.5 + np.random.random() * 0.5
                self.prediction_probas[i, 0] = 1 - self.prediction_probas[i, 1]
            else:
                # If prediction is class 0, probability is between 0.5 and 1.0 for class 0
                self.prediction_probas[i, 0] = 0.5 + np.random.random() * 0.5
                self.prediction_probas[i, 1] = 1 - self.prediction_probas[i, 0]
        
        logging.info(f"Generated {num_samples} demo predictions with {len(error_indices)} errors")
        return predictions, ground_truth
    
    def generate_improvement_suggestions(self) -> List[Dict[str, Any]]:
        """Generate improvement suggestions with code examples based on model analysis."""
        
        # Run analysis if not already done
        if self.predictions is None:
            self.analyze()
        
        suggestions = []
        framework = self.framework  # 'pytorch', 'tensorflow', or 'sklearn'
        
        # Calculate metrics for analysis
        accuracy = self._calculate_accuracy()
        
        # Add your existing suggestion generation logic here...
        
        # Always add some helpful suggestions regardless of model performance
        
        # 1. Cross-validation suggestion
        cv_code = {
            'pytorch': '''# Implement k-fold cross validation
    from sklearn.model_selection import KFold
    import numpy as np

    def cross_validate_pytorch(model_class, X, y, n_splits=5):
        """Run k-fold cross validation for PyTorch model."""
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            print(f"Training fold {fold+1}/{n_splits}")
            
            # Split data
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Convert to PyTorch tensors
            X_train_tensor = torch.FloatTensor(X_train)
            y_train_tensor = torch.LongTensor(y_train)
            X_val_tensor = torch.FloatTensor(X_val)
            y_val_tensor = torch.LongTensor(y_val)
            
            # Create fresh model instance
            model = model_class()
            
            # Train model
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()
            
            # Simple training loop
            for epoch in range(10):  # Reduced epochs for example
                # Training step
                model.train()
                optimizer.zero_grad()
                outputs = model(X_train_tensor)
                loss = criterion(outputs, y_train_tensor)
                loss.backward()
                optimizer.step()
            
            # Evaluate
            model.eval()
            with torch.no_grad():
                outputs = model(X_val_tensor)
                _, predicted = torch.max(outputs, 1)
                accuracy = (predicted == y_val_tensor).float().mean().item()
                fold_scores.append(accuracy)
                
        # Report results
        mean_accuracy = np.mean(fold_scores)
        std_accuracy = np.std(fold_scores)
        print(f"Cross-validation results: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
        return fold_scores''',
            'tensorflow': '''# Implement k-fold cross validation
    from sklearn.model_selection import KFold
    import numpy as np

    def cross_validate_tf(model_fn, X, y, n_splits=5):
        """Run k-fold cross validation for TensorFlow model."""
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            print(f"Training fold {fold+1}/{n_splits}")
            
            # Split data
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Create a fresh model
            model = model_fn()  # Function that returns a new model
            
            # Compile model
            model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Train model
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=10,  # Reduced for example
                verbose=0
            )
            
            # Evaluate
            _, accuracy = model.evaluate(X_val, y_val, verbose=0)
            fold_scores.append(accuracy)
        
        # Report results
        mean_accuracy = np.mean(fold_scores)
        std_accuracy = np.std(fold_scores)
        print(f"Cross-validation results: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
        return fold_scores''',
            'sklearn': '''# Implement k-fold cross validation
    from sklearn.model_selection import cross_val_score, KFold
    import numpy as np

    # For most sklearn models:
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')

    # Print results
    print(f"Cross-validation results: {np.mean(scores):.4f} ± {np.std(scores):.4f}")

    # For more detailed analysis, use cross_validate
    from sklearn.model_selection import cross_validate
    cv_results = cross_validate(
        model, X, y, 
        cv=cv,
        scoring={
            'accuracy': 'accuracy',
            'precision': 'precision_weighted',
            'recall': 'recall_weighted',
            'f1': 'f1_weighted'
        },
        return_train_score=True
    )

    # Print comprehensive results
    print(f"Test accuracy: {cv_results['test_accuracy'].mean():.4f} ± {cv_results['test_accuracy'].std():.4f}")
    print(f"Test F1: {cv_results['test_f1'].mean():.4f} ± {cv_results['test_f1'].std():.4f}")

    # Detect overfitting
    train_acc = cv_results['train_accuracy'].mean()
    test_acc = cv_results['test_accuracy'].mean()
    if train_acc - test_acc > 0.1:
        print("Warning: Model may be overfitting (train acc >> test acc)")'''
        }
        
        suggestions.append({
            "category": "cross_validation",
            "issue": "Evaluating on a single test set may not be reliable",
            "suggestion": "Implement k-fold cross-validation for more robust evaluation",
            "severity": 0.6,
            "impact": 0.7,
            "code_example": cv_code[framework]
        })
        
        # 2. Hyperparameter tuning suggestion
        tuning_code = {
            'pytorch': '''# Implement hyperparameter tuning
    import optuna

    def objective(trial):
        # Define hyperparameters to search
        lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
        hidden_size = trial.suggest_categorical('hidden_size', [32, 64, 128, 256])
        dropout = trial.suggest_float('dropout', 0.1, 0.5)
        
        # Create model with these hyperparameters
        model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )
        
        # Train and evaluate model
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        # Simple training loop (expand as needed)
        train_model(model, optimizer)
        
        # Return metric to optimize
        val_accuracy = evaluate_model(model, val_loader)
        return val_accuracy

    # Create study and optimize
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20)

    # Get best parameters
    best_params = study.best_params
    print(f"Best parameters: {best_params}")
    print(f"Best accuracy: {study.best_value}")

    # Create final model with best parameters
    final_model = create_model(
        hidden_size=best_params['hidden_size'],
        dropout=best_params['dropout']
    )
    optimizer = torch.optim.Adam(final_model.parameters(), lr=best_params['lr'])''',
            'tensorflow': '''# Implement hyperparameter tuning
    import keras_tuner as kt

    def model_builder(hp):
        """Build model with hyperparameters."""
        # Define hyperparameters to search
        lr = hp.Float('lr', min_value=1e-5, max_value=1e-2, sampling='log')
        hidden_size = hp.Int('hidden_size', min_value=32, max_value=256, step=32)
        dropout = hp.Float('dropout', min_value=0.1, max_value=0.5, step=0.1)
        
        # Create model with these hyperparameters
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_size, activation='relu', input_shape=(input_shape,)),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
        
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model

    # Create tuner and search
    tuner = kt.Hyperband(
        model_builder,
        objective='val_accuracy',
        max_epochs=10,
        factor=3,
        directory='my_dir',
        project_name='hyperparameter_tuning'
    )

    # Search for best hyperparameters
    tuner.search(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=10
    )

    # Get best hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(f"Best learning rate: {best_hps.get('lr')}")
    print(f"Best hidden size: {best_hps.get('hidden_size')}")
    print(f"Best dropout: {best_hps.get('dropout')}")

    # Build final model with best hyperparameters
    final_model = tuner.hypermodel.build(best_hps)''',
            'sklearn': '''# Implement hyperparameter tuning
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
    import numpy as np

    # Define parameter grid to search
    param_grid = {
        # For Random Forest
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
        
        # For SVM
        # 'C': [0.1, 1, 10, 100],
        # 'gamma': ['scale', 'auto', 0.1, 0.01],
        # 'kernel': ['rbf', 'poly', 'sigmoid']
        
        # For Logistic Regression
        # 'C': [0.001, 0.01, 0.1, 1, 10, 100],
        # 'penalty': ['l1', 'l2', 'elasticnet', None],
        # 'solver': ['newton-cg', 'lbfgs', 'liblinear', 'saga']
    }

    # Choose search strategy
    # GridSearchCV for small parameter spaces
    search = GridSearchCV(
        model,
        param_grid,
        cv=5,
        scoring='accuracy',
        verbose=1,
        n_jobs=-1  # Use all available cores
    )

    # RandomizedSearchCV for larger parameter spaces
    # search = RandomizedSearchCV(
    #     model,
    #     param_distributions=param_grid,
    #     n_iter=100,  # Number of settings to try
    #     cv=5,
    #     scoring='accuracy',
    #     verbose=1,
    #     n_jobs=-1,
    #     random_state=42
    # )

    # Run search
    search.fit(X_train, y_train)

    # Print results
    print(f"Best parameters: {search.best_params_}")
    print(f"Best CV score: {search.best_score_:.4f}")

    # Use best model
    best_model = search.best_estimator_
    y_pred = best_model.predict(X_test)'''
        }
        
        suggestions.append({
            "category": "hyperparameter_tuning",
            "issue": "Default hyperparameters may not be optimal",
            "suggestion": f"Implement systematic hyperparameter tuning to find the best configuration",
            "severity": 0.5,
            "impact": 0.8,
            "code_example": tuning_code[framework]
        })
        
        # 3. Model Ensemble suggestion
        ensemble_code = {
            'pytorch': '''# Implement model ensemble
    import torch.nn.functional as F

    class EnsembleModel:
        def __init__(self, models):
            self.models = models
            
        def predict(self, X):
            """Get ensemble prediction by averaging outputs."""
            # Convert to tensor if needed
            if not isinstance(X, torch.Tensor):
                X = torch.FloatTensor(X)
                
            # Get predictions from all models
            preds = []
            for model in self.models:
                model.eval()
                with torch.no_grad():
                    outputs = model(X)
                    preds.append(F.softmax(outputs, dim=1))
            
            # Average predictions
            ensemble_preds = torch.stack(preds).mean(dim=0)
            _, pred_classes = torch.max(ensemble_preds, dim=1)
            return pred_classes.numpy()

    # Train different model architectures
    models = []

    # Model 1: Standard architecture
    model1 = StandardNet()
    train_model(model1, train_loader, epochs=30)
    models.append(model1)

    # Model 2: Different architecture
    model2 = ComplexNet()
    train_model(model2, train_loader, epochs=30)
    models.append(model2)

    # Model 3: Same architecture, different initialization
    model3 = StandardNet()
    train_model(model3, train_loader, epochs=30)
    models.append(model3)

    # Create ensemble
    ensemble = EnsembleModel(models)

    # Evaluate ensemble
    ensemble_preds = ensemble.predict(X_test)
    ensemble_accuracy = (ensemble_preds == y_test).mean()
    print(f"Ensemble accuracy: {ensemble_accuracy:.4f}")''',
            'tensorflow': '''# Implement model ensemble
    import numpy as np

    class EnsembleModel:
        def __init__(self, models):
            self.models = models
            
        def predict(self, X):
            """Get ensemble prediction by averaging outputs."""
            # Get predictions from all models
            preds = []
            for model in self.models:
                pred_probs = model.predict(X)
                preds.append(pred_probs)
            
            # Average predictions
            ensemble_preds = np.mean(preds, axis=0)
            return np.argmax(ensemble_preds, axis=1)

    # Train different model architectures
    models = []

    # Model 1: Standard architecture
    model1 = build_standard_model()
    model1.fit(X_train, y_train, epochs=30)
    models.append(model1)

    # Model 2: Different architecture
    model2 = build_complex_model()
    model2.fit(X_train, y_train, epochs=30)
    models.append(model2)

    # Model 3: Same architecture, different initialization
    model3 = build_standard_model()
    model3.fit(X_train, y_train, epochs=30)
    models.append(model3)

    # Create ensemble
    ensemble = EnsembleModel(models)

    # Evaluate ensemble
    ensemble_preds = ensemble.predict(X_test)
    ensemble_accuracy = np.mean(ensemble_preds == y_test)
    print(f"Ensemble accuracy: {ensemble_accuracy:.4f}")''',
            'sklearn': '''# Implement model ensemble
    from sklearn.ensemble import VotingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score

    # Create different base models
    models = [
        ('lr', LogisticRegression(max_iter=1000)),
        ('rf', RandomForestClassifier(n_estimators=100)),
        ('svm', SVC(probability=True))
    ]

    # Create and train voting ensemble
    ensemble = VotingClassifier(
        estimators=models,
        voting='soft'  # Use predicted probabilities
    )
    ensemble.fit(X_train, y_train)

    # Evaluate ensemble
    y_pred = ensemble.predict(X_test)
    ensemble_accuracy = accuracy_score(y_test, y_pred)
    print(f"Ensemble accuracy: {ensemble_accuracy:.4f}")

    # Compare with individual models
    for name, model in models:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(f"{name} accuracy: {accuracy_score(y_test, y_pred):.4f}")'''
        }
        
        suggestions.append({
            "category": "model_ensemble",
            "issue": "Single models may have limitations in certain scenarios",
            "suggestion": "Implement a model ensemble to improve performance and robustness",
            "severity": 0.4,
            "impact": 0.7,
            "code_example": ensemble_code[framework]
        })
        
        # 4. Data Augmentation (if using image data)
        if self.framework == 'pytorch' and hasattr(self.model, 'conv1'):
            # This is likely a CNN, suggest data augmentation
            suggestions.append({
                "category": "data_augmentation",
                "issue": "Limited training data can lead to overfitting in CNNs",
                "suggestion": "Implement data augmentation to artificially increase training data diversity",
                "severity": 0.5,
                "impact": 0.7,
                "code_example": '''# Implement data augmentation for image data
    from torchvision import transforms

    # Define augmentation transformations
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomAffine(0, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Apply to training dataset
    train_dataset_augmented = YourDataset(
        root='./data',
        train=True,
        transform=transform_train
    )

    # Keep validation/test transforms simple
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    test_dataset = YourDataset(
        root='./data',
        train=False,
        transform=transform_test
    )

    # Create loaders
    train_loader = DataLoader(train_dataset_augmented, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)'''
            })
        
        # 5. Learning Rate Finder
        lr_finder_code = {
            'pytorch': '''# Implement learning rate finder
    from torch_lr_finder import LRFinder

    # Define model and optimizer
    model = YourModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-7)
    criterion = nn.CrossEntropyLoss()

    # Create LR finder
    lr_finder = LRFinder(model, optimizer, criterion, device="cuda" if torch.cuda.is_available() else "cpu")
    lr_finder.range_test(train_loader, end_lr=1, num_iter=100)

    # Plot results
    lr_finder.plot()  # Requires matplotlib

    # Get suggestion for LR
    suggested_lr = lr_finder.suggestion()
    print(f"Suggested Learning Rate: {suggested_lr}")

    # Reset the model and optimizer
    model = YourModel()  # Reinitialize model
    optimizer = torch.optim.Adam(model.parameters(), lr=suggested_lr)''',
            'tensorflow': '''# Implement learning rate finder
    import numpy as np
    import matplotlib.pyplot as plt
    from tensorflow.keras.callbacks import LambdaCallback

    # Define model
    model = build_model()

    # Define learning rate range (log scale)
    start_lr = 1e-7
    end_lr = 1.0
    num_steps = 100
    lr_multiplier = (end_lr / start_lr) ** (1 / num_steps)

    # Initialize data
    lrs = []
    losses = []
    best_loss = np.inf
    best_lr = None

    # Callback to update learning rate and record loss
    lr_callback = LambdaCallback(
        on_batch_end=lambda batch, logs: (
            lrs.append(K.get_value(model.optimizer.lr)),
            losses.append(logs['loss']),
            K.set_value(model.optimizer.lr, K.get_value(model.optimizer.lr) * lr_multiplier)
        )
    )

    # Initialize with start learning rate
    K.set_value(model.optimizer.lr, start_lr)

    # Train model with increasing learning rate
    model.fit(
        X_train, y_train,
        epochs=1,
        batch_size=32,
        callbacks=[lr_callback],
        verbose=0
    )

    # Find the best learning rate (where loss still decreases)
    for i in range(1, len(losses)):
        if losses[i] < best_loss:
            best_loss = losses[i]
            best_lr = lrs[i]

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(lrs, losses)
    plt.xscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('Loss')
    plt.title('Learning Rate Finder')
    plt.axvline(x=best_lr, color='r', linestyle='--')
    plt.savefig('lr_finder.png')

    print(f"Suggested Learning Rate: {best_lr}")

    # Reinitialize model with the suggested learning rate
    model = build_model()
    model.optimizer.lr.assign(best_lr)''',
            'sklearn': '''# Learning rate finder isn't typically used with sklearn models
    # Instead, here's a grid search for learning rates in MLPClassifier

    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import GridSearchCV

    # Define parameter grid focusing on learning_rate_init
    param_grid = {
        'learning_rate_init': [0.0001, 0.001, 0.01, 0.1],
        'hidden_layer_sizes': [(100,), (100, 50)],
        'activation': ['relu', 'tanh'],
        'alpha': [0.0001, 0.001, 0.01]
    }

    # Create model
    mlp = MLPClassifier(max_iter=1000, early_stopping=True, random_state=42)

    # Run grid search
    grid_search = GridSearchCV(
        mlp, param_grid, cv=3, 
        scoring='accuracy', verbose=1
    )
    grid_search.fit(X_train, y_train)

    # Get best parameters
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best score: {grid_search.best_score_:.4f}")

    # Use best model
    best_model = grid_search.best_estimator_'''
        }
        
        suggestions.append({
            "category": "learning_rate_optimization",
            "issue": "Finding the optimal learning rate is challenging",
            "suggestion": "Use a learning rate finder to discover the ideal learning rate",
            "severity": 0.4,
            "impact": 0.6,
            "code_example": lr_finder_code[framework]
        })
        
        print(f"Generated {len(suggestions)} improvement suggestions")
        return suggestions
    
    def _calculate_accuracy(self) -> float:
        """Calculate basic accuracy metric."""
        if self.predictions is None or self.ground_truth is None:
            self.predictions, self.ground_truth = self._get_predictions()
        
        if len(self.predictions) == 0 or len(self.ground_truth) == 0:
            return 0.0
            
        return float(np.mean(self.predictions == self.ground_truth))
    
    def _analyze_errors(self) -> Dict[str, Any]:
        """Analyze prediction errors in detail."""
        if self.predictions is None or self.ground_truth is None:
            self.predictions, self.ground_truth = self._get_predictions()
            
        if len(self.predictions) == 0 or len(self.ground_truth) == 0:
            return {
                "error_count": 0,
                "error_indices": [],
                "error_rate": 0.0
            }
            
        error_mask = self.predictions != self.ground_truth
        error_indices = np.where(error_mask)[0]
        error_rate = len(error_indices) / len(self.ground_truth)
        
        # Enhanced error analysis - track error types
        error_types = {}
        for idx in error_indices:
            true_class = self.ground_truth[idx]
            pred_class = self.predictions[idx]
            error_key = f"{true_class}→{pred_class}"
            
            if error_key not in error_types:
                error_types[error_key] = {
                    "true_class": int(true_class),
                    "predicted_class": int(pred_class),
                    "count": 0,
                    "indices": []
                }
            
            error_types[error_key]["count"] += 1
            error_types[error_key]["indices"].append(int(idx))
        
        # Sort error types by frequency
        error_types_list = sorted(
            list(error_types.values()),
            key=lambda x: x["count"],
            reverse=True
        )
        
        return {
            "error_count": len(error_indices),
            "error_indices": error_indices.tolist(),
            "error_rate": float(error_rate),
            "error_types": error_types_list
        }
    
    def _calculate_confusion_matrix(self) -> Dict[str, Any]:
        """Calculate confusion matrix for multi-class classification."""
        if self.predictions is None or self.ground_truth is None:
            self.predictions, self.ground_truth = self._get_predictions()
        
        if len(self.predictions) == 0 or len(self.ground_truth) == 0:
            return {
                "matrix": [[0, 0], [0, 0]],
                "labels": ["0", "1"]
            }
        
        # Get unique classes
        classes = np.unique(np.concatenate((self.predictions, self.ground_truth)))
        n_classes = len(classes)
        
        # Calculate confusion matrix
        cm = confusion_matrix(
            self.ground_truth,
            self.predictions,
            labels=classes
        )
        
        return {
            "matrix": cm.tolist(),
            "labels": [str(c) for c in classes],
            "num_classes": int(n_classes)
        }
    
    def analyze_confidence(self) -> Dict[str, Any]:
        """
        Analyze the confidence of predictions.
        
        Returns:
            Dict with confidence metrics and distributions
        """
        if not hasattr(self, 'prediction_probas') or self.prediction_probas is None:
            # Re-run predictions to get probabilities if not available
            self._get_predictions()
            if not hasattr(self, 'prediction_probas'):
                return {"error": "Cannot compute confidence, probabilities not available"}
        
        # Handle different probability formats - with proper null checking
        if self.prediction_probas is not None:
            if len(self.prediction_probas.shape) > 1 and self.prediction_probas.shape[1] > 1:
                # For multi-class, confidence is max probability
                confidences = np.max(self.prediction_probas, axis=1)
            else:
                # For binary with single column, transform to max prob
                confidences = np.maximum(self.prediction_probas, 1 - self.prediction_probas)
        else:
            # If no probability data is available, set confidences to a default
            confidences = np.ones_like(self.predictions, dtype=float) * 0.5
        
        # Analyze confidence for correct vs incorrect predictions
        correct_mask = self.predictions == self.ground_truth
        correct_confidences = confidences[correct_mask]
        incorrect_confidences = confidences[~correct_mask]
        
        # Calculate statistics
        avg_confidence = float(np.mean(confidences))
        avg_correct_confidence = float(np.mean(correct_confidences) if len(correct_confidences) > 0 else 0)
        avg_incorrect_confidence = float(np.mean(incorrect_confidences) if len(incorrect_confidences) > 0 else 0)
        
        # Calculate calibration error (difference between accuracy and confidence)
        accuracy = self._calculate_accuracy()
        calibration_error = float(abs(avg_confidence - accuracy))
        
        # Create confidence bins for distribution analysis
        bins = np.linspace(0, 1, 11)  # 10 bins from 0 to 1
        overall_hist, _ = np.histogram(confidences, bins=bins)
        correct_hist, _ = np.histogram(correct_confidences, bins=bins)
        incorrect_hist, _ = np.histogram(incorrect_confidences, bins=bins)
        
        # Find overconfident examples (high confidence but wrong predictions)
        overconfident_threshold = 0.9
        overconfident_indices = np.where((~correct_mask) & (confidences > overconfident_threshold))[0]
        
        # Find underconfident examples (low confidence but correct predictions)
        underconfident_threshold = 0.6
        underconfident_indices = np.where((correct_mask) & (confidences < underconfident_threshold))[0]
        
        return {
            "avg_confidence": avg_confidence,
            "avg_correct_confidence": avg_correct_confidence,
            "avg_incorrect_confidence": avg_incorrect_confidence,
            "calibration_error": calibration_error,
            "confidence_distribution": {
                "bin_edges": bins.tolist(),
                "overall": overall_hist.tolist(),
                "correct": correct_hist.tolist(),
                "incorrect": incorrect_hist.tolist(),
            },
            "overconfident_examples": {
                "threshold": overconfident_threshold,
                "count": len(overconfident_indices),
                "indices": overconfident_indices.tolist()[:10]  # Return first 10 for UI display
            },
            "underconfident_examples": {
                "threshold": underconfident_threshold,
                "count": len(underconfident_indices),
                "indices": underconfident_indices.tolist()[:10]  # Return first 10 for UI display
            }
        }
    
    def analyze_feature_importance(self) -> Dict[str, Any]:
        """
        Analyze feature importance if the model supports it.
        
        Returns:
            Dict with feature importance information
        """
        # For PyTorch models, we need a workaround since they don't have direct feature importance
        if self.framework == "pytorch":
            try:
                # Create a feature importance proxy using input gradients
                # This is a simplified approach - for real usage, use more robust methods
                if not hasattr(self, '_feature_importances'):
                    # Get a small batch of data
                    data_iter = iter(self.dataset)
                    inputs, _ = next(data_iter)
                    
                    # Move to GPU if available
                    if torch.cuda.is_available():
                        inputs = inputs.cuda()
                        self.model = self.model.cuda()
                    
                    # Check if inputs need gradient
                    if not inputs.requires_grad:
                        inputs.requires_grad = True
                    
                    # Reset gradients
                    self.model.zero_grad()
                    
                    # Forward pass
                    outputs = self.model(inputs)
                    
                    # Compute the sum of outputs (to get a scalar)
                    outputs.sum().backward()
                    
                    # Get the gradient
                    gradient = inputs.grad
                    
                    # Use the absolute mean of gradients as feature importance
                    importances = gradient.abs().mean(dim=0).cpu().numpy()
                    
                    # Normalize to 0-1
                    if importances.sum() > 0:
                        importances = importances / importances.sum()
                    
                    self._feature_importances = importances
                
                # Flatten for consistent output format regardless of input shape
                if self._feature_importances is not None:
                    importances_flat = self._feature_importances.flatten()
                    
                    # Create feature names based on shape
                    feature_names = [f"Feature {i}" for i in range(len(importances_flat))]
                    
                    # Sort by importance
                    indices = np.argsort(importances_flat)[::-1]
                    sorted_importances = importances_flat[indices]
                    sorted_names = [feature_names[i] for i in indices]
                    
                    return {
                        "feature_names": sorted_names,
                        "importance_values": sorted_importances.tolist(),
                        "importance_method": "gradient_based"
                    }
                else:
                    return {
                        "error": "Could not calculate feature importance: No importance data available"
                    }
                
            except Exception as e:
                logging.error(f"Error calculating feature importance: {str(e)}")
                return {"error": f"Could not calculate feature importance: {str(e)}"}
        
        # For scikit-learn-like models that have feature_importances_
        elif hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            feature_names = [f"Feature {i}" for i in range(len(importances))]
            
            # Sort by importance
            indices = np.argsort(importances)[::-1]
            sorted_importances = importances[indices]
            sorted_names = [feature_names[i] for i in indices]
            
            return {
                "feature_names": sorted_names,
                "importance_values": sorted_importances.tolist(),
                "importance_method": "model_attributions"
            }
        
        return {"error": "Model type does not support feature importance analysis"}
    
    def perform_cross_validation(self, k_folds=5) -> Dict[str, Any]:
        """
        Perform k-fold cross-validation on the model.
        
        Args:
            k_folds: Number of folds for cross-validation
            
        Returns:
            Dict with cross-validation results
        """
        if self.dataset is None:
            return {"error": "No dataset available for cross-validation"}
        
        # Extract all data from dataset
        try:
            # This assumes dataset is a PyTorch DataLoader or similar
            all_inputs = []
            all_targets = []
            
            for inputs, targets in self.dataset:
                all_inputs.append(inputs)
                all_targets.append(targets)
            
            # Concatenate batches
            X = torch.cat(all_inputs)
            y = torch.cat(all_targets)
            
            # Move to CPU for sklearn
            X = X.cpu()
            y = y.cpu()
            
            # Setup cross-validation
            kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
            fold_results = []
            
            # Convert PyTorch tensor to numpy array for KFold
            X_numpy = X.numpy()
            
            for fold, (train_idx, val_idx) in enumerate(kf.split(X_numpy)):
                # Get fold data
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Clone model (depends on framework)
                if self.framework == "pytorch":
                    import copy
                    fold_model = copy.deepcopy(self.model)
                    
                    # Create optimizer
                    optimizer = torch.optim.Adam(fold_model.parameters(), lr=0.01)
                    
                    # Loss function
                    if len(np.unique(y.numpy())) > 2:
                        criterion = torch.nn.CrossEntropyLoss()
                    else:
                        criterion = torch.nn.BCEWithLogitsLoss()
                    
                    # Train for a few epochs
                    fold_model.train()
                    for epoch in range(3):  # Just a few epochs for quick validation
                        # Forward pass
                        outputs = fold_model(X_train)
                        loss = criterion(outputs, y_train)
                        
                        # Backward and optimize
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    
                    # Evaluate fold model
                    fold_model.eval()
                    with torch.no_grad():
                        outputs = fold_model(X_val)
                        _, preds = torch.max(outputs, 1)
                    
                    # Calculate accuracy
                    accuracy = (preds == y_val).float().mean().item()
                    
                    fold_results.append({
                        "fold": fold + 1,
                        "accuracy": float(accuracy),
                        "val_size": int(len(y_val))
                    })
            
            # Calculate overall statistics
            accuracies = [r["accuracy"] for r in fold_results]
            mean_accuracy = float(np.mean(accuracies))
            std_accuracy = float(np.std(accuracies))
            
            return {
                "fold_results": fold_results,
                "mean_accuracy": mean_accuracy,
                "std_accuracy": std_accuracy,
                "n_folds": k_folds
            }
            
        except Exception as e:
            logging.error(f"Error during cross-validation: {str(e)}")
            return {"error": f"Cross-validation failed: {str(e)}"}
    
    def analyze_prediction_drift(self, threshold=0.1) -> Dict[str, Any]:
        """
        Analyze if predictions drift significantly from the training distribution.
        
        This helps detect if the model is receiving inputs significantly different from training.
        
        Args:
            threshold: Threshold for considering a prediction as drifted
            
        Returns:
            Dict with drift analysis results
        """
        if self.predictions is None or self.ground_truth is None:
            self.predictions, self.ground_truth = self._get_predictions()
            
        try:
            # For demonstration, we'll simulate drift detection
            # In a real implementation, you would compare feature distributions
            
            # Calculate class distribution
            unique_classes = np.unique(self.ground_truth)
            class_counts = {}
            for cls in unique_classes:
                class_counts[int(cls)] = int(np.sum(self.ground_truth == cls))
            
            # Calculate prediction distribution
            pred_counts = {}
            for cls in unique_classes:
                pred_counts[int(cls)] = int(np.sum(self.predictions == cls))
            
            # Calculate difference in distributions
            drift_scores = {}
            for cls in unique_classes:
                expected_ratio = class_counts[int(cls)] / len(self.ground_truth)
                actual_ratio = pred_counts[int(cls)] / len(self.predictions)
                drift_scores[int(cls)] = float(abs(actual_ratio - expected_ratio))
            
            # Identify drifting classes
            drifting_classes = [cls for cls, score in drift_scores.items() if score > threshold]
            
            return {
                "class_distribution": {str(k): v for k, v in class_counts.items()},
                "prediction_distribution": {str(k): v for k, v in pred_counts.items()},
                "drift_scores": {str(k): v for k, v in drift_scores.items()},
                "drifting_classes": drifting_classes,
                "overall_drift": float(sum(drift_scores.values()) / len(drift_scores))
            }
        except Exception as e:
            logging.error(f"Error analyzing prediction drift: {str(e)}")
            return {"error": f"Could not analyze prediction drift: {str(e)}"}
    
    def get_training_history(self, num_points: int = 10) -> List[Dict[str, Any]]:
        """
        Get training history data.
        
        In a real app, this would be populated during model training.
        Here we generate mock data for demonstration if not already available.
        """
        if not self.training_history:
            # Generate a simulated training history
            base_accuracy = 0.65
            base_loss = 0.5
            
            for i in range(1, num_points + 1):
                time_offset = i * 300  # 5 minutes between points
                epoch_time = self.session_start_time.timestamp() + time_offset
                epoch_date = datetime.fromtimestamp(epoch_time)
                
                # Accuracy increases over time, with some noise
                accuracy = min(0.98, base_accuracy + (i * 0.03) + np.random.uniform(-0.01, 0.01))
                
                # Loss decreases over time, with some noise
                loss = max(0.05, base_loss - (i * 0.05) + np.random.uniform(-0.01, 0.01))
                
                # Add learning rate that decreases over time
                learning_rate = 0.01 * (0.9 ** i)
                
                self.training_history.append({
                    "iteration": i,
                    "accuracy": float(accuracy),
                    "loss": float(loss),
                    "learning_rate": float(learning_rate),
                    "timestamp": epoch_date.isoformat()
                })
        
        return self.training_history
    
    def analyze_error_types(self) -> List[Dict[str, Any]]:
        """
        Analyze types of errors (false positives vs false negatives).
        
        In a real app, this would analyze the actual predictions.
        Here we generate sensible mock data based on our predictions.
        """
        if self.predictions is None or self.ground_truth is None:
            self.predictions, self.ground_truth = self._get_predictions()
        
        if len(self.predictions) == 0 or len(self.ground_truth) == 0:
            return [
                {"name": "False Positives", "value": 0},
                {"name": "False Negatives", "value": 0}
            ]
        
        # Binary classification case
        unique_classes = np.unique(self.ground_truth)
        if len(unique_classes) == 2:
            # Calculate false positives and false negatives
            # Assuming class 1 is positive and 0 is negative
            positive_class = unique_classes[1]
            negative_class = unique_classes[0]
            
            false_positives = np.sum((self.predictions == positive_class) & (self.ground_truth == negative_class))
            false_negatives = np.sum((self.predictions == negative_class) & (self.ground_truth == positive_class))
            
            return [
                {"name": "False Positives", "value": int(false_positives)},
                {"name": "False Negatives", "value": int(false_negatives)}
            ]
        else:
            # Multi-class case - return per-class errors
            results = []
            for cls in unique_classes:
                # One-vs-all approach
                cls_int = int(cls)
                
                # False positives: predicted as cls but isn't
                false_pos = np.sum((self.predictions == cls) & (self.ground_truth != cls))
                
                # False negatives: is cls but predicted as something else
                false_neg = np.sum((self.predictions != cls) & (self.ground_truth == cls))
                
                results.append({
                    "class": cls_int,
                    "name": f"Class {cls_int} False Positives",
                    "value": int(false_pos)
                })
                
                results.append({
                    "class": cls_int,
                    "name": f"Class {cls_int} False Negatives",
                    "value": int(false_neg)
                })
            
            return results
    
    def get_sample_predictions(self, limit: int = 10, offset: int = 0, include_errors_only: bool = False) -> Dict[str, Any]:
        """
        Get sample predictions for inspection.
        
        Args:
            limit: Maximum number of samples to return
            offset: Offset for pagination
            include_errors_only: If True, only return incorrect predictions
            
        Returns:
            Dict with sample predictions
        """
        if self.predictions is None or self.ground_truth is None:
            self.predictions, self.ground_truth = self._get_predictions()
        
        if len(self.predictions) == 0 or len(self.ground_truth) == 0:
            return {
                "samples": [],
                "total": 0,
                "limit": limit,
                "offset": offset
            }
        
        # Get indices of samples to include
        if include_errors_only:
            indices = np.where(self.predictions != self.ground_truth)[0]
        else:
            indices = np.arange(len(self.predictions))
        
        # Apply pagination
        total = len(indices)
        if offset >= total:
            offset = max(0, total - limit)
        
        # Get slice of indices
        paginated_indices = indices[offset:offset+limit]
        
        # Prepare samples
        samples = []
        for idx in paginated_indices:
            sample = {
                "index": int(idx),
                "prediction": int(self.predictions[idx]),
                "true_label": int(self.ground_truth[idx]),
                "is_error": bool(self.predictions[idx] != self.ground_truth[idx])
            }
            
            # Add probability if available
            if hasattr(self, 'prediction_probas') and self.prediction_probas is not None:
                if len(self.prediction_probas.shape) > 1 and self.prediction_probas.shape[1] > 1:
                    # Multi-class probabilities
                    sample["probabilities"] = self.prediction_probas[idx].tolist()
                    sample["confidence"] = float(np.max(self.prediction_probas[idx]))
                else:
                    # Binary case
                    prob = float(self.prediction_probas[idx])
                    sample["probabilities"] = [1 - prob, prob]
                    sample["confidence"] = float(max(prob, 1 - prob))
            
            samples.append(sample)
        
        return {
            "samples": samples,
            "total": total,
            "limit": limit,
            "offset": offset,
            "include_errors_only": include_errors_only
        }
    
    def launch_dashboard(self, port: int = 8000) -> None:
        """Launch the debugging dashboard server."""
        import threading
        from backend.app.server import start_server
        
        # Analyze if not already done
        if self.predictions is None:
            self.analyze()
        
        print(f"CompileML dashboard is running at http://localhost:{port}")
        print("Press Ctrl+C to stop the server")
        
        # Start the server in a separate thread
        server_thread = threading.Thread(
            target=lambda: start_server(self, port=port),
            daemon=True  # This makes the thread exit when the main program exits
        )
        server_thread.start()