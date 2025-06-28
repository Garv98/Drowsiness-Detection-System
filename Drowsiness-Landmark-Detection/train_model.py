import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
import lightgbm as lgb
import joblib
import seaborn as sns
from pathlib import Path

def load_and_preprocess_data(features_path):
    """Load and preprocess the feature data."""
    print("Loading and preprocessing data...")
    df = pd.read_csv(features_path)
    
    # Separate features and target
    feature_cols = ['ear', 'left_ear', 'right_ear', 'pitch', 'yaw', 'roll', 'perclos']
    X = df[feature_cols]
    # Convert state to binary (0: alert, 1: drowsy)
    y = (df['state'] == 'drowsy').astype(int)
    
    # Handle missing values
    X = X.fillna(X.mean())
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Save scaler for inference
    joblib.dump(scaler, 'model/scaler.pkl')
    
    return X_scaled, y, feature_cols

def plot_learning_curves(model, X, y, title):
    """Plot learning curves to analyze bias-variance tradeoff."""
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=5, n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='accuracy'
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, label='Training score')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
    plt.plot(train_sizes, val_mean, label='Cross-validation score')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1)
    
    plt.xlabel('Training Examples')
    plt.ylabel('Score')
    plt.title(f'Learning Curves ({title})')
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig(f'model/learning_curve_{title.lower().replace(" ", "_")}.png')
    plt.close()

def plot_roc_curves(models_dict, X_test, y_test):
    """Plot ROC curves for all models."""
    plt.figure(figsize=(10, 6))
    
    for name, model in models_dict.items():
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison')
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig('model/roc_curves_comparison.png')
    plt.close()

def train_and_evaluate_models(X, y, feature_names):
    """Train and evaluate multiple models."""
    print("\nTraining and evaluating models...")
    print(f"Total samples: {len(X)}")
    print(f"Features used: {feature_names}")
    print(f"Class distribution: {pd.Series(y).value_counts().to_dict()}\n")
    
    # Calculate class weights for imbalanced data
    class_weights = dict(enumerate(compute_class_weight('balanced', classes=np.unique(y), y=y)))
    
    # Initialize models with tuned hyperparameters
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight=class_weights,
            random_state=42
        ),
        'XGBoost': xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=class_weights[1]/class_weights[0],
            random_state=42
        ),
        'LightGBM': lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight=class_weights,
            random_state=42
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        )
    }
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    best_score = 0
    best_model = None
    best_model_name = None
    trained_models = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Train model
        model.fit(X_train, y_train)
        trained_models[name] = model
        
        # Generate learning curves
        plot_learning_curves(model, X_train, y_train, name)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        print(f"{name} CV Score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        # Evaluate on test set
        score = model.score(X_test, y_test)
        print(f"{name} Test Score: {score:.3f}")
        
        if score > best_score:
            best_score = score
            best_model = model
            best_model_name = name
    
    # Plot ROC curves comparison
    plot_roc_curves(trained_models, X_test, y_test)
    
    return best_model, best_model_name, (X_test, y_test)

def analyze_feature_importance(model, feature_names, model_name):
    """Analyze and plot feature importance."""
    print("Analyzing feature importance...")
    
    if model_name == 'XGBoost':
        importance = model.feature_importances_
    else:
        importance = model.feature_importances_
    
    # Create feature importance DataFrame
    feature_imp = pd.DataFrame({'feature': feature_names, 'importance': importance})
    feature_imp = feature_imp.sort_values('importance', ascending=False)
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_imp)
    plt.title(f'Feature Importance ({model_name})')
    plt.tight_layout()
    plt.savefig('model/feature_importance.png')
    plt.close()

def evaluate_best_model(model, test_data, model_name):
    """Detailed evaluation of the best model."""
    print(f"Evaluating best model: {model_name}")
    X_test, y_test = test_data
    
    # Generate predictions
    y_pred = model.predict(X_test)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix ({model_name})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('model/confusion_matrix.png')
    plt.close()

def save_model(model, model_name):
    """Save the trained model."""
    print("Saving model...")
    model_path = Path('model')
    model_path.mkdir(exist_ok=True)
    joblib.dump(model, model_path / 'best_model.pkl')
    
    # Save model info
    with open(model_path / 'model_info.txt', 'w') as f:
        f.write(f"Best Model: {model_name}\n")
        f.write(f"Saved on: {pd.Timestamp.now()}")

def main():
    """Main function to run the model training pipeline."""
    # Create model directory if it doesn't exist
    Path('model').mkdir(exist_ok=True)
    
    # Load and preprocess data
    features_path = Path('../Data/features.csv')  # Update path to match your workspace structure
    if not features_path.exists():
        features_path = Path('features.csv')  # Try in root directory if not found
    
    if not features_path.exists():
        raise FileNotFoundError(
            "features.csv not found! Please run the feature extraction process first "
            "to generate the features file."
        )
    
    X, y, feature_names = load_and_preprocess_data(features_path)
    
    # Train and evaluate models
    best_model, best_model_name, test_data = train_and_evaluate_models(X, y, feature_names)
    
    # Analyze feature importance
    analyze_feature_importance(best_model, feature_names, best_model_name)
    
    # Evaluate best model
    evaluate_best_model(best_model, test_data, best_model_name)
    
    # Save the model
    save_model(best_model, best_model_name)
    
    print("\nModel training and evaluation complete!")
    print("Check the 'model' directory for:")
    print("- best_model.pkl: The trained model")
    print("- scaler.pkl: Feature scaler")
    print("- feature_importance.png: Feature importance plot")
    print("- confusion_matrix.png: Confusion matrix plot")
    print("- model_info.txt: Model metadata")

if __name__ == "__main__":
    main()
