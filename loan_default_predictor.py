import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

class LoanDefaultPredictor:
    def __init__(self, recovery_rate=0.1):
        self.recovery_rate = recovery_rate
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def load_and_prepare_data(self, file_path):
        """Load and prepare the loan data"""
        self.data = pd.read_csv(file_path)
        
        # Feature engineering
        self.data['debt_to_income'] = self.data['total_debt_outstanding'] / self.data['income']
        self.data['loan_to_income'] = self.data['loan_amt_outstanding'] / self.data['income']
        
        # Select features for modeling
        features = ['credit_lines_outstanding', 'loan_amt_outstanding', 'total_debt_outstanding', 
                   'income', 'years_employed', 'fico_score', 'debt_to_income', 'loan_to_income']
        
        X = self.data[features]
        y = self.data['default']
        
        self.feature_names = features
        return X, y
    
    def train_models(self, X, y):
        """Train multiple models for comparison"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Store test data for evaluation
        self.X_test = X_test_scaled
        self.y_test = y_test
        
        # 1. Logistic Regression
        lr_model = LogisticRegression(random_state=42)
        lr_model.fit(X_train_scaled, y_train)
        self.models['Logistic Regression'] = lr_model
        
        # 2. Decision Tree
        dt_model = DecisionTreeClassifier(random_state=42, max_depth=10)
        dt_model.fit(X_train, y_train)  # Trees don't need scaling
        self.models['Decision Tree'] = dt_model
        
        # 3. Random Forest
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        rf_model.fit(X_train, y_train)  # Trees don't need scaling
        self.models['Random Forest'] = rf_model
        
        return X_train, X_test, y_train, y_test
    
    def evaluate_models(self):
        """Evaluate all trained models"""
        results = {}
        
        for name, model in self.models.items():
            if name == 'Logistic Regression':
                y_pred_proba = model.predict_proba(self.X_test)[:, 1]
                y_pred = model.predict(self.X_test)
            else:
                # For tree-based models, use original test data (not scaled)
                X_test_orig = self.scaler.inverse_transform(self.X_test)
                y_pred_proba = model.predict_proba(X_test_orig)[:, 1]
                y_pred = model.predict(X_test_orig)
            
            auc_score = roc_auc_score(self.y_test, y_pred_proba)
            
            results[name] = {
                'AUC Score': auc_score,
                'Predictions': y_pred,
                'Probabilities': y_pred_proba
            }
        
        return results
    
    def predict_default_probability(self, loan_features, model_name='Random Forest'):
        """Predict probability of default for a single loan"""
        model = self.models[model_name]
        
        # Ensure input is in correct format
        if isinstance(loan_features, dict):
            # Convert dict to DataFrame with correct feature order
            loan_df = pd.DataFrame([loan_features])
            # Add engineered features
            loan_df['debt_to_income'] = loan_df['total_debt_outstanding'] / loan_df['income']
            loan_df['loan_to_income'] = loan_df['loan_amt_outstanding'] / loan_df['income']
            loan_features = loan_df[self.feature_names].values
        
        if model_name == 'Logistic Regression':
            loan_features_scaled = self.scaler.transform(loan_features.reshape(1, -1))
            prob_default = model.predict_proba(loan_features_scaled)[0, 1]
        else:
            prob_default = model.predict_proba(loan_features.reshape(1, -1))[0, 1]
        
        return prob_default
    
    def calculate_expected_loss(self, loan_amount, prob_default):
        """Calculate expected loss given loan amount and probability of default"""
        loss_given_default = loan_amount * (1 - self.recovery_rate)
        expected_loss = prob_default * loss_given_default
        return expected_loss
    
    def loan_risk_assessment(self, loan_details, model_name='Random Forest'):
        """Complete risk assessment for a loan"""
        # Extract loan amount
        loan_amount = loan_details.get('loan_amt_outstanding', 0)
        
        # Predict probability of default
        prob_default = self.predict_default_probability(loan_details, model_name)
        
        # Calculate expected loss
        expected_loss = self.calculate_expected_loss(loan_amount, prob_default)
        
        return {
            'probability_of_default': prob_default,
            'expected_loss': expected_loss,
            'loan_amount': loan_amount,
            'recovery_rate': self.recovery_rate
        }

def main():
    # Initialize predictor
    predictor = LoanDefaultPredictor(recovery_rate=0.1)
    
    # Load and prepare data
    print("Loading and preparing data...")
    X, y = predictor.load_and_prepare_data('Task 3 and 4_Loan_Data.csv')
    
    print(f"Dataset shape: {X.shape}")
    print(f"Default rate: {y.mean():.2%}")
    
    # Train models
    print("\nTraining models...")
    X_train, X_test, y_train, y_test = predictor.train_models(X, y)
    
    # Evaluate models
    print("\nEvaluating models...")
    results = predictor.evaluate_models()
    
    print("\nModel Performance (AUC Scores):")
    for model_name, metrics in results.items():
        print(f"{model_name}: {metrics['AUC Score']:.4f}")
    
    # Example prediction
    print("\n" + "="*50)
    print("EXAMPLE LOAN RISK ASSESSMENT")
    print("="*50)
    
    # Example loan application
    example_loan = {
        'credit_lines_outstanding': 2,
        'loan_amt_outstanding': 5000,
        'total_debt_outstanding': 8000,
        'income': 60000,
        'years_employed': 3,
        'fico_score': 650
    }
    
    print(f"Loan Details: {example_loan}")
    
    # Assess risk with different models
    for model_name in predictor.models.keys():
        assessment = predictor.loan_risk_assessment(example_loan, model_name)
        print(f"\n{model_name} Assessment:")
        print(f"  Probability of Default: {assessment['probability_of_default']:.2%}")
        print(f"  Expected Loss: ${assessment['expected_loss']:.2f}")
    
    return predictor

if __name__ == "__main__":
    predictor = main()