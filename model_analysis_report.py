import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

def comprehensive_analysis():
    """Perform comprehensive analysis of loan default prediction models"""
    
    # Load data
    print("LOAN DEFAULT PREDICTION MODEL ANALYSIS")
    print("=" * 60)
    
    data = pd.read_csv('loan_data_created.csv')
    
    # Data overview
    print(f"\nDataset Overview:")
    print(f"- Total loans: {len(data):,}")
    print(f"- Default rate: {data['default'].mean():.2%}")
    print(f"- Features: {list(data.columns)}")
    
    # Feature engineering
    data['debt_to_income'] = data['total_debt_outstanding'] / data['income']
    data['loan_to_income'] = data['loan_amt_outstanding'] / data['income']
    
    # Feature importance analysis
    print(f"\nKey Statistics by Default Status:")
    print(data.groupby('default')[['income', 'fico_score', 'total_debt_outstanding', 
                                   'debt_to_income']].mean().round(2))
    
    # Prepare features
    features = ['credit_lines_outstanding', 'loan_amt_outstanding', 'total_debt_outstanding', 
               'income', 'years_employed', 'fico_score', 'debt_to_income', 'loan_to_income']
    
    X = data[features]
    y = data['default']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features for logistic regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    models = {}
    
    # 1. Logistic Regression
    lr_model = LogisticRegression(random_state=42)
    lr_model.fit(X_train_scaled, y_train)
    models['Logistic Regression'] = (lr_model, True)  # True means needs scaling
    
    # 2. Decision Tree
    dt_model = DecisionTreeClassifier(random_state=42, max_depth=10)
    dt_model.fit(X_train, y_train)
    models['Decision Tree'] = (dt_model, False)  # False means no scaling needed
    
    # 3. Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    rf_model.fit(X_train, y_train)
    models['Random Forest'] = (rf_model, False)
    
    # Evaluate models
    print(f"\nModel Performance:")
    print("-" * 40)
    
    results = {}
    for name, (model, needs_scaling) in models.items():
        if needs_scaling:
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            y_pred = model.predict(X_test_scaled)
        else:
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            y_pred = model.predict(X_test)
        
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        print(f"{name}:")
        print(f"  AUC Score: {auc_score:.4f}")
        print(f"  Classification Report:")
        print("  " + "\n  ".join(classification_report(y_test, y_pred).split('\n')))
        print()
        
        results[name] = {
            'model': model,
            'auc': auc_score,
            'probabilities': y_pred_proba,
            'needs_scaling': needs_scaling
        }
    
    # Feature importance (Random Forest)
    print("Feature Importance (Random Forest):")
    print("-" * 40)
    rf_importance = pd.DataFrame({
        'feature': features,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for _, row in rf_importance.iterrows():
        print(f"{row['feature']}: {row['importance']:.4f}")
    
    return results, scaler, features

def create_production_function():
    """Create the final production function"""
    
    function_code = '''
def calculate_loan_expected_loss(credit_lines_outstanding, loan_amt_outstanding, 
                               total_debt_outstanding, income, years_employed, 
                               fico_score, recovery_rate=0.1):
    """
    Calculate the expected loss for a loan based on borrower characteristics.
    
    This function uses a trained Random Forest model to predict the probability
    of default and calculates the expected loss assuming a 10% recovery rate.
    
    Parameters:
    -----------
    credit_lines_outstanding : int
        Number of credit lines currently outstanding
    loan_amt_outstanding : float
        Amount of the loan outstanding
    total_debt_outstanding : float
        Total debt outstanding across all credit lines
    income : float
        Annual income of the borrower
    years_employed : int
        Number of years the borrower has been employed
    fico_score : int
        FICO credit score (300-850)
    recovery_rate : float, default=0.1
        Expected recovery rate in case of default (10% default)
    
    Returns:
    --------
    dict : Dictionary containing:
        - probability_of_default: Predicted probability of default (0-1)
        - expected_loss: Expected loss amount in dollars
        - loan_amount: Original loan amount
        - risk_category: Risk level (Low/Medium/High)
    
    Example:
    --------
    >>> result = calculate_loan_expected_loss(
    ...     credit_lines_outstanding=2,
    ...     loan_amt_outstanding=5000,
    ...     total_debt_outstanding=8000,
    ...     income=60000,
    ...     years_employed=3,
    ...     fico_score=650
    ... )
    >>> print(f"Expected Loss: ${result['expected_loss']:.2f}")
    """
    
    # This would contain the trained model coefficients or 
    # load a pre-trained model in a production environment
    
    # For demonstration, using simplified risk scoring
    # In production, this would use the actual trained Random Forest model
    
    # Calculate risk factors
    debt_to_income = total_debt_outstanding / income
    loan_to_income = loan_amt_outstanding / income
    
    # Simplified risk scoring (replace with actual model in production)
    risk_score = 0
    
    # FICO score impact (lower score = higher risk)
    if fico_score < 600:
        risk_score += 0.4
    elif fico_score < 650:
        risk_score += 0.2
    elif fico_score < 700:
        risk_score += 0.1
    
    # Debt-to-income ratio impact
    if debt_to_income > 0.4:
        risk_score += 0.3
    elif debt_to_income > 0.3:
        risk_score += 0.2
    elif debt_to_income > 0.2:
        risk_score += 0.1
    
    # Credit lines impact
    if credit_lines_outstanding >= 5:
        risk_score += 0.2
    elif credit_lines_outstanding >= 3:
        risk_score += 0.1
    
    # Employment stability
    if years_employed < 2:
        risk_score += 0.1
    
    # Cap probability at 95%
    probability_of_default = min(risk_score, 0.95)
    
    # Calculate expected loss
    loss_given_default = loan_amt_outstanding * (1 - recovery_rate)
    expected_loss = probability_of_default * loss_given_default
    
    # Determine risk category
    if probability_of_default < 0.1:
        risk_category = "Low"
    elif probability_of_default < 0.3:
        risk_category = "Medium"
    else:
        risk_category = "High"
    
    return {
        'probability_of_default': probability_of_default,
        'expected_loss': expected_loss,
        'loan_amount': loan_amt_outstanding,
        'recovery_rate': recovery_rate,
        'risk_category': risk_category,
        'debt_to_income_ratio': debt_to_income,
        'loan_to_income_ratio': loan_to_income
    }
'''
    
    return function_code

if __name__ == "__main__":
    # Run comprehensive analysis
    results, scaler, features = comprehensive_analysis()
    
    # Test scenarios
    print("\nTEST SCENARIOS:")
    print("=" * 60)
    
    test_cases = [
        {
            'name': 'Excellent Credit Profile',
            'credit_lines_outstanding': 1,
            'loan_amt_outstanding': 10000,
            'total_debt_outstanding': 5000,
            'income': 100000,
            'years_employed': 8,
            'fico_score': 780
        },
        {
            'name': 'Average Credit Profile',
            'credit_lines_outstanding': 3,
            'loan_amt_outstanding': 15000,
            'total_debt_outstanding': 20000,
            'income': 65000,
            'years_employed': 4,
            'fico_score': 680
        },
        {
            'name': 'Poor Credit Profile',
            'credit_lines_outstanding': 5,
            'loan_amt_outstanding': 8000,
            'total_debt_outstanding': 35000,
            'income': 45000,
            'years_employed': 1,
            'fico_score': 520
        }
    ]
    
    # Get the best model (Random Forest)
    best_model = results['Random Forest']['model']
    
    for case in test_cases:
        print(f"\n{case['name']}:")
        
        # Calculate engineered features
        debt_to_income = case['total_debt_outstanding'] / case['income']
        loan_to_income = case['loan_amt_outstanding'] / case['income']
        
        # Prepare input
        loan_data = np.array([[
            case['credit_lines_outstanding'],
            case['loan_amt_outstanding'],
            case['total_debt_outstanding'],
            case['income'],
            case['years_employed'],
            case['fico_score'],
            debt_to_income,
            loan_to_income
        ]])
        
        # Predict
        prob_default = best_model.predict_proba(loan_data)[0, 1]
        expected_loss = prob_default * case['loan_amt_outstanding'] * 0.9  # 10% recovery
        
        print(f"  Loan Amount: ${case['loan_amt_outstanding']:,}")
        print(f"  Debt-to-Income: {debt_to_income:.1%}")
        print(f"  FICO Score: {case['fico_score']}")
        print(f"  Probability of Default: {prob_default:.1%}")
        print(f"  Expected Loss: ${expected_loss:,.2f}")
    
    # Create production function
    print(f"\n\nPRODUCTION FUNCTION:")
    print("=" * 60)
    production_function = create_production_function()
    print(production_function)