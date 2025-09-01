import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

def train_and_save_model():
    """Train the model and save it for production use"""
    # Load data
    data = pd.read_csv('loan_data_created.csv')
    
    # Feature engineering
    data['debt_to_income'] = data['total_debt_outstanding'] / data['income']
    data['loan_to_income'] = data['loan_amt_outstanding'] / data['income']
    
    # Features
    features = ['credit_lines_outstanding', 'loan_amt_outstanding', 'total_debt_outstanding', 
               'income', 'years_employed', 'fico_score', 'debt_to_income', 'loan_to_income']
    
    X = data[features]
    y = data['default']
    
    # Train Random Forest (best performing model)
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X, y)
    
    # Save model
    joblib.dump(model, 'loan_default_model.pkl')
    
    return model, features

def calculate_expected_loss(credit_lines_outstanding, loan_amt_outstanding, total_debt_outstanding, 
                          income, years_employed, fico_score, recovery_rate=0.1):
    """
    Calculate expected loss for a loan given borrower characteristics.
    
    Parameters:
    -----------
    credit_lines_outstanding : int
        Number of credit lines outstanding
    loan_amt_outstanding : float
        Loan amount outstanding
    total_debt_outstanding : float
        Total debt outstanding
    income : float
        Annual income
    years_employed : int
        Years of employment
    fico_score : int
        FICO credit score
    recovery_rate : float, default=0.1
        Expected recovery rate in case of default (10%)
    
    Returns:
    --------
    dict : Dictionary containing probability of default and expected loss
    """
    
    try:
        # Load pre-trained model
        model = joblib.load('loan_default_model.pkl')
    except:
        # If model doesn't exist, train it
        print("Model not found. Training new model...")
        model, _ = train_and_save_model()
    
    # Calculate engineered features
    debt_to_income = total_debt_outstanding / income
    loan_to_income = loan_amt_outstanding / income
    
    # Prepare input data with feature names
    feature_names = ['credit_lines_outstanding', 'loan_amt_outstanding', 'total_debt_outstanding', 
                    'income', 'years_employed', 'fico_score', 'debt_to_income', 'loan_to_income']
    
    loan_data = pd.DataFrame([[credit_lines_outstanding, loan_amt_outstanding, total_debt_outstanding,
                              income, years_employed, fico_score, debt_to_income, loan_to_income]], 
                             columns=feature_names)
    
    # Predict probability of default
    prob_default = model.predict_proba(loan_data)[0, 1]
    
    # Calculate expected loss
    loss_given_default = loan_amt_outstanding * (1 - recovery_rate)
    expected_loss = prob_default * loss_given_default
    
    return {
        'probability_of_default': prob_default,
        'expected_loss': expected_loss,
        'loan_amount': loan_amt_outstanding,
        'recovery_rate': recovery_rate,
        'risk_level': 'High' if prob_default > 0.3 else 'Medium' if prob_default > 0.1 else 'Low'
    }

# Example usage and testing
if __name__ == "__main__":
    # Example loan scenarios
    test_cases = [
        {
            'name': 'Low Risk Borrower',
            'credit_lines_outstanding': 1,
            'loan_amt_outstanding': 3000,
            'total_debt_outstanding': 5000,
            'income': 80000,
            'years_employed': 5,
            'fico_score': 750
        },
        {
            'name': 'Medium Risk Borrower',
            'credit_lines_outstanding': 3,
            'loan_amt_outstanding': 5000,
            'total_debt_outstanding': 12000,
            'income': 50000,
            'years_employed': 2,
            'fico_score': 650
        },
        {
            'name': 'High Risk Borrower',
            'credit_lines_outstanding': 5,
            'loan_amt_outstanding': 8000,
            'total_debt_outstanding': 25000,
            'income': 40000,
            'years_employed': 1,
            'fico_score': 550
        }
    ]
    
    print("LOAN RISK ASSESSMENT EXAMPLES")
    print("=" * 50)
    
    for case in test_cases:
        print(f"\n{case['name']}:")
        result = calculate_expected_loss(
            case['credit_lines_outstanding'],
            case['loan_amt_outstanding'],
            case['total_debt_outstanding'],
            case['income'],
            case['years_employed'],
            case['fico_score']
        )
        
        print(f"  Loan Amount: ${case['loan_amt_outstanding']:,.2f}")
        print(f"  Probability of Default: {result['probability_of_default']:.2%}")
        print(f"  Expected Loss: ${result['expected_loss']:,.2f}")
        print(f"  Risk Level: {result['risk_level']}")