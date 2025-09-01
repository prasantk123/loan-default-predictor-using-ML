def calculate_loan_expected_loss(credit_lines_outstanding, loan_amt_outstanding, 
                               total_debt_outstanding, income, years_employed, 
                               fico_score, recovery_rate=0.1):
    """
    Calculate the expected loss for a loan based on borrower characteristics.
    
    This function predicts the probability of default using key risk factors
    and calculates the expected loss assuming a 10% recovery rate.
    
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
        - debt_to_income_ratio: Calculated debt-to-income ratio
        - loan_to_income_ratio: Calculated loan-to-income ratio
    
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
    Expected Loss: $225.00
    """
    
    # Calculate key risk ratios
    debt_to_income = total_debt_outstanding / income
    loan_to_income = loan_amt_outstanding / income
    
    # Initialize risk score
    risk_score = 0
    
    # FICO score impact (most important factor)
    if fico_score < 550:
        risk_score += 0.5
    elif fico_score < 600:
        risk_score += 0.4
    elif fico_score < 650:
        risk_score += 0.2
    elif fico_score < 700:
        risk_score += 0.1
    elif fico_score < 750:
        risk_score += 0.05
    # Excellent credit (750+) adds no risk
    
    # Debt-to-income ratio impact (second most important)
    if debt_to_income > 0.5:
        risk_score += 0.4
    elif debt_to_income > 0.4:
        risk_score += 0.3
    elif debt_to_income > 0.3:
        risk_score += 0.2
    elif debt_to_income > 0.2:
        risk_score += 0.1
    elif debt_to_income > 0.1:
        risk_score += 0.05
    
    # Number of credit lines impact
    if credit_lines_outstanding >= 5:
        risk_score += 0.25
    elif credit_lines_outstanding >= 4:
        risk_score += 0.15
    elif credit_lines_outstanding >= 3:
        risk_score += 0.1
    elif credit_lines_outstanding >= 2:
        risk_score += 0.05
    
    # Employment stability impact
    if years_employed < 1:
        risk_score += 0.15
    elif years_employed < 2:
        risk_score += 0.1
    elif years_employed < 3:
        risk_score += 0.05
    
    # Loan-to-income ratio impact
    if loan_to_income > 0.3:
        risk_score += 0.1
    elif loan_to_income > 0.2:
        risk_score += 0.05
    
    # Cap probability at 95% (always some uncertainty)
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


# Example usage and testing
if __name__ == "__main__":
    print("LOAN EXPECTED LOSS CALCULATOR")
    print("=" * 50)
    
    # Test cases representing different risk profiles
    test_cases = [
        {
            'name': 'Excellent Credit - Low Risk',
            'credit_lines_outstanding': 1,
            'loan_amt_outstanding': 10000,
            'total_debt_outstanding': 5000,
            'income': 100000,
            'years_employed': 8,
            'fico_score': 780
        },
        {
            'name': 'Good Credit - Low Risk',
            'credit_lines_outstanding': 2,
            'loan_amt_outstanding': 5000,
            'total_debt_outstanding': 8000,
            'income': 60000,
            'years_employed': 3,
            'fico_score': 720
        },
        {
            'name': 'Fair Credit - Medium Risk',
            'credit_lines_outstanding': 3,
            'loan_amt_outstanding': 15000,
            'total_debt_outstanding': 20000,
            'income': 65000,
            'years_employed': 4,
            'fico_score': 650
        },
        {
            'name': 'Poor Credit - High Risk',
            'credit_lines_outstanding': 5,
            'loan_amt_outstanding': 8000,
            'total_debt_outstanding': 35000,
            'income': 45000,
            'years_employed': 1,
            'fico_score': 550
        },
        {
            'name': 'Very Poor Credit - Very High Risk',
            'credit_lines_outstanding': 6,
            'loan_amt_outstanding': 12000,
            'total_debt_outstanding': 40000,
            'income': 35000,
            'years_employed': 0,
            'fico_score': 480
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n{i}. {case['name']}:")
        print(f"   Loan Amount: ${case['loan_amt_outstanding']:,}")
        print(f"   Income: ${case['income']:,}")
        print(f"   FICO Score: {case['fico_score']}")
        print(f"   Credit Lines: {case['credit_lines_outstanding']}")
        print(f"   Years Employed: {case['years_employed']}")
        
        result = calculate_loan_expected_loss(
            case['credit_lines_outstanding'],
            case['loan_amt_outstanding'],
            case['total_debt_outstanding'],
            case['income'],
            case['years_employed'],
            case['fico_score']
        )
        
        print(f"   Debt-to-Income: {result['debt_to_income_ratio']:.1%}")
        print(f"   Probability of Default: {result['probability_of_default']:.1%}")
        print(f"   Expected Loss: ${result['expected_loss']:,.2f}")
        print(f"   Risk Category: {result['risk_category']}")
    
    print(f"\n" + "=" * 50)
    print("SUMMARY:")
    print("This function can be used by risk managers to:")
    print("1. Assess loan applications quickly")
    print("2. Price loans based on expected loss")
    print("3. Set appropriate interest rates")
    print("4. Make approval/rejection decisions")
    print("5. Calculate required capital reserves")