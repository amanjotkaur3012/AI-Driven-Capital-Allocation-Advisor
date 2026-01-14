import numpy as np

def safe_divide(a, b):
    """Avoid division by zero"""
    return a / b if b != 0 else 0

def format_currency(value):
    """Format values in ₹ Crore"""
    return f"₹ {value:,.2f} Cr"
