import pandas as pd
from sklearn.model_selection import train_test_split

FEATURES = [
    "purchase_hour",
    "purchase_weekday",
    "purchase_month",
    "item_count",
    "seller_coutn",
    "total_price",
    "total_freight",
    "customer_state",
]

TARGET = "is_late"

def split_data(data: pd.DataFrame):
    """Divide o dataset em conjunto de treino e teste
    """
    X = data[FEATURES]
    Y = data[TARGET]
    
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42, stratify=y)
    
    return X_train, X_test, Y_train, Y_test
    
