from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# As listas de features ficam em features.py (fonte única).
from module_olist.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def create_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers = [
            ("numeric", "passthrough", NUMERIC_FEATURES),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    
def create_gradient_boosting_pipeline() -> Pipeline:
    """
    Cria o pipeline do pré-processamento e o treinamento do modelo.
    """
    preprocessador = create_preprocessor()
    
    model = GradientBoostingClassifier(
        n_estimators = 100, # Número de árvores na floresta
        learning_rate = 0.1, # Taxa de aprendizado
        max_depth = 3, # Profundidade máxima das árvores
        random_state = 42, # Semente para reproduzibilidade
    )
    
    return Pipeline(
        steps = [
            ("preprocessor", preprocessador),
            ("model", model),
        ]
    )
    
def create_xgboost_pipeline() -> Pipeline:
    preprocessador = create_preprocessor()
        
    model = XGBClassifier(
        n_estimators = 100, # Número de árvores na floresta
        learning_rate = 0.1, # Taxa de aprendizado
        max_depth = 3, # Profundidade máxima das árvores
        random_state = 42, # Semente para reproduzibilidade
    )
        
    return Pipeline(
        steps = [
            ("preprocessor", preprocessador),
            ("model", model),
        ]
    )

def create_lightgbm_pipeline() -> Pipeline:
    preprocessador = create_preprocessor()
            
    model = LGBMClassifier(
        n_estimators = 100, # Número de árvores na floresta
        learning_rate = 0.1, # Taxa de aprendizado
        max_depth = 3, # Profundidade máxima das árvores
        random_state = 42, # Semente para reproduzibilidade
    )
            
    return Pipeline(
        steps = [
            ("preprocessor", preprocessador),
            ("model", model),
        ]
    )