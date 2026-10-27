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
            # sparse_output=False: a saída do One-Hot é uma matriz densa (zeros reais).
            # Com matriz esparsa, o XGBoost interpreta os zeros como VALOR AUSENTE,
            # e o modelo treinado passa a se comportar diferente da mesma entrada em
            # formato denso, que é o formato usado pelo SHAP. Com saída densa,
            # treino, inferência e explicação enxergam exatamente os mesmos dados.
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
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