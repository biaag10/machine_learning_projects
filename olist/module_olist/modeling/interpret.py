import pandas as pd
import shap # # Biblioteca utilizada para explicar as previsões dos modelos (contribuição de cada feature)
from scipy import sparse # O OneHotEncoder pode gerar esse tipo de matriz para economizar memória.


def prepare_data_for_shap(pipeline, X):
    """
    Aplica o mesmo pré-processamento
    utilizado durante o treinamento.
    """

    # Recupera o pré-processador
    preprocessor = pipeline.named_steps["preprocessor"]

    # Aplica as transformações
    X_transformed = preprocessor.transform(X)

    # Se for matriz esparsa,
    # converte para matriz normal
    if sparse.issparse(X_transformed):
        X_transformed = (X_transformed.toarray())

    # Recupera os nomes das features
    feature_names = (preprocessor.get_feature_names_out())

    # Converte para DataFrame
    X_transformed = pd.DataFrame(
        X_transformed,
        columns=feature_names,
        index=X.index,
    )

    return X_transformed

def create_explainer(pipeline):
    """
    Cria o explicador SHAP
    para o modelo treinado.
    """

    model = pipeline.named_steps["model"]

    # Crie um explicador especializado em entender as previsões deste modelo baseado em árvores.
    explainer = shap.TreeExplainer(model)

    return explainer


def calculate_shap_values(explainer, X_transformed):
    """
    Calcula as contribuições SHAP
    das observações.
    """

    # Explica o que esse modelo faz para as observações fornecidas.
    shap_values = explainer(X_transformed)

    return shap_values