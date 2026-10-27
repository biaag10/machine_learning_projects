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


def map_to_original_feature(transformed_name, categorical_features):
    """
    Converte o nome gerado pelo pré-processador para a feature original.

    Exemplos:
    - "numeric__promised_days"          -> "promised_days"
    - "categorical__customer_state_SP"  -> "customer_state"
    """

    name = transformed_name.split("__", 1)[-1]

    for feature in categorical_features:
        if name.startswith(f"{feature}_"):
            return feature

    return name


def aggregate_shap_by_feature(shap_values, X_original, categorical_features):
    """
    Soma as contribuições SHAP das colunas One-Hot de volta para a feature original.

    O One-Hot divide "customer_state" em 27 colunas (uma por estado). Olhando
    coluna a coluna, o estado parece pouco importante, porque a importância fica
    espalhada. Como o SHAP é ADITIVO, podemos somar as contribuições das colunas
    de uma mesma feature e analisar o efeito do estado como um todo.
    """

    original_names = [
        map_to_original_feature(name, categorical_features)
        for name in shap_values.feature_names
    ]

    values = pd.DataFrame(shap_values.values, columns=original_names)
    aggregated = values.T.groupby(level=0, sort=False).sum().T

    features = list(aggregated.columns)

    # Valores originais (ex.: "SP" em vez de 0/1) para exibir nos gráficos.
    # Categóricas de texto viram NaN só na cor do beeswarm (ficam cinza).
    display_data = X_original[features].reset_index(drop=True)
    numeric_data = display_data.apply(pd.to_numeric, errors="coerce")

    return shap.Explanation(
        values=aggregated.to_numpy(),
        base_values=shap_values.base_values,
        data=numeric_data.to_numpy(dtype=float),
        display_data=display_data.to_numpy(dtype=object),
        feature_names=features,
    )