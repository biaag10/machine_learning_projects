import pandas as pd

def create_features(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    
    # Calcula quantos dias a empresa prometeu para realizar a entrega,
    # considerando como início o momento da aprovação do pagamento.
    data["promised_days"] = (
        data["order_estimated_delivery_date"]  # Data prometida para a entrega.
        - data["order_approved_at"]            # Data de aprovação do pagamento.
    ).dt.total_seconds().div(86_400)          # Converte segundos para dias.


    # Extrai o número do mês em que a compra foi realizada.
    # Exemplo: janeiro = 1, fevereiro = 2, ..., dezembro = 12.
    data["purchase_month"] = (
        data["order_purchase_timestamp"].dt.month
    )


    # Extrai o dia da semana em que a compra foi realizada.
    #
    # O Pandas representa os dias da seguinte forma:
    # 0 = segunda-feira
    # 1 = terça-feira
    # 2 = quarta-feira
    # 3 = quinta-feira
    # 4 = sexta-feira
    # 5 = sábado
    # 6 = domingo
    data["purchase_weekday"] = (
        data["order_purchase_timestamp"].dt.dayofweek
    )


    # Extrai a hora em que a compra foi realizada.
    # Os valores variam de 0 a 23.
    #
    # Exemplo:
    # 0  = meia-noite
    # 8  = 8 horas
    # 14 = 14 horas
    # 23 = 23 horas
    data["purchase_hour"] = (
        data["order_purchase_timestamp"].dt.hour
    )

    # Conta quantos pedidos apresentam prazo prometido menor ou igual a zero.
    #
    # Esses casos seriam suspeitos porque significariam que a data prometida
    # ocorreu antes ou exatamente no momento da aprovação do pagamento.
    print(
        "Prazos não positivos:",
        data["promised_days"].le(0).sum()
    )
    
    return data