# Olist Orders Dataset

## Visão geral

O arquivo `olist_orders_dataset.csv` contém informações sobre os **pedidos realizados na plataforma Olist**. Cada linha representa **um único pedido** e registra seu cliente, status e principais datas do ciclo do pedido, desde a compra até a entrega.

A tabela é especialmente útil para análises de **volume de pedidos, status, tempo de aprovação, tempo de envio, prazo de entrega, atrasos e performance logística**.

## Dimensões da base

- **Quantidade de registros:** 99.441 pedidos
- **Quantidade de colunas:** 8
- **Granularidade:** 1 linha por pedido
- **Chave primária:** `order_id`
- **Período de compra:** 04/09/2016 a 17/10/2018

## Dicionário de dados

| Coluna | Tipo sugerido | Descrição |
| `order_id` | string | Identificador único do pedido. É a principal chave para relacionar esta tabela com itens, pagamentos e avaliações. |
| `customer_id` | string | Identificador do cliente associado ao pedido. Pode ser relacionado à tabela `olist_customers_dataset.csv`. |
| `order_status` | string/categoria | Situação do pedido no momento registrado na base. |
| `order_purchase_timestamp` | datetime | Data e hora em que o pedido foi realizado pelo cliente. |
| `order_approved_at` | datetime | Data e hora em que o pagamento/pedido foi aprovado. |
| `order_delivered_carrier_date` | datetime | Data e hora em que o pedido foi entregue à transportadora. |
| `order_delivered_customer_date` | datetime | Data e hora em que o pedido foi efetivamente entregue ao cliente. |
| `order_estimated_delivery_date` | datetime | Data prevista para a entrega informada ao cliente. |

## Status dos pedidos

A coluna `order_status` possui 8 valores distintos:

| Status        | Quantidade | Interpretação                                                 |
| ------------- | ---------: | ------------------------------------------------------------- |
| `delivered`   |     96.478 | Pedido entregue ao cliente.                                   |
| `shipped`     |      1.107 | Pedido já enviado, mas ainda não registrado como entregue.    |
| `canceled`    |        625 | Pedido cancelado.                                             |
| `unavailable` |        609 | Pedido/produto ficou indisponível para conclusão.             |
| `invoiced`    |        314 | Pedido faturado.                                              |
| `processing`  |        301 | Pedido em processamento/preparação.                           |
| `created`     |          5 | Pedido criado, ainda em estágio inicial.                      |
| `approved`    |          2 | Pedido aprovado, mas sem avanço posterior registrado na base. |

Aproximadamente **97% dos pedidos estão com status `delivered`**.

## Relacionamentos com outras tabelas

O arquivo funciona como uma tabela central para diversas análises da base Olist.

- `order_id` → `olist_order_items_dataset.csv`
- `order_id` → `olist_order_payments_dataset.csv`
- `order_id` → `olist_order_reviews_dataset.csv`
- `customer_id` → `olist_customers_dataset.csv`

### Atenção aos joins

As tabelas de itens, pagamentos e avaliações podem possuir **mais de uma linha para o mesmo pedido**. Portanto, juntar todas diretamente e depois somar valores pode gerar duplicação e inflação das métricas.

Para análises financeiras, uma prática segura é primeiro agregar cada tabela por `order_id` e somente depois realizar os joins.

## Qualidade dos dados

### Completude

| Coluna                          | Valores ausentes | Percentual |
| ------------------------------- | ---------------: | ---------: |
| `order_id`                      |                0 |      0,00% |
| `customer_id`                   |                0 |      0,00% |
| `order_status`                  |                0 |      0,00% |
| `order_purchase_timestamp`      |                0 |      0,00% |
| `order_approved_at`             |              160 |      0,16% |
| `order_delivered_carrier_date`  |            1.783 |      1,79% |
| `order_delivered_customer_date` |            2.965 |      2,98% |
| `order_estimated_delivery_date` |                0 |      0,00% |

Os valores nulos nas datas de envio e entrega **não devem ser automaticamente tratados como erro**, pois podem estar associados a pedidos cancelados, indisponíveis ou ainda não concluídos.

### Unicidade

- `order_id`: 99.441 valores distintos para 99.441 linhas.
- `customer_id`: 99.441 valores distintos nesta tabela.
- Não foram encontradas linhas completamente duplicadas.

### Consistência temporal

Foram identificadas algumas sequências de datas que merecem investigação:

- **1.359 pedidos** possuem `order_delivered_carrier_date` anterior a `order_approved_at`.
- **23 pedidos** possuem `order_delivered_customer_date` anterior a `order_delivered_carrier_date`.

Esses casos podem representar inconsistências de registro e devem ser sinalizados antes de análises detalhadas de lead time/logística.

## Métricas que podem ser criadas

A partir das datas desta tabela é possível calcular, por exemplo:

### Tempo de aprovação

`order_approved_at - order_purchase_timestamp`

Mede quanto tempo passou entre a realização da compra e sua aprovação.

### Tempo até o envio

`order_delivered_carrier_date - order_approved_at`

Mede quanto tempo o pedido levou para ser preparado e entregue à transportadora.

### Tempo total de entrega

`order_delivered_customer_date - order_purchase_timestamp`

Mede o lead time total entre a compra e a chegada ao cliente.

### Atraso ou antecipação

`order_delivered_customer_date - order_estimated_delivery_date`

- Resultado maior que zero → entrega atrasada.
- Resultado menor ou igual a zero → entrega dentro do prazo ou antecipada.

## Cuidados na análise

1. Converta todas as colunas de data para o tipo `datetime` antes de realizar cálculos.
2. Não remova datas nulas sem analisar o `order_status` correspondente.
3. Para medir performance logística, normalmente faz sentido trabalhar principalmente com pedidos `delivered`.
4. Sinalize ou trate sequências temporais inconsistentes antes de calcular tempos médios.
5. Os últimos meses da série possuem poucos pedidos registrados; portanto, análises temporais devem considerar a possibilidade de período incompleto no final da base.
6. Use `order_id` como chave principal para relacionar pedidos às demais tabelas transacionais.

## Resumo

`olist_orders_dataset.csv` é a principal tabela para acompanhar o **ciclo de vida dos pedidos da Olist**. Sua granularidade é de um registro por pedido e seus campos permitem analisar desde o momento da compra até a entrega ao cliente. A base possui boa completude e unicidade, mas apresenta alguns valores nulos esperados e pequenas inconsistências temporais que devem ser consideradas em análises de Data Quality e performance logística.
