# Previsão de Preços de Aluguel de Apartamentos em São Paulo com Machine Learning

Projeto de regressão end-to-end seguindo o checklist de projetos de ML: análise exploratória, preparação dos dados, comparação de modelos, validação cruzada e ajuste de hiperparâmetros.

## Dados
`sao-paulo-properties-april-2019.csv` — anúncios de imóveis de São Paulo (abril/2019) com preço, condomínio, área, quartos, banheiros, suítes, vagas, elevador, mobília, piscina, bairro e coordenadas. O projeto usa apenas anúncios de **aluguel**.

## Metodologia
1. **Exploração** — mapa interativo (Plotly/Mapbox) de preços, histogramas, correlação com o preço.
2. **Preparação** — remoção de colunas irrelevantes e *one-hot encoding* do bairro (`pd.get_dummies`).
3. **Modelos** — Regressão Linear, Decision Tree e Random Forest.
4. **Avaliação** — RMSE no treino e *cross-validation* (10 folds).
5. **Otimização** — `GridSearchCV` no Random Forest e avaliação final no conjunto de teste (70/30).

## Resultados
O melhor Random Forest no GridSearch teve RMSE de validação cruzada ≈ R$ 1.800 (veja o notebook para os números do conjunto de teste).

## Como executar
```bash
pip install -r requirements.txt
jupyter notebook dados-imobiliarios.ipynb
```
O mapa usa Mapbox: crie um arquivo `mapbox_token` (ignorado pelo git) com seu token público em https://account.mapbox.com/ — ou troque por `px.scatter_map` / `open-street-map`.

## Stack
Python · pandas · scikit-learn · Plotly · Matplotlib · Seaborn

## Créditos
Projeto desenvolvido durante o curso *Fundamentos de IA e Machine Learning* da Asimov Academy.
