  # SEIVA
  ### Sistema de Estimativa Integrada de Vigilância Ambiental

<p align="center">
  <img src="seiva_logo_03_territorio_card.png" alt="Texto alternativo" width="300">
</p>

  **Previsão de Risco de Queimadas**

  <p><em>Da detecção à decisão: transformando dados abertos de fogo em prioridade de ação para quem protege o território.</em></p>

</div>


---

## Sobre o projeto

O **SEIVA** pega os dados públicos de focos de queimada do INPE (BDQueimadas), aplica um modelo de ciência de dados que aprende os padrões históricos de fogo por município, e entrega ao gestor público uma **previsão semanal de risco**, traduzida em um mapa, um ranking de prioridades e um alerta simples.

O valor não está em *detectar* fogo, mas em **antecipar e priorizar** a ação de quem precisa decidir com recursos limitados.

---

## O problema

O Brasil tem dados de queimada de classe mundial, públicos e gratuitos via INPE. Mas eles chegam em formato bruto (tabelas com coordenadas, datas e satélites) que exigem conhecimento técnico para interpretar. Secretarias municipais de meio ambiente, justamente nas regiões mais afetadas, frequentemente não têm equipe técnica para transformar esse dado em decisão.

Esse é o **"último quilômetro" do dado**: a informação existe, mas não chega de forma acionável a quem precisa agir.

**O contexto que torna isso urgente:**

- A participação do fogo na destruição da Amazônia saltou de cerca de 1% (2022) para 51% (2024), segundo o DETER/INPE, enquanto o corte raso recua, o fogo avança.
- Em 2024, a área queimada no Brasil cresceu cerca de 79% em relação a 2023, potencializada pela seca extrema do El Niño (MapBiomas Fogo, 2025).
- Estudo publicado na *Science* (Lapola et al., 2023) estima que cerca de 38% da floresta amazônica remanescente já sofre algum tipo de degradação, somando fogo, efeito de borda, extração de madeira e seca extrema.

---

## Público-alvo

- **Secretarias Municipais de Meio Ambiente (SEMMA):** gestores responsáveis por prevenção, com equipe e orçamento limitados, que hoje agem de forma reativa.
- **Coordenação estadual:** quem precisa distribuir brigadistas e recursos escassos entre muitos municípios e se beneficia do ranking de priorização.
- **Defesa Civil, Corpo de Bombeiros, ICMBio e ONGs ambientais locais:** atores que atuam de forma integrada na prevenção e no combate.

---

## A solução e onde se encaixa

Um modelo de **classificação** que aprende, a partir do histórico, a estimar a probabilidade de cada município ter queimada relevante na semana seguinte, usando variáveis simples, robustas e explicáveis.

**A pergunta que o modelo responde:**

> "Qual a probabilidade de o município X registrar queimada relevante na próxima semana?"

A saída é uma probabilidade de 0 a 100%, que vira a cor do mapa e a posição no ranking.

O INPE já oferece o **Risco de Fogo (RF)**, um indicador meteorológico robusto que estima a probabilidade de incêndio com base em secura, vento, umidade, vegetação e topografia. Reconhecer que ele existe é parte da maturidade do projeto. O ponto é que o RF entrega o dado técnico e para por aí — não traduz o risco em priorização operacional, não dimensiona esforço, e fica num portal que exige capacidade técnica para ser usado.

| O que os sistemas oficiais já fazem bem | O que o SEIVA adiciona |
|---|---|
| Detectam focos de fogo por satélite, diariamente | Aprende o padrão histórico por município e prevê risco semanal |
| Calculam risco meteorológico (Risco de Fogo / INPE) | Traduz o risco em **ranking de prioridade** entre municípios |
| Disponibilizam dados abertos em portais técnicos | Entrega **alerta simples e acionável** a quem não é técnico |
| Oferecem visão geral por estado/bioma | Apoia a **decisão operacional local**: onde e quando agir |

> *"O Risco de Fogo do INPE oferece a base. O SEIVA oferece a camada de decisão. Não substituímos o que já existe, começamos exatamente onde ele termina, transformando informação técnica em ação para quem precisa proteger o território e não tem equipe especializada."*

---

## Features do modelo

| Variável | Por que entra |
|---|---|
| **Focos nas semanas anteriores** | Tendência recente — fogo tende a puxar mais fogo na mesma região |
| **Mês / estação (sazonalidade)** | O fogo é fortemente sazonal; a estação seca concentra a maioria dos focos |
| **Média histórica do município** | Calibra o que é "normal" para aquele município naquela época do ano |
| **Área do município** | Corrige o viés de municípios grandes registrarem naturalmente mais focos |
| **Condição climática (El Niño / La Niña)** | Ajusta o risco em anos atípicos de seca, evitando que o modelo se "engane" em anos extremos |

> 🚧 **PENDENTE:** documentar o modelo final escolhido, hiperparâmetros e métricas de avaliação (ex.: ROC-AUC, Log Loss) quando o pipeline estiver consolidado.

---

## Estrutura do repositório

> 🚧 **PENDENTE:** a estrutura abaixo reflete o estado atual e deve ser atualizada conforme o código for organizado em módulos.

```
previsao_incendios/
├── assets/                     # Logo e recursos visuais
├── dados/                      # CSVs do BDQueimadas/INPE e dados de apoio (IBGE) — não versionados
├── leitor_csv.py               # Função de carregamento dos CSVs em chunks
├── tratamentos_files.ipynb     # Limpeza, correção de encoding e features temporais
├── auxiliar.ipynb              # Pipeline: agregação, features, balanceamento e treino do modelo
├── .gitignore
└── README.md
```

 <!-- vale a pena colocar essa parte ou os repositórios falam por si só?
 ---

 ## Fontes de dados

- **BDQueimadas / INPE** — focos de calor: https://terrabrasilis.dpi.inpe.br/queimadas/bdqueimadas/
- **Dados abertos / Programa Queimadas**: https://terrabrasilis.dpi.inpe.br/queimadas/portal/dados-abertos/
- **IBGE** — estrutura territorial (área dos municípios): https://www.ibge.gov.br/geociencias/organizacao-do-territorio/estrutura-territorial/
- **MapBiomas** (referência de contexto): https://plataforma.brasil.mapbiomas.org/

> Os arquivos de dados **não** são versionados no repositório devido ao volume. Baixe-os nas fontes acima e coloque-os na pasta `dados/`.
-->
---

## Como executar

> **PENDENTE:** instruções refletem o estado atual (notebooks). Serão revisadas quando o pipeline for modularizado e o dashboard/alerta forem incorporados.

1. **Clone o repositório**
   ```bash
   git clone https://github.com/vitorvitortoledo-art/previsao_incendios.git
   cd previsao_incendios
   ```

2. **Instale as dependências**
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost jupyter
   ```
   <!-- 🚧 PENDENTE: substituir por `pip install -r requirements.txt` quando o arquivo for criado -->

3. **Adicione os dados**
   Coloque os arquivos de focos do INPE (e dados de apoio) dentro da pasta `dados/`.

4. **Rode os notebooks** na ordem:
   - `tratamentos_files.ipynb` — exploração e limpeza inicial
   - `auxiliar.ipynb` — pipeline completo até o treino do modelo

---

<!-- aqui eu fiquei na duvida se colocamos assim ou eu faço um sobre para informar que é um projeto de faculdade e colocar os nomes dos responsáveis

## Equipe

| Nome | RM |
|---|---|
| Gabriel Monari | 571879 |
| Vitor Nogueira | 570155 |
| Letícia Santos | 572591 |
| Paulo Moraes | 572297 |

-->

## Links

- **Dashboard:** 🚧 *pendente*

---

## Licença

> 🚧 **PENDENTE:**
