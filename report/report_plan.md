# Plano do Relatório — TP2: Redes Neurais Artificiais (DCC642 / UFMG)

## Restrições Formais (Seções 4, 5 e 6 de `tp2-rna.pdf`)
- **Formato:** Template oficial **AAAI-27** (`aaai2027.sty`), duas colunas, papel US Letter (`letterpaper`), compilado com **PDFLaTeX** (`pdflatex`). **Não é permitido alterar o formato** (sem modificar `aaai2027.sty`, margens, fontes, espaçamentos ou usar pacotes proibidos como `geometry`, `hyperref`, `titlesec`, `setspace`, `savetrees`, `float`, `multicol`, `wrapfig`).
- **Limite de páginas:** Estritamente **máximo de 3 páginas** (incluindo figuras, tabelas e a Declaração de Uso de IA).
- **Idioma:** Português (`\usepackage[brazilian]{babel}`, codificação UTF-8 + `T1` fornecida pelo `aaai2027.sty`).
- **Entrega (Moodle):** Dois arquivos enviados separadamente, sem compactar: `learn.py` e `relatorio.pdf`.
- **Critério mínimo de desempenho:** A melhor configuração reportada deve atingir **acurácia de teste $\ge 90\%$**.

---

## Orçamento de Páginas (Total alvo: 2,8 – 3,0 páginas)

| Página | Coluna | Conteúdo Planejado | Fração Estimada |
| :--- | :--- | :--- | :--- |
| **Página 1** | Topo (largura total) | Título, autor, afiliação (DCC/UFMG, DCC642 2026/2) + Resumo (`abstract`, ~5 linhas) | ~0,25 pág. |
| **Página 1** | Coluna esquerda | **1. Introdução e Objetivo** (~0,35 col.) + início de **2. Metodologia** (~0,40 col.) | ~0,75 col. |
| **Página 1** | Coluna direita | Continuação de **2. Metodologia** (arquitetura, equações concisas, treinamento, protocolo experimental) | ~0,75 col. |
| **Página 2** | Coluna esquerda | **3. Experimentos e Resultados** — texto introdutório + **Tabela 1** (varredura controlada de hiperparâmetros) | ~1,00 col. |
| **Página 2** | Coluna direita | Melhor configuração ($\ge 90\%$) + **Figura 1** (curvas de perda e acurácia) + início de **3.1 Análise Qualitativa** | ~1,00 col. |
| **Página 3** | Coluna esquerda | Conclusão de **3.1 Análise Qualitativa** (~0,25 col.) + **4. Discussão** (trade-offs e limitações, ~0,75 col.) | ~1,00 col. |
| **Página 3** | Coluna direita | **5. Conclusão** (~0,50 col.) + **Declaração de Uso de IA** (~0,35 col.) — margem de segurança de ~0,15 col. | ~0,85 col. |

> **Regra de controle de espaço:** Manter o texto direto e analítico, usar `booktabs` com `\small` na Tabela 1 e combinar as curvas de perda e acurácia em uma figura composta compacta (lado a lado em largura de 1 coluna ou empilhadas com boa legibilidade) para garantir que o PDF final nunca ultrapasse 3 páginas.

---

## Detalhamento por Seção

### Cabeçalho e Resumo (`abstract`)
- **Título:** Classificação de Dígitos Manuscritos com Rede Neural Artificial de Duas Camadas no Conjunto MNIST.
- **Resumo (80–110 palavras):**
  - Problema e conjunto de dados (MNIST, 70.000 imagens $28\times28$).
  - Implementação da MLP de duas camadas ($784 \to h \to 10$) com ReLU, Softmax, entropia cruzada e gradiente descendente em lote completo.
  - Varredura controlada de taxa de aprendizado, número de épocas e neurônios na camada escondida.
  - Melhor configuração obtida ($\ge 90\%$ de acurácia de teste) e principal conclusão sobre o comportamento da rede.

---

### 1. Introdução e Objetivo (~180–220 palavras)
- Contextualizar a tarefa de reconhecimento de dígitos manuscritos (0 a 9) utilizando o conjunto de dados **MNIST** (70.000 imagens em escala de cinza de resolução $28 \times 28$ pixels, vetorizadas em 784 atributos).
- Apresentar a proposta do trabalho: modelar e implementar do zero (em NumPy, no arquivo `src/learn.py`) uma Rede Neural Artificial (MLP) de duas camadas (uma camada escondida e uma camada de saída).
- Explicitar os objetivos centrais:
  1. Implementar inicialização de parâmetros, funções de ativação (ReLU e Softmax), propagação direta (*forward*), função de perda (entropia cruzada), atualização via gradiente descendente e cálculo de acurácia.
  2. Avaliar sistematicamente o efeito de hiperparâmetros-chave (taxa de aprendizado, épocas e largura da camada escondida) na perda de treinamento, acurácia de teste e tempo computacional.
  3. Selecionar uma configuração que atinja ao menos **90% de acurácia no conjunto de teste** e validar seu comportamento qualitativamente na interface de desenho (`server.py`).

---

### 2. Metodologia (~420–480 palavras)
Organizada em subtópicos compactos (ou parágrafos com `\textbf{...}` para economizar espaço vertical):

1. **Conjunto de Dados e Pré-processamento (`load_mnist`):**
   - Carregamento de `mnist_784` via `fetch_openml`.
   - Normalização das intensidades de pixel de $[0, 255]$ para $[0, 1]$ (`X / 255.0`).
   - Divisão estratificada/aleatória fixa **80/20** (`test_size=0.2`, `random_state=42`): 56.000 amostras de treinamento e 14.000 de teste.
   - Codificação *one-hot* dos rótulos para $C = 10$ classes.

2. **Arquitetura e Inicialização (`initialize_parameters`):**
   - MLP totalmente conectada: entrada $d_{\text{in}} = 784$, camada escondida com $h$ neurônios (`hidden_size`), saída com $C = 10$ classes.
   - Pesos $W_1 \in \mathbb{R}^{784 \times h}$ e $W_2 \in \mathbb{R}^{h \times 10}$ inicializados com distribuição normal $\mathcal{N}(0, 0{,}01^2)$ (`np.random.randn * 0.01`, semente global `np.random.seed(42)`).
   - Viéses $b_1 \in \mathbb{R}^{1 \times h}$ e $b_2 \in \mathbb{R}^{1 \times 10}$ inicializados com zeros.

3. **Propagação Direta e Funções de Ativação (`forward`, `relu`, `softmax`):**
   - Equações:
     $$Z_1 = X W_1 + b_1, \quad A_1 = \text{ReLU}(Z_1) = \max(0, Z_1)$$
     $$Z_2 = A_1 W_2 + b_2, \quad A_2 = \text{softmax}(Z_2)$$
   - Decisão de projeto para estabilidade numérica na Softmax: subtração do valor máximo por amostra ($z_j - \max_k z_k$) antes da exponenciação para evitar *overflow*.

4. **Função de Perda, Otimização e Avaliação (`cross_entropy_loss`, `backward`, `gradient_descent_step`, `accuracy`):**
   - Entropia cruzada multiclasse:
     $$L = -\frac{1}{m}\sum_{i=1}^{m}\sum_{c=1}^{C} y_{ic}\log(\hat{y}_{ic})$$
   - Decisão de projeto: *clipping* de $\hat{y}_{ic}$ em $[10^{-12}, 1{,}0]$ antes do logaritmo para evitar $\log(0)$.
   - Treinamento via **gradiente descendente em lote completo (*full-batch GD*)**, usando os gradientes exatos de `backward()` sobre todo o conjunto de treino a cada época: $w \leftarrow w - \alpha \frac{\partial L}{\partial w}$.
   - Métrica de avaliação: acurácia no conjunto de teste via comparação entre `argmax` das previsões e dos rótulos reais.

5. **Protocolo Experimental:**
   - Experimentos controlados variando **apenas um hiperparâmetro por vez**, mantendo os demais fixos em uma configuração-base explícita (a ser preenchida com os valores usados pelo Agent 2, ex.: $\alpha_{\text{base}}$, $\text{épocas}_{\text{base}}$, $h_{\text{base}}$).
   - Medição de perda final de treinamento, acurácia final de teste e tempo total de execução em CPU.

---

### 3. Experimentos e Resultados (~380–440 palavras + 1 Tabela + 1 Figura)

1. **Varredura Controlada de Hiperparâmetros — Tabela 1 (`tab:experimentos`):**
   - Tabela única compacta em 1 coluna (`booktabs`), agrupada em 3 blocos:
     - **Taxa de aprendizado ($\alpha$):** `0,001`, `0,01`, `0,1`, `0,5`
     - **Número de épocas:** `50`, `100`, `200`, `500`
     - **Neurônios na camada escondida ($h$):** `4`, `16`, `64`, `256`
   - **Colunas:** Grupo / Hiperparâmetro, Valor, Perda final (treino), Acurácia final (teste, %), Tempo (s).
   - Destaque em negrito para o melhor valor de cada bloco e declaração explícita dos valores fixos da base na legenda e no texto.
   - Síntese objetiva das tendências observadas em cada bloco (sem repetir exaustivamente todos os números da tabela).

2. **Melhor Configuração ($\ge 90\%$ de Acurácia de Teste) e Curvas de Aprendizado — Figura 1 (`fig:curvas`):**
   - Informar os hiperparâmetros escolhidos para maximizar a acurácia de teste, a perda final, a acurácia de teste final ($\ge 90\%$) e o tempo de treino.
   - **Figura 1:** Dois painéis (esquerda: curva de perda de treinamento; direita: curva de acurácia de teste ao longo das épocas), gerados com estilo limpo e legível em PDF/PNG de alta resolução (compatível com `plot.py` / `model/losses.csv` e `model/accuracies.csv`, sem cortar rótulos).
   - Descrever o formato das curvas (queda acentuada inicial, ponto de inflexão, regime de platô/convergência).

3. **3.1 Análise Qualitativa na Interface Interativa (`server.py`):**
   - Testes com **pelo menos 3 dígitos desenhados à mão** na interface web (`http://localhost:5001`) carregando os pesos da melhor configuração.
   - Relatar para cada dígito testado:
     - Dígito desenhado, classe prevista e confiança (probabilidade Softmax).
     - Quais dígitos a rede acerta com facilidade (ex.: `0`, `1`, `7` com traços distintos) e quais tende a confundir (ex.: `4` vs. `9`, `3` vs. `8`, `5` vs. `3` ou `2` vs. `7`).
     - Possíveis motivos: sensibilidade à posição/centralização do traço no canvas $28\times28$, espessura do pincel diferente do padrão MNIST, ausência de pré-processamento de centralização por centro de massa e falta de invariância espacial em MLPs.

---

### 4. Discussão (~350–400 palavras)
Análise crítica estruturada nos pontos exigidos pela Seção 4 do enunciado:
- **Convergência vs. estabilidade (taxa de aprendizado):**
  - Taxas muito baixas (`0,001`, `0,01`) produzem atualizações pequenas e convergência lenta em *full-batch GD*.
  - Taxas intermediárias/altas (`0,1`, `0,5`) aceleram substancialmente a queda da perda, mas valores excessivos podem induzir oscilações em torno de mínimos ou instabilidade numérica.
- **Retornos decrescentes vs. orçamento de épocas:**
  - Ganho expressivo nas primeiras centenas de épocas seguido de achatamento da curva (ganho marginal menor por unidade de tempo computacional).
- **Capacidade da rede vs. custo computacional e generalização:**
  - Com $h = 4$, há claro *underfitting* (gargalo informacional para separar 10 classes).
  - Aumentar $h$ para `16`, `64` e `256` eleva a expressividade, mas aumenta linearmente o número de parâmetros ($784h + h + 10h + 10$) e o custo das multiplicações matriciais por época, com retornos decrescentes e maior potencial de *overfitting* em treinos longos.
- **Limitações arquiteturais e algorítmicas:**
  - *Full-batch GD* realiza apenas 1 atualização de pesos por época (56.000 exemplos), exigindo muitas épocas e maior custo por atualização.
  - Inicialização gaussiana fixa ($\sigma = 0{,}01$) não escala com a largura da camada (ao contrário da inicialização de He para ReLU).
  - Vetorização dos pixels ($28 \times 28 \to 784$) destrói a topologia espacial 2D, tornando a rede sensível a translações, rotações e variações de escala (evidenciado na análise qualitativa).

---

### 5. Conclusão (~140–180 palavras)
- Recapitular a implementação bem-sucedida da MLP de duas camadas e a melhor configuração obtida (hiperparâmetros e acurácia de teste $\ge 90\%$).
- Resumir as principais lições sobre sensibilidade a hiperparâmetros.
- Listar melhorias concretas e justificadas para trabalhos futuros:
  1. **Otimização:** *Mini-batch SGD* com *momentum* ou **Adam** para convergência muito mais rápida.
  2. **Inicialização:** **Inicialização de He** ($\mathcal{N}(0, 2/n_{\text{in}})$) adequada para ativações ReLU.
  3. **Regularização e dados:** Regularização $L_2$ (*weight decay*), *Dropout* e *data augmentation* (pequenas translações/rotações) para robustez a desenhos manuais.
  4. **Arquitetura:** **Redes Neurais Convolucionais (CNNs)** para explorar localidade espacial e invariância a translação.

---

### Declaração de Uso de IA (Seção não numerada — obrigatória pela Seção 6 de `tp2-rna.pdf`)
- Declarar explicitamente:
  - **Ferramentas utilizadas:** ex. Antigravity / assistente de IA utilizado no apoio ao desenvolvimento.
  - **Finalidade:** apoio na configuração do template LaTeX AAAI-27, criação de scripts auxiliares para execução automatizada de experimentos/coleta de métricas/geração de gráficos e apoio na revisão estrutural e redacional do relatório (deixando claro que a implementação principal de `learn.py` e a análise crítica foram conduzidas e validadas pelo aluno).
  - **Links para as conversas:** `[TODO: inserir URLs/links das conversas realizadas antes da submissão final]`.
