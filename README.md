# machine learning

- machine learning: habilidade de um computador aprender algo sem ser explicitamente programado (mais oportunidades de aprender \= melhor performance)  
- ex: jogo de xadrez

##### *terminologias*

- *training set* – dados usados para treinar um modelo  
- *plotting data* (plotar dados) – representar informações de forma gráfica para facilitar a análise; converter dados brutos (tabelas, planilhas) em visuais (gráficos)  
- *x* – variável input/recurso  
- *y* – variável output/alvo (resposta)  
- *m* – número de exemplos de treino  
- *(x, y) –* um exemplo de treino específico  
- *(x^(i), y^(i)) –* exemplo de treino número X (i^th) – *i* é uma linha específica na tabela (index)  
- *f* – (função hipótese) recebe um input *x* e retorna uma estimativa *y-hat*  
- *y* (*y-hat)* – valor previsto (output) para *y* (não necessariamente o *y* correto)


##### *python -- NumPy*

- *x* → *x\_train*  
- *y* → y*\_train*  
- *x^(i), y^(i)* → *x\_i, y\_i*  
- *m → \[var\].shape* (tupla)  
- plottar dados → *malplotlib.scatter()*

##### *jupyter notebook*

- markdown cell – descreve o código  
- code cell – código (shift \+ enter para rodar)  
- *run*: roda 1 célula por vez

## ***algoritmos de aprendizado supervisionado***

- mais usado, avanços rápidos, o modo de uso das ferramentas gera resultados melhores  
- mapeia x (input) para y (output)  
- disponibiliza para o algoritmo aprender exemplos de respostas corretas *y* a partir de um dado input *x* (pergunta \+ resposta) – o algoritmo vai identificando padrões em milhares de exemplos  
- o algoritmo eventualmente aprende a receber o input sozinho e retorna uma previsão razoavelmente precisa de saída

**exemplos**

| input (x) | → | output (y) | aplicação |
| :---- | :---- | :---- | :---- |
| email | → | spam? (0/1) | filtragem de spam |
| áudio | → | transcrição em texto | reconhecimento de fala |
| inglês | → | espanhol | tradução |
| anúncio, dados do usuário | → | clica? (0/1) | anúncios online $$$ |
| foto de um produto | → | defeito? (0/1) | inspeção visual |
| imagem, foto de radar | → | posição de outros carros | carro autônomo |

### *regression*

- como define se usa uma função de linha reta, curva ou outra para os dados? algoritmo escolhe a linha mais apropriada  
- prever um **número** (preço, temperatura, idade) entre **infinitas** possibilidades de outputs

**exemplos**

1. precificação de uma casa por tamanho 

2. histórico de clima → prever temperatura de amanhã

#### ***linear regression***

- linha reta nos dados  

##### ***univariate (one variable)*** 

-  size → *f* → estimated price  
  **fw,b (x) \= wx \+ b**	OU	**f (x) \= wx \+ b**  
  - *w* e *b* são números  
  - a função faz a linha

##### ***squared error cost function***

- função de erro mais usada em regressão linear  
- mostra quão bem o modelo está indo – a linha passa perto dos exemplos de treinamento

###### ***recapitulando**:*

- **fw,b (x) \= wx \+ b**	  
  - **w**, **b** \= parâmetros/coeficientes/pesos (variáveis que podem ser ajustadas para aprimorar o modelo  
  - dependendo dos valores de w e b, a função fica diferente → gera uma linha diferente no gráfico  
  - **w** → inclinação/declive da reta, determina quanto *f(x)* aumenta (w+) ou diminui (w–) a cada unidade adicionada em *x*   
  - **b** → bias/viés, determina o ponto de partida (onde a reta cruza o eixo vertical *y* quando *x \= 0*); se *b* aumenta/diminui, a reta inteira sobe/desce

**exemplo**: w \= 0 e b \= 1.5  

- f(x) \= 0 \* x \+ 1.5  
- y-hat \= 1.5 ( \= b) → b \= y-intercept (onde cruza o eixo y no gráfico)

**exemplo**: w \= 0.5 e b \= 0  

- f(x) \= 0.5 \* x  
- declive \= 0.5 / 1 (valor de **w**)

**exemplo**: w \= 0.5 e b \= 1  

- f(x) \= 0.5 \* x \+ 1  
- declive \= 0.5 / 1 (valor de **w**)

###### ***voltando a COST FUNCTION***

- a linha (y-hat) tenta prever resultados, mas nem sempre bate com os training examples  
- mede a diferença entre as previsões do modelo e os valores reais de y
 
**J(w,b) \= 12mi \= 1m y(i)\-y(i)2 →** ou

**Jw, b= 12mi \= 1m fw,b(x(i))\-y(i) 2**

- y  \= fw, b(x(i))  
- soma os resultados desde *i \= 1* até *i \= m* e depois faz a média (para que o número não necessariamente suba a cada training set a mais), e /2 para ficar mais limpo  
- *m* \= quantidade de training examples  
- J deve ser o menor possível  
- **y** é próximo de y(i) em todos (x(i), y(i))  
- objetivo:  

###### ***como usar a cost function para encontrar os melhores parâmetros para o modelo***

- para que a linha bata com os exemplos de treino  
- versão simplificada (b \= 0\) → a reta é obrigada a passar pela origem

**fw(x)=wx**   
**J(w)= 12mi \= 1m fw(x(i))\-y(i) 2**

**exemplo:**

- w fixo, função de x  
- m \= 3  
- erro: diferença de *y* entre (*y-hat* – *y real*)  
- traça a função J com diferentes valores de *w*  
  - cada valor de *w* corresponde a um valor na linha de *J*   
  - quando *w \= 1*, *J(w) \= 0*   
1. w \= 1  
   **J(w)= 12mi \= 1m wx(i)\-y(i) 2 \= 12m( 02+ 02+ 02\) \= 0**  
     
2. w \= 0.5   
- erro *x \= 1* \=  **(0.5-1)2**  
- erro *x \= 2* \=  **(1-2)2**  
- erro *x \= 3* \=  **(1.5-3)2**  
- erro total → J (0) *\=* **\[3.5\]\*** **12 \* 3 \= 3.56 \~ \= 0.58**   
    
3. w \= 0  
- erro *x \= 1* \=  **(1-0)2**  
- erro *x \= 2* \=  **(2-0)2**  
- erro *x \= 3* \=  **(3-0)2**  
- erro total *\=* **\[14\]\*** **12 \* 3 \= 146 \~ \= 2.3**   
    
4. w \= –0.5  
- erros \=  **(-0.5-1)2 \+(-1-2)2+ (-1.5-3)2**  
- erros \= **\[31.5\]\*** **12 \* 3 \~ \= 5.25**


J  
- usa a cost function para encontrar o valor de *w* (ou de *w* e *b*) que minimiza *J* 


### ***classification***

- prever uma **categoria**, a saída *y* é uma classe/rótulo (sim/não, 0/1, “cachorro”, “gato”) entre **poucas** possibilidades limitadas de outputs

**exemplos**

1. se o email é spam (0/1)  
2. se o produto tem defeito (0/1)  
3. se o usuário clica no anúncio (0/1)  
4. detecção de câncer de mama – verifica se o caroço é maligno (1) ou benigno (0)    
a. 1 input: tamanho  
b. 2 inputs: tamanho e idade → algoritmo encontra um *limite* que separa benigno-maligno    
c. mais inputs: grossura, uniformidade das células, formato

## ***algoritmos de aprendizado não supervisionado***

- super supervised learning – fornece dados que não são associados a nenhum output y (sem rótulos)  
  - ex: dados de tamanho de tumor \+ idade, sem definir benigno/maligno

### ***clustering (agrupamento)***

- não está supervisionando o algoritmo para ver se ele dá uma resposta correta para cada input → algoritmo deve buscar padrões nos dados e dividir os dados em conjuntos sozinho

**exemplos**

1. google news – pega milhões de notícias e agrupa os semelhantes em conjuntos → encontra as palavras-chave por conta própria  
   1. junta notícias com as palavras panda \+ twin \+ zoo  
2. microarray DNA – cada coluna representa a atividade de DNA de cada pessoa, cada linha representa um gene específico (cor de olho, altura)  
   1. medir quanto um gene específico é expressado em diferentes pessoas (ativo ou não), dividir pessoas semelhantes no mesmo grupo  
   2. linkar que não gostar de vegetais é genético  
3. dividir tipos de clientes entre os segmentos do mercado – com base nos principais interesses/motivações, recomendar cursos semelhantes  
   

### *anomaly detection*

- identificar eventos incomuns  
- ex: detecção de fraudes

### *dimensionality reduction*

- comprimir um conjunto de big-data para um conjunto menor perdendo o mínimo de informação possível



