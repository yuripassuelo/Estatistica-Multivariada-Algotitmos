# Algoritmos Estatística Multivariada

## Obejtivo
O presente diretório apresenta uma série de algoritmos de estatística multivariada como:

- Regressão Linear;
- Regressão Logística;
	- Versão simples com saída Binária;
	- Versão Softmax para saídas com dimensões maiores que n = 2;
	- Algoritmos de classificação como One vs One e One vs Rest;
- Algoritmos de Arvore e suas multiplas derivações:
	- ID3: Suporta apenas variáveis de classe;
	- C4.5: Suporta variáveis independentes de classe e numéricas;
	- CART: Arvore de regressão que suporta variável dependente continua e independentes continuas e de classe.

## 1. Estrutura do diretório:

**src** : Contém os scripts das classes de cada algoritmo, cada script é exclusivo para um assunto, assim temos os scripts:
- **Logistic_Simples_Multinomial.py**: Algoritmos de regressão Logística que contém diferentes versões, uma versão usa um otimizador do pacote `scipy`, enquanto outro implementa manualmente o algoritmo de *gradient descent*;
- **Multiclass_Logistic_Regression_Classifier.py**: Scripts relacionados a algoritmos de classificação multinomial `One vs One` e `One vs Rest`.Para esse script em especifico os modelos de regressão logística utilizados para contrução dos algoritmos *One vs One* e *One vs Rest* são provenientes dos pacotes `scikit learn`, porém a estrutura é feita de forma manual;
- **Softmax_Logistic_Regression.py**: Algoritmo de regrssão logística multinomial, diferente de uma regressão logística comum o algoritmo consegue classificar mais de 2 classes, diferentemente do algoritmo de saída binária, da mesma forma que o script `Logistic_Simples_Multinomial.py` tem duas versões, a versão com algoritmo de otimização do pacote `scipy` e outro que utiliza o algoritmo *gradient descent* ;
- **Trees_Algorithms.py**: Algoritmos de arvore, os algoritmos de arvore em diferentes versões, contendo os algoritmos `ID3`, `C4.5` e `CART` ;

**notebooks** : Notebooks mostrando a utilização destes algoritmos de teste e faz comparações com funções do pacote `scikit learn`, cada notebook conta com algumas explicações teóricas com formulas e exemplos com alguns *datasets* para mostrar exemplos e lógicas de funcionamento dos algoritmos.

## Observações

Os algoritmos não são "puros" e tem uma série de dependências em outros pacotes, alguns pacotes utilizados são:
- numpy ;
- pandas ;
- scipy ;
- sklearn ;
- itertools ;
- functools ;
- collections ;
- re ;
- matplotlib ;
- seaborn .


Outra observação importante é que os algoritmos foram construidos com um intuito de exemplificar seu funcionamento em situações simples, não são feitos necessariamente para ser o mais eficiente possíveis e é possivel que em algumas situações não funcionem propriamente.


*Futuras Inclusões:*

- Algoritmos de redes neurais;
- Algoritmos de classificação e regressão como boost;
- Random Forest;