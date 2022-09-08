
import numpy as np
import scipy as sc

import pandas as pd
import itertools
"""
Implementação Manual do modelo de regressão logística simples.

Classe deve receber os parametros:
    X : Lista de variaveis Independentes
    y : Variavel Dependente ( Binaria )
    - Opicionais
    int_ : Intercepto ( Modelo calculado com Intercepto se True )

Classe após a inclusão dos parametros é possivel chamar metodo `fit`.

    - Calcula um modelo de regressão logística simples e deixa disponivel o objeto `model`

    - Objeto `model` pode conter o histórico de parametros, os parametros finais e o custo associado.

Com a aplicação do metodo `fit`, é possivel aplicar o metodo `predict`, que para um dando input `Xp` sendo
    	Xp - Array de imput de variaveis Independentes.

Calcula o resultado de classificação do modelo com as probabilidades calculadas e a classe inferida para cada observação

"""


class Logistic_Reg:
    
    def __init__( self, X, y, int_ = True ):
        """
        Inicialização
        --------------
        X    : Lista de variaveis Independentes
        y    : Variavel Dependente ( Binaria )
        int_ : Intercepto ( Modelo calculado com Intercepto se True )
        --------------
        """
        
        self.X    = X
        self.y    = y
        
        self.int_ = int_
        
        if int_ :
            self.b    = np.zeros( len( X ) + 1 )
            
        else :
            self.b    = np.zeros( len( X ) )
        
    def Sig( self, X, b ):
        """
        Função Sigmoid
        --------------
        X : Lista de variaveis Independentes
        b : Lista de Parametros
        
        --------------
        return 
        --------------
        
        Lista com as probabilidades da funcao Sigmoid
        f( X ) = 1 / ( 1 + exp ( X'b ) ) 
        --------------
        """
    
        return 1/(1 + np.exp( - X @ b ) )

    def cost( self, b, X, y ):
        """
        Função Custo
        --------------
        b : Lista de Parametros
        X : Lista de variaveis Independentes
        y : Variavel Dependente ( Binaria )
        
        --------------
        return 
        --------------
        
        Custo total do modeo
        y*log( Sig ) - (1-y)*log( 1 - Sig )
        --------------
        """
    
        p = self.Sig( X, b )
        N = len( y )
    
        return  - np.sum( y.reshape( N ).astype(float) * np.log( p ) + (1-y.reshape( N ).astype(float))*np.log( ( 1 - p ) ) )
    
    def gradient( self, b, X, y ):
        """
        Calculo do Gradiente dos parametros
        --------------
        b : Lista de Parametros
        X : Lista de variaveis Independentes
        y : Variavel Dependente ( Binaria )
        
        --------------
        return 
        --------------
        
        Gradiente por variavel
        ( 1/N ) * X.T * ( Sig - y )
        --------------
        """
        
    
        #print( b )
    
        p = self.Sig( X, b )
        N = len( y )
    
        grad =  (1/N) * X.T @ ( p - y.reshape( N ).astype(float) ) 
    
        return grad
        
    def update_weigths( self, X, y, lr, itr, reg = "none", lamb = 1.0 ):
        """
        Calculo do Gradiente dos parametros
        --------------
        X    : Lista de variaveis Independentes
        y    : Variavel Dependente ( Binaria )
        lr   : Learning Rate
        itr  : Número de iterações
        reg  : Regularização ( `l1`, `l2`, `none` ) 
        lamb : Lambda Regularização
        
        --------------
        return 
        --------------
        
        Historico de Custo, e dos parametros por iterações
        --------------
        """
    
        b = np.zeros( X.shape[1] )
    
        cost_hist = []
        par_hist  = [ b ]
    
    
        if reg == "l2":
            l1, l2 = 0, 1
        if reg == "l1":
            l1, l2 = 1, 0
        if reg == "none":
            l1, l2 = 0, 0
    
        #print( l1,l2 )
    
    
        for i in range( 0, itr ):
        
            grd = self.gradient( b, X, y ) 
        
            cst = self.cost( b, X, y )
        
            r = l2*lr*lamb*np.array(b)/len(y)

            b = np.array( b ) - lr*grd - r
        
            par_hist.append( b )
            cost_hist.append( cst )
        
        return cost_hist, par_hist
    
    def fit( self, reg = "none", lamb = 1.0, par_hist = True, lr = 0.5, itr = 100000 ):
        """
        Função Fit do modelo
        --------------
        reg      : Lista de variaveis Independentes
        lamb     : Variavel Dependente ( Binaria )
        par_hist : Guardar Histórico dos parametros
        lr       : Learning Rate
        itr      : Número de iterações

        --------------
        return 
        --------------
        
        Log_Reg model object
        --------------
        """
    
        
        if self.int_ :
            
            self.X = np.concatenate( ( np.ones( len( self.X ) ).reshape( -1, 1 ), self.X ), axis=1 )
        
        # Checando Dimensões
        
        if len( self.X ) != len( self.y ):
            
            raise ValueError("Rows of `X`, different from `y`")
        
        self.lr  = lr
        self.itr = itr
        
        res = self.update_weigths( X = self.X, y = self.y, lr = self.lr, itr = self.itr, reg = reg, lamb = lamb )
        
        if not par_hist :
            _model = { 'final_par' : res[1][len( res[1] ) - 1 ],
                       'cost_hist' : res[0] }
            
        else :
            _model = { 'final_par' : res[1][len( res[1] ) - 1 ],
                       'par_hist'  : res[1],
                       'cost_hist' : res[0] }
        
        self.model =  _model
        
        return self
    
    def predict( self, Xp ):
        
        # Checando dimensões e parametros
        #if len( self.X ) != len( Xp ):
            
        #    raise ValueError("Different number of dimensions between `Xp` and `X`" )
        
        if self.int_ :
            
            Xp = np.concatenate( ( np.ones( len( Xp ) ).reshape( -1, 1 ), Xp ), axis=1 )
        
        #print( self.model['final_par'] )
        #print( self.model['final_par'].shape )
        #print( np.array( Xp ).shape )
        
        probs = self.Sig( np.array( Xp ), self.model['final_par']  )
        
        self.probs = probs
        
        self.pred_clas = np.array( list( map( lambda x: np.argmax( x ), probs ) ) )
        
        return self


"""
Implementação Manual do modelo de regressão logística multinomial.

Classe deve receber os parametros:
    X : Lista de variaveis Independentes
    y : Variavel Dependente ( Binaria )
    - Opicionais
    intercept : Intercepto ( Modelo calculado com Intercepto se True )

Classe após a inclusão dos parametros é possivel chamar metodo `fit`.

    - Calcula um modelo de regressão logística multinomial e deixa disponivel o objeto `model`

    - Objeto `model` pode conter o histórico de parametros, os parametros finais e o custo associado.

Com a aplicação do metodo `fit`, é possivel aplicar o metodo `predict`, que para um dando input `Xp` sendo
    	Xp - Array de imput de variaveis Independentes.

Calcula o resultado de classificação do modelo com as probabilidades calculadas e a classe inferida para cada observação

Método `plot_model` é ideal para casos aonde desejamos apenas plotar a relação de duas variáveis e a classificação.

"""

class Multinomial_LogReg_gd:
    
    def __init__( self, X, y, intercept = False ):
        self.X   = np.array( X )
        self.Y   = np.array( y )
        self.y   = onehot_encoder.fit_transform( np.array( y ).reshape(-1,1))
        self.int = intercept
        self.lr  = lr
        self.itr = itr
        
    def softmax_fun( self, b, X ):
        
        y_hat = np.exp( - X @ b )
        
        sum_yhat = np.sum( y_hat, axis = 1 ).reshape( -1,1 )
        
        return y_hat/sum_yhat 
    
    def cost( self, b, X, y ):
        
        p = self.softmax_fun( b, X )
        
        N = len( y )
        
        return  1/N * (np.trace(X @ b @ y.T) + np.sum(np.log(np.sum(np.exp( - X @ b ), axis=1))))
        
    def gradient( self, b, X, y ):
        
        p = self.softmax_fun( b, X )
    
        N = len( y )

        grad = (1/N)*( X.T @ ( ( y - p ) ) )
    
        return grad
        
    def update_weights( self, X, y, lr, itr ):
        
        # Parametros iniciais
        
        b = np.zeros( ( len( X[0] ), len( y[0] ) ) )

    
        cost_hist, par_hist = [], [ b ]
    
        # Loop de iterações
    
        for i in range( 0, itr ):
                 
            # Calculo do Gradiente e Custo
        
            grd = self.gradient( X = X , y = y, b = b )

            cst = self.cost( b = b, X = X, y = y )
        
            # Atualiza Parametro com Informações do Gradiente
        
            b = np.array( np.array( b ) - lr*grd )
            
            # Armazena histórico
            
            cost_hist.append( cst )
            
            par_hist.append( b )
        
        return { 'Cost_Hist' : cost_hist , 'Par_Hist' : par_hist } #{ "Parametro" : b, "Custo" : cst }
    
    
    def fit( self, par_hist = False, lr = 0.1, itr = 800000 ):
        
        if self.int :
            
            self.X = np.concatenate( ( np.ones( len( self.X ) ).reshape( -1, 1 ), self.X ), axis=1 )
        
        # Checando Dimensões
        
        if len( self.X ) != len( self.y ):
            
            raise ValueError("Rows of `X`, different from `y`")
            
        res = self.update_weights( X = self.X, y = self.y, lr = lr, itr = itr )
        
        if not par_hist :
            _model = { 'final_par' : res['Par_Hist'][len( res['Par_Hist'] ) - 1 ],
                       'cost_hist' : res['Cost_Hist'] }
            
        else :
            _model = { 'final_par' : res['Par_Hist'][len( res['Par_Hist'] ) - 1 ],
                       'par_hist'  : res['Par_Hist'],
                       'cost_hist' : res['Cost_Hist'] }
        
        self.model =  _model
        
        return self
    
    def predict( self, Xp ):
        
        # Checando dimensões e parametros
        #if len( self.X ) != len( Xp ):
            
        #    raise ValueError("Different number of dimensions between `Xp` and `X`" )
        
        if self.int :
            
            Xp = np.concatenate( ( np.ones( len( Xp ) ).reshape( -1, 1 ), Xp ), axis=1 )
        
        #print( self.model['final_par'] )
        #print( self.model['final_par'].shape )
        #print( np.array( Xp ).shape )
        
        probs = self.softmax_fun( self.model['final_par'], np.array( Xp ) )
        
        self.probs = probs
        
        self.pred_clas = np.array( list( map( lambda x: np.argmax( x ), probs ) ) )
        
        return self
    
    def predict_b( self, Xp, bp ):
        
        if self.int :
            
            Xp = np.concatenate( ( np.ones( len( Xp ) ).reshape( -1, 1 ), Xp ), axis=1 )
        
        
        probs = self.softmax_fun( bp , np.array( Xp ) )
        
        pred_clas = np.array( list( map( lambda x: np.argmax( x ), probs ) ) )
        
        return pred_clas
    
    def plot_model( self,  h = 0.01, alpha = 0.5 ):
        
        # Confere se modelo tem Intercepto
        
        d = 0
        
        if self.int :
            
            d = 1
        
        print(  self.X.shape  )
        print(  len( self.X )  )
        # Confere se o modelo tem apenas 2 dimensoes ( X1 e X2 )
        
        if len( self.X[0] ) - d != 2 :
            
            raise ValueError( "Number of dimensions different than 2" )
    
        # Constroi variáveis inputs para plots
    
        x_min, x_max = self.X[:, 0 + d].min() - 0.5, self.X[:, 0 + d].max() + 0.5
        y_min, y_max = self.X[:, 1 + d].min() - 0.5, self.X[:, 1 + d].max() + 0.5
    
        xx, yy = np.meshgrid( np.arange(x_min, x_max, h), np.arange(y_min, y_max, h) )
        
        x_shape = xx.shape
        
        aa = np.ones( x_shape[0]*x_shape[1] ).reshape( x_shape )
        
        Z = self.predict( Xp = np.c_[xx.ravel(), yy.ravel()] ).pred_clas
        Z = Z.reshape(xx.shape)
        
        # Etapa Plot
    
        f, ax = plt.subplots()
    
        ax.pcolormesh( xx, yy, Z, cmap=plt.cm.Pastel1, alpha = alpha )
        ax.scatter( self.X[:, 0+d], self.X[:, 1+d], c= self.Y, edgecolors="k", cmap=plt.cm.Pastel1 )
    
        return ax
