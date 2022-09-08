import numpy     as np
import scipy     as sc
import pandas    as pd
import itertools
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.optimize import minimize
from scipy.optimize import fmin_bfgs

from sklearn.preprocessing import OneHotEncoder
onehot_encoder = OneHotEncoder(sparse=False)

"""
Implementação Manual do modelo de regressão logística multinomial ou softmax.

Classe deve receber os parametros:
    X : Lista de variaveis Independentes
    y : Variavel Dependente ( Binaria )
    - Opicionais
    init_par  : Parâmetros iniciais
    Intercept : Intercepto ( Modelo calculado com Intercepto ) padrão é False
    

Classe após a inclusão dos parametros é possivel chamar metodo `fit`.

    - Calcula um modelo de regressão softmax e deixa disponivel o objeto `model`

    - Objeto `model` pode conter o histórico  de parametros, os parametros finais e o custo associado.

Com a aplicação do metodo `fit`, é possivel aplicar o metodo `predict`, que para um dando input `Xp` sendo
    	Xp - Array de imput de variaveis Independentes.

Calcula o resultado de classificação do modelo com as probabilidades calculadas e a classe inferida para cada observação

"""

class Multinomial_LogReg:
    
    def __init__( self, X, y, init_par = None, Intercept = False ):
        
        self._int     = Intercept
        self.X        = X
        self.y        = y
        self.init_par = init_par
    
    def softmax( self, b ):
        '''
        Função Softmax
        --------------
        b : Lista de Parametros ( n = classes )
        X : Lista de variaveis Independentes
        
        --------------
        return 
        --------------
        
        Lista com as probabilidades da funcao SoftMax
        `exp( bi'x ) / sum_j_N exp( bj'x )
        --------------
        '''
        y_hat = np.dot( b.T, self.X )
        
        y_hat = np.exp( y_hat )
        
        sum_yhat = np.sum( y_hat, 0 )
        
        return y_hat/sum_yhat 

    # Funcao Custo
    
    def softmax_cost (self, b ):
        '''
        Função de Custo Softmax
        --------------
        b : Lista de Parametros ( n = classes )
        X : Lista de variaveis Independentes
        y : Lista de variaveis dependentes
        
        --------------
        return
        --------------
        Retorna Custo Calculado
        
        '''
        
        p = np.array( b ).reshape( -1, len( self.y ) )
        
        # print( p )

        return - np.mean( np.multiply( self.y, np.log( self.softmax( p ) ) ) )
    

    def obj_function( self, b ):
        '''
        Funçao Objetivo
        --------------
        b : Lista de Parametros
        
        --------------
        return
        --------------
        Retorna calculo de custo
        
        '''
        
        # p = np.array( b ).reshape(-1,len(X))
        
        return self.softmax_cost( b = b )


    def softmax_fit( self ):
        '''
        Função fit Softmax
        --------------
        X : Lista de variaveis Independentes
        y : Lista de variaveis dependentes
        
        --------------
        return
        --------------
        
        '''
        # Caso intercepto
        
        n = 0
        
        if self._int :
             
            self.X = np.append( np.array( [ np.ones( len( self.X[ 0 ] ) ) ] ), self.X  ).reshape( len( self.X ) + 1, len( self.X[ 0 ] ) )
            
            n = 1
            
        # Valida dimensões e parametros

        ndin_x, ndin_y = len( self.X ), len( self.y )
        
        if ndin_x == 0 or ndin_y == 0:
            raise ValueError("Number of dimensions is equal a 0")
        
        row_x, row_y = len( self.X[0] ), len( self.y[0] )
        
        if row_x == 0 or row_y == 0:
            raise ValueError("Number of rows is equal a 0")
            
        if row_x != row_y:
            raise ValueError("Different number of rows between `X` and `y`")
            
        
        # Parametro Inicial
        
        if self.init_par != None:
            
            # Validando parametros Iniciais
            if len( self.init_par ) != ndin_y + n :
                raise ValueError("Dimension of Initial Parameters, different from dependent variable `y`")
            
            else: 
                
                par_lens = [ len( i ) == ndin_x for i in self.init_par ]
                
                if np.sum( par_lens ) != len( par_lens ) :
                    ValueError("Different Numebers of dimension passed as initial parameters from indepdente variables")
                    
                
            init_par = self.init_par
            
        # print( self.init_par )
        
        # print( ndin_x )
            
        if self.init_par == None:
            
            # Cria parametros Iniciais
            init_par = [ [ 0 for j in range(0, ndin_x  ) ] for k in range(0, len( self.y ) ) ]
            
        # print( init_par )
        # print( self.X )
            
        return fmin_bfgs( self.obj_function, init_par )
    
    def predict( self, X ):
        
        '''
        Predict Probability for the classes given set of response variables
        --------------
        X : Vector os Independent variables
        
        --------------
        return
        --------------
        Predicted probalities for `n` classes
        
        '''
        
        model_par = np.array( self.softmax_fit( ) ).reshape( -1, len( self.y ) )
        
        
        return self.softmax( model_par, self.X ).reshape( -1, len( self.y ) )




def plot_softmax_model( model_, X, y, h = 0.01, alpha = 0.5 ):
    
    '''
    Função de plot fronteiras Softmax
    --------------
    model_ : Objeto de Modelo Softmax - SciKit Learn
    X      : Objeto Pandas DataFrame com as variáveis independentes
    y      : Objeto Pandas DataFrame com a variável dependente
    h      : Passo para construir a matrix de variáveis dependentes
    alpha  : Parametro `alpha` de transparencia
    --------------
    return
    --------------
    Objeto subplots da biblioteca matplotlib.pyplot com fronteiras e scatterplot
    '''
    
    
    X = np.array( X ).reshape( -1, len( X.columns ) )
    
    # Constroi variáveis inputs para plots
    
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Predizendo Classes com inputs criados
    
    Z = model_.predict( np.c_[xx.ravel(), yy.ravel()] )
    Z = Z.reshape(xx.shape)
    
    # Etapa Plot
    
    f, ax = plt.subplots()
    
    ax.pcolormesh( xx, yy, Z, cmap=plt.cm.Pastel1, alpha = alpha )
    ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors="k", cmap=plt.cm.Pastel1 )
    

    return ax
    
    #plt.pcolormesh( xx, yy, Z, cmap=plt.cm.Pastel1, alpha = 0.5 )
    #plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolors="k", cmap=plt.cm.Pastel1 )



'''
Versão Gradient Descent do modelo softmax

'''

from sklearn.preprocessing import OneHotEncoder
onehot_encoder = OneHotEncoder(sparse=False)



class Multinomial_LogReg_gd:
    
    def __init__( self, X, y, intercept = False, lr = 0.1, itr = 800000 ):
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
    
    
    def fit( self, par_hist = False ):
        
        if self.int :
            
            self.X = np.concatenate( ( np.ones( len( self.X ) ).reshape( -1, 1 ), self.X ), axis=1 )
        
        # Checando Dimensões
        
        if len( self.X ) != len( self.y ):
            
            raise ValueError("Rows of `X`, different from `y`")
            
        res = self.update_weights( X = self.X, y = self.y, lr = self.lr, itr = self.itr )
        
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

'''
Plot da evolução dos parametros calculados no modelo de softmax via gradient descent

'''

def plot_pars_hist( par_hist ):
    
    shapes = par_hist.shape
    
    data_par = pd.DataFrame( par_hist.reshape( len( par_hist ), shapes[1]*shapes[2] ) )
    
    cords    = [ [j,i]  for j in range( 0, shapes[1] )  for i in range( 0,  shapes[2] ) ]
    
    # Plots
    
    fig, ax = plt.subplots( 3, 3 )

    for i in range( 0, len( cords ) ) :
        
        # print( "b ("+str(cords[i][0])+ ","+ str(cords[i][1] )+ ")" )
        k = r"{"+str(cords[i][0])+","+ str(cords[i][1])+ r"}" 
        
        ax[ cords[i][0], cords[i][1] ].plot(  np.array( data_par[ data_par.columns[ i ] ] ) )
        ax[ cords[i][0], cords[i][1] ].set_title( r"$\theta_"+k+r"$" )
    
    return ax
