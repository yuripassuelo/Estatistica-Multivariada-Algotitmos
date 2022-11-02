
import re
import numpy as np
import pandas as pd
import itertools

# Arvore ID3 ( Classificadora com dados categóricos )

    
# Arvore de classificação que aceita dados categóricos e variaveis numéricas

"""
@data - Conjunto de dados utilizados para modelagem no formato Pandas DataFrame
@X    - Vetor do nome das variáveis independentes utilizadas
@y    - Nome da variável utilizada como dependente

-------------
Métodos:

fit - Traça o fit do modelo dado o conjunto de dados e as variáveis dependentes e independentes destacadas;

Parametros Opicionais:
@cutoff     - Ponto de Corte para a classificação final. 

--------------

predict - Dado a estimação da arvore é possivel estimar novos resultados para um conjunto de novas Observações X

Parametros:

@dp - Novos


"""


class ID3_Tree:
    
    def __init__( self, data, X, y ):
        
        self.data = data
        self.X    = X
        self.y    = y
    
    # Calculo da entropia por variavel
    
    def entropy( self, X, y, n ):
        
        clas = list( set( X ) )
        #print( clas, len( x ), list( x ) )
        vector_class = [ [list( X )[j] == i for j in range(0, len(list( X ))) ] for i in clas ]
    
        probs = [ np.mean( list( itertools.compress( y, k ) ) ) for k in vector_class ]
    
        return { n : - np.sum( [ k*np.log2(k) for k in probs ] ) } 
    
    # 
    
    def id3_class_tree( self, data, x, y ):
        
        classes = set( data[y] )
        occur   = [ list(data[y]).count(i) for i in classes ]
        
        if len( classes ) == 1:
            
            return { 'N' : str( len(data[y]) ),
                     'v' : { item : list(data[y]).count(item) for item in set( classes ) },
                     'm' : np.mean(data[y]) }
        
        if len( x ) == 0:
            
            return { 'N' : str( len(data[y]) ),
                     'v' : { item : list(data[y]).count(item) for item in set( classes ) },
                     'm' : np.mean(data[y]) }
        
        # Ordena as variáveis com maior entropia
        
        f_ent = sorted( [ self.entropy( X = data[ i ], 
                                        y = data['y'], 
                                        n = i ) for i in x ], 
                        key = lambda x: list(x.values())[0] )
        
        ord_ent = [ list( dic.keys() )[0] for dic in f_ent ]
        
        return { str( ord_ent[0] ) : { k : 
                   self.id3_class_tree( data = data[ data[ord_ent[0]] == k ], 
                                        x = ord_ent[1:], 
                                        y = y ) for k in set( data[ ord_ent[0] ] ) } }
    
    def classifica(  self, classes : dict ):
        
        d = self.model_
        
        while 'm' not in [ k[0] for k in list( d.items() ) ]:
            
            for i,j in d.items():
                r = list( d.items() )[0]
                
                if len( j ) == 2:
                    for s in range(0, 2):
                        if classes[i] == [ k[0] for k in list( r[1].items() ) ][s]:
                            d = list( r[1].items() )[s][1]
                else :
                    d = list( r[1].items() )[0][1]
        return d['m']
    
    def fit( self, cutoff = 0.5 ):
        
        self.model_ = self.id3_class_tree( data = self.data, x = self.X, y = self.y )
        
        #print( self.model.items() )
    
        self.est_values = [ self.classifica( dict( self.data.loc[ n, self.X ] ) ) for n in range(0, len( self.data )) ]
        
        self.est_class  = [ 1 if v >= cutoff else 0 for v in self.est_values ]
        
        return self
    
    def predict( self, dp, cutoff = 0.5 ):
        
        self.predict_values = [ self.classifica( dict( dp.loc[ n, self.X] ) ) for n in range(0, len( dp ) ) ] 
        
        self.predict_class  = [ 1 if v >= cutoff else 0 for v in self.predict_values ]
        
        return self
    
# Arvore de classificação que aceita dados categóricos e variaveis numéricas

"""
@data - Conjunto de dados utilizados para modelagem no formato Pandas DataFrame
@X    - Vetor do nome das variáveis independentes utilizadas
@y    - Nome da variável utilizada como dependente

-------------
Métodos:

fit - Traça o fit do modelo dado o conjunto de dados e as variáveis dependentes e independentes destacadas;

Parametros Opicionais:
@cutoff     - Ponto de Corte para a classificação final. 

--------------

predict - Dado a estimação da arvore é possivel estimar novos resultados para um conjunto de novas Observações X

Parametros:

@dp - Novos


"""

class C45_Tree:
    
    def __init__( self, data, X, y ):
        
        self.data = data
        self.X    = X
        self.y    = y
        
    def gini( self, x, y, n, c, name = "", k = "" ):
        
        clas = list( set( x ) )
    
        vector_class = [ [list(x)[j] == i for j in range(0, len(list( x ))) ] for i in clas ]
    
        probs = [ np.mean( list( itertools.compress( y, k ) ) ) for k in vector_class ]
        pops  = [ len( list( itertools.compress( y, k ) ) ) for k in vector_class ]
        gini  = [ 2*k*(1-k) for k in probs ]
    
        if c == "cat":
            return { n  : np.sum([ pops[i]*gini[i] for i in range( 0, len( pops ) ) ])/np.sum(pops), 
                     'c': c } 
        
        else :
            return { n      : np.sum([ pops[i]*gini[i] for i in range( 0, len( pops ) ) ])/np.sum(pops), 
                     'c'    : c,
                     'name' : name,
                     'split': k} 
        
    def split_cont_var( self, x, y, n, c ):   
        
        set_test = sorted( set( x ) ) 
        
        if len( set_test ) == 1:
            return self.gini( x = x <= set_test[0], y = y, n = n+"<="+str(set_test[0]), c = c, name = n, k = set_test[0] )
    
        split_x  = [ np.mean([set_test[i-1], set_test[i]]) for i in range(1, len(set_test) ) ]
        ginis    = [ self.gini( x = x <= i, y = y, n = n+"<="+str(i), c = c, name = n, k = i ) for i in split_x ]
    
    
        return sorted( ginis, key = lambda x: list(x.values())[0] )[0]
    
    
    def c45_class_tree( self, data, x, y ):
    
        classes = list( set( data[y] ) )
    
        occur   = [ list(data[y]).count(i) for i in classes ]
    
        # Possiveis Condicoes para encerrar arvore:
        # 1. Todas observacoes pertecem a mesma classe ´y´;
        # 2. Nao existem mais variaveis independentes para serem usadas;
        # 3. Variaveis nao contém informacao adicional (pertecem a uma mesma classe);
    
        if len( classes ) == 1:
            
            return { 'N' : str( len(data[y]) ), 
                     'v' : { item : list(data[y]).count(item) for item in set( classes ) }, 
                     'm' : np.mean(data[y]) }
        
        if len( x ) == 0:
        
            return { 'N' : str( len(data[y]) ), 
                     'v' : { item : list(data[y]).count(item) for item in set( classes ) }, 
                     'm' : np.mean(data[y]) }
        
        if len( x ) == 1 and len( set( data[x[0]] ) ) == 1:
        
            return { 'N' : str( len(data[y]) ), 
                     'v' : { item : list(data[y]).count(item) for item in set( classes ) }, 
                     'm' : np.mean(data[y]) }
        
        if np.sum( [ len( set( data[k] ) ) for k in x ] ) == len( x ):
        
            return { 'N' : str( len(data[y]) ), 
                     'v' : { item : list(data[y]).count(item) for item in set( classes ) }, 
                     'm' : np.mean(data[y]) }
        
            
        # Pega conjunto de variaveis e ordena de acordo com maior Gini (Maior discriminacao)
        # Classifica variavel entre númerica e categorica:
        # 1. Númerica presente no intervalo de inteiros e floats com duas ou mais classes;
        # 2. Categórica apenas aponta variavel binaria (0,1);
    
        f_ent = sorted( [ self.gini( x = data[ i ], 
                                     y = data[ y ], 
                                     n = i,
                                     c = "cat") if len( set( data[i] ) ) == 2 and list( set( data[i] ) ) == [0,1]
                          else self.split_cont_var( x = data[ i ],
                                                    y = data[ y ],
                                                    n = i,
                                                    c = "num") for i in x ], 
                         key = lambda x: list(x.values())[0] )
    
        ord_ent = [ list( dic.keys() )[0] for dic in f_ent ]
    
        # Recursao referente a classificao de arvore em duas opcoes:
        # 1. Saida numérica
        # 2. Saida categorica
    
    
        if f_ent[0]["c"] == "num":
        
            return { str( ord_ent[0] ) : { k :
                        self.c45_class_tree( data = data[ ( data[f_ent[0]["name"]] <= f_ent[0]["split"] ) == k  ],
                                             x = x,
                                             y = y) for k in [True, False] }}
    
        if f_ent[0]["c"] == "cat":
        
            ord_ent_2 = [ re.findall( "^.*?(?=<=)", word )[0] if "<=" in word  else word for word in ord_ent ]
        
            return { str( ord_ent[0] ) : { k : 
                       self.c45_class_tree( data = data[ data[ord_ent[0]] == k ], 
                                            x = ord_ent_2[1:], 
                                            y = y ) for k in set( data[ ord_ent[0] ] ) } }



    def classifica( self, classes : dict ):
    
        d = self.model
    
        while 'm' not in [ k[0] for k in list( d.items() ) ]:
            for i,j in d.items():
                #print( i )
                r = list( d.items() )[0]
                if "<=" in i:
                    string_ = i.split("<=")
                    if classes[ string_[0] ] <= float( string_[1] ):
                        d = list( r[1].items() )[0][1]
                    else:
                        d = list( r[1].items() )[1][1]
                else:
                    d = list( r[1].items() )[0][1]
    
        return d['m']
    

    def fit( self, cutoff = 0.5 ):      
        self.model = self.c45_class_tree( self.data, self.X, self.y )
        
        #print( self.model )
        
        self.est_value = [ self.classifica( dict( self.data.loc[ n, self.X ] ) ) for n in range(0,len(self.data)) ]
        
        self.est_class = [ 1 if v >= cutoff else 0 for v in self.est_value ]
        
        #print( self.pred_class )
        
        return self
    
    def predict( self, dp ):
        
        self.predict_values = [ self.classifica( dict( dp.loc[ n, self.X] ) ) for n in range(0, len( dp ) ) ] 
        
        self.predict_class  = [ 1 if v >= cutoff else 0 for v in self.predict_values ]
        
        return self

# Arvore de classificação e Regressão

"""
@data - Conjunto de dados utilizados para modelagem no formato Pandas DataFrame
@X    - Vetor do nome das variáveis independentes utilizadas
@y    - Nome da variável utilizada como dependente

-------------
Métodos:

fit - Traça o fit do modelo dado o conjunto de dados e as variáveis dependentes e independentes destacadas;

Parametros Opicionais:
@metodo     - Metodo de calculo do erro (MAE, MSE, Poisson Difference)
@max|_depth - Profundidade da arvore, quantos nós a arvore se aprofunda
--------------

predict - Dado a estimação da arvore é possivel estimar novos resultados para um conjunto de novas Observações X

Parametros:

@dp - Novos


"""

class CART_Tree:
    
    def __init__( self, data, X, y ):
        
        self.data = data
        self.X    = X
        self.y    = y
        
    # MAE ( Mean Absolute Error )
    def mae( self, yp, y ):
        return abs( yp - y )

    # Poisson Difference
    def poisson( self, yp, y ):
        return y*np.log( y/yp ) - y + yp

    # MSE ( Mean Squared Error )
    def mse( self, yp, y ):
        return (yp - y)**2

    # Calculo ponto de quebra por meio do desvio de poisson 
    #
    # @s - Ponto de quebra da variavel continua - x
    # @x - Variavel explicativa
    # @y - Variavel Dependente

    def calc_split_poid( self, s, x, y ):
        n1, n2 = len( x[ x <= s ] ), len( x[ x > s ] )
        m1, m2 = np.mean( y[ x <= s ] ), np.mean( y[ x > s ] )
        return np.mean( np.concatenate( ([ self.poisson( m1, y[ x <= s ][i] ) for i in range(0,n1) ],
                                         [ self.poisson( m2, y[ x > s  ][i] ) for i in range(0,n2) ] ) ) )

    # Calculo ponto de quebra por meio do desvio de poisson 
    #
    # @s - Ponto de quebra da variavel continua - x
    # @x - Variavel explicativa
    # @y - Variavel Dependente

    def calc_split_mse( self, s, x, y ):
        n1, n2 = len( x[ x <= s ] ), len( x[ x > s ] )
        m1, m2 = np.mean( y[ x <= s ] ), np.mean( y[ x > s ] )
        return np.mean( np.concatenate( ([ self.mse( m1, y[ x <= s ][i] ) for i in range(0,n1) ],
                                         [ self.mse( m2, y[ x > s  ][i] ) for i in range(0,n2) ] ) ) )
     
        
        
    def get_best_split( self, x, y, name, metodo ):
        
        set_x   = sorted( list( set( x ) ) )
    
        split_x = [ np.mean( [set_x[i-1], set_x[i]] ) for i in range( 1, len( set_x ) ) ]
    
        # print( "splits length = ", len( split_x) )
    
        if metodo == "squared":
            x_best  = [ { name+"<="+str(k): self.calc_split_mse( s = k, x = np.array(x), y = np.array(y))} for k in split_x ]
        
        if metodo == "poisson":
            x_best  = [ { name+"<="+str(k): self.calc_split_poid( s = k, x = np.array(x), y = np.array(y))} for k in split_x ]
    
        return sorted( x_best, key = lambda x: list(x.values())[0] )



    # Constroi Modelo de arvore de regressao
    #
    # @data      - Pandas Data Frame com conjunto de dados
    # @x         - Lista com variaveis explicativas do modelo
    # @y         - String com variavel dependente
    # @metodo    - String sendo "squared" para metodo MSE ou "poisson" para Poisson
    # @max_depth - Maximo de iteracoes permitidas 
    # @return    - retorna um dicionario com as variaveis escolhidas, pontos de corte e resultados

    def reg_tree_model( self, data, x, y, max_depth, metodo = "squared" ):
    
        # Divisão de retorno entre
        #
        # 1.  
        # 2.
        
        if max_depth > 0 :
        
            #print( "md = ",max_depth,
            #       "rows = ", len( data[y] ) )
        
            #print( { i: len( list( set( data[i] ) ) )  for i in x } )
        
            if len( data[y] ) == 1 :
                return { 'm' : np.mean( data[y] ), 'N' : len( data[y] ) }  
        
            x_vars = list( filter( lambda k: len( set( data[k] ) )> 1, x ) )
            #print( x_vars )
        
            if len( x_vars ) == 0:
                return { 'm' : np.mean( data[y] ), 'N' : len( data[y] ) }
        
        
            best_var_splits = sorted( [ self.get_best_split( data[i], data[y], metodo = metodo, name = i )[0] for i in x_vars ],
                                      key = lambda x: list( x.values() )[0] )
        
            items = list( best_var_splits[0].keys() )[0].split("<=")
        
            #print( [ len( set( data.loc[ data[ items[0] ] <= float( items[1] ), items[0] ] ) ),
            #         len( set( data.loc[ data[ items[0] ] >  float( items[1] ), items[0] ] ) ) ] )
        
            #if len( set( data.loc[ data[ items[0] ] <= float( items[1] ), items[0] ] ) ) == 1:
            #    print( "ok" )
            #    return { 'm' : np.mean( data[y] ), 'N' : len( data[y] ) }  
            #else:
            return { "<=".join(items):
                        { True  : self.reg_tree_model( data = data[ data[items[0]] <= float(items[1]) ], 
                                                       x = x, 
                                                       y = y, 
                                                       metodo = metodo, 
                                                       max_depth = max_depth - 1 ),
                         False : self.reg_tree_model( data = data[ data[items[0]] > float(items[1]) ], 
                                                      x = x, 
                                                      y = y, 
                                                      metodo = metodo, 
                                                      max_depth = max_depth - 1 ) } }
        
        if max_depth == 0:
            return { 'm' : np.mean( data[y] ), 'N' : len( data[y] ) }
        
        
        
    def classifica( self, classes : dict ):
        
        d = self.model
        
        while 'm' not in [ k[0] for k in list( d.items() ) ]:
            for i,j in d.items():
                r = list( d.items() )[0]
                if "<=" in i:
                    string_ = i.split("<=")
                    if classes[ string_[0] ] <= float( string_[1] ):
                        d = list( r[1].items() )[0][1]
                    else:
                        d = list( r[1].items() )[1][1]
                else:
                    d = list( r[1].items() )[0][1]
    
        return d['m']
    
    def fit( self, metodo = "squared", max_depth = 5 ):
        
        self.model = self.reg_tree_model( data = self.data, x = self.X, y = self.y, metodo = metodo, max_depth = max_depth ) 
        
        self.est_values = [ self.classifica( dict( self.data.loc[ n, self.X ] ) ) for n in range(0,len( self.data )) ]
        
        return self
    
    def predict( self, dp ):
        
        self.predict_values = [ self.classifica( dict( dp.loc[ n, self.X ]) ) for n in range( 0, len( dp ) ) ]
        
        return self
