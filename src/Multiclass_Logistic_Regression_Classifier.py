
import numpy     as np
import scipy     as sc
import pandas    as pd
import itertools

import sklearn
from sklearn.linear_model import LogisticRegression

from functools   import reduce
from collections import Counter

class MultiClass_Model:
    
    def __init__( self, X , y, intercept = False, penalty = 'l2', random_state = 0 ):
        
        self.X = X
        self.y = y
        
        self.y_s = np.unique( y )
        
        self.intercept = intercept
        self.penalty   = penalty
        self.random_state = random_state
        
    def column_check( self, comp ):
        
        return lambda d: d["class"] == comp
    
    def comb_unicas( self, l ):
    
        combs = []
        
        for i in l:
            for j in l:
                if i == j or [j,i] in combs:
                    pass
                else:
                    combs.append( [i , j] )
        
        return combs
    
    def model_Logistic( self, X, y, _itens, _filter = False ):
        
        #print( y[ y.isin( _itens ) ] )
        
        if _filter:
            Xs, Ys = X.loc[ y.isin( _itens ) , : ] , y[ y.isin( _itens ) ]
        
        else:
            Xs, Ys = X, y
            
        # Modelo
    
        model = LogisticRegression( random_state  = self.random_state, 
                                    fit_intercept = self.intercept, 
                                    penalty       = self.penalty ).fit( Xs, Ys )
    
        return model

    def One_Vs_One( self ):
        
        self.model_name = "One_Vs_One"
        
        # y_s = np.unique( self.y )
        
        combs_unicas = self.comb_unicas( self.y_s )
        
        #_filter = True
        
        # Calculo dos modelos
        
        print( combs_unicas )
        
        self.models     = list( map( lambda k: self.model_Logistic( X = self.X, y = self.y,  _itens = k , _filter = True ), combs_unicas  ) )
        
        self.pred_proba = list( map( lambda m: m.predict( self.X ), self.models ) )
        
        pred_class = list( map( lambda m: m.predict( self.X ), self.models ) )
        
        pred_shapes  = np.array( pred_class ).shape
        
        probas_class = [ [ pred_class[i][j] for i in range( 0, pred_shapes[0] ) ] for j in range( 0, pred_shapes[1] ) ] 
    
        # Classificação Final
    
        self.Class_Final  = [ list( Counter( probas_class[ i ] ).keys() )[ list( Counter( probas_class[ i ] ).values() ).index( max( list( Counter( probas_class[ i ] ).values() ) ) ) ] for i in range( 0, len( probas_class ) ) ]
    
        
        return self
        
        
    def One_Vs_Rest( self ):
        
        self.model_name = "One_Vs_Rest"
        
        # y_s = np.unique( self.y )
        
        # Calculo dos modelos
        
        self.models     = list( map( lambda k: self.model_Logistic( X = self.X, y = self.y == k, _itens = k, _filter = False ) , self.y_s ) )
        
        self.pred_proba = list( map( lambda m: m.predict_proba( self.X ), self.models ) )
        
        probas_class    = list( map( lambda k: list( map( lambda i : i[1], self.pred_proba[k] ) ), list( range(0, len( self.pred_proba ) ) ) ) )  
        
        shapes          = np.array( probas_class ).shape
        
        # Classificação final
        
        self.scores      = [ [ probas_class[i][j] for i in range( 0, shapes[0] ) ] for j in range( 0, shapes[1] ) ]
    
        # print( self.scores )
    
        self.Class_Final = list( map( lambda d: self.y_s[ d.index( max( d ) ) ], self.scores ) )
        
        return self
        
        
    def predict( self, Xs ):
        
        if self.model_name == "One_Vs_One":
            
            self.predicted_probas = list( map( lambda m: m.predict( Xs ), self.models ) )
            
            pred_class   = list( map( lambda m: m.predict( self.X ), self.models ) )
        
            pred_shapes  = np.array( pred_class ).shape
        
            probas_class = [ [ pred_class[i][j] for i in range( 0, pred_shapes[0] ) ] for j in range( 0, pred_shapes[1] ) ] 
    
            # Classificação Final
    
            self.predicted_class  = [ list( Counter( probas_class[ i ] ).keys() )[ list( Counter( probas_class[ i ] ).values() ).index( max( list( Counter( probas_class[ i ] ).values() ) ) ) ] for i in range( 0, len( probas_class ) ) ]
            
            return self
            
            
        if self.model_name == "One_Vs_Rest":
        
            self.predicted_probas = list( map( lambda m: m.predict_proba( Xs ), self.models ) )
            
            probas_class    = list( map( lambda k: list( map( lambda i : i[1], self.predicted_probas[k] ) ), list( range(0, len( self.predicted_probas ) ) ) ) )  
            
            shapes = np.array( probas_class ).shape
        
            # Classificação final
        
            scores      = [ [ probas_class[i][j] for i in range( 0, shapes[0] ) ] for j in range( 0, shapes[1] ) ]
            
            self.predicted_class  = list( map( lambda d: self.y_s[ d.index( max( d ) ) ], scores ) )
            
            return self
        
        
        
