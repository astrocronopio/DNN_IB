from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier,plot_tree
from sklearn.ensemble import BaggingRegressor, AdaBoostRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import pandas 
import numpy as np
import graphviz 

import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.rcParams.update({
'font.size': 20,
'figure.figsize': [12, 11],
'figure.autolayout': True,
'font.family': 'serif',
'font.sans-serif': ['Palatino']})

class arbol_decision(object):
    def __init__(self, type_tree):
        self.tree = type_tree
        
    def train_tree(self, X, y):
        self.tree=self.tree.fit(X,y)
        
    def test_tree(self, X_test):
        return self.tree.predict(X_test)
    
    def acc_error(self, y_pred,train_x, train_y, val_x, val_y ):
        self.acc= self.tree.score(train_x, train_y)
        self.val_acc= self.tree.score(val_x, val_y)
        self.error=np.sum((val_y-y_pred)**2)/len(val_y)**2
        print("Error <tree>: {} de error de {} ejemplos. {}%, acc:{}, val_acc:{}".format(                                                                  
                self.error, len(val_y), 100*np.sum(val_y-y_pred)/len(val_y),
                self.acc, self.val_acc))        
    
    def plot_save_tree(self, val_y, y_pred, plot_path):
        names= ["CompPrice","Income","Advertising",
        "Population","Precio","Posición en vitrina","Edad",
        "Education","Urban","US"]
        plt.figure(12)
        plot_tree(self.tree, filled=True, feature_names=names, rounded=True, fontsize=16,
        class_names=["Por encima  de 8", "Por debajo de 8"])
        
        fig = mpl.pyplot.gcf()
        fig.set_size_inches(14, 12)
        fig.savefig("tree_"+plot_path)
        
        plt.figure(5)
        fig1 = mpl.pyplot.gcf()
        fig1.set_size_inches(12,8)
        plt.xlabel("Dato")
        plt.ylabel("Predicho")
        plt.scatter(val_y, y_pred, alpha=0.6)
        plt.plot(val_y, val_y, color='black', alpha=0.6)
        fig1.savefig("fit_"+plot_path)
        plt.show()

def data_loading():
    df = pandas.read_csv("./Carseats.csv", header='infer')
    df.replace(('Yes', 'No'), (1, 0), inplace=True)
    df.replace(('Good', 'Medium', 'Bad'), (3, 2, 1), inplace=True)
    
    df = df.to_numpy()
    sales = df[:,0]
    high = np.array([[ 1 if sale>8 else 0 for sale in sales]]).T
    df = np.concatenate((df, high), axis=1 )
    return df

# Podria haber usado sklearn.model_selection.train_test_split
def split_data(train, val, df):   
    train= int(400*train)
    val= int(400*val)

    df_train = df[:train ,:]
    df_val = df[-val:,:]
    
    return df_train, df_val

def separate_data(data):
    return  data[:,1:-2], data[:,0], data[:,-1]


def item_B(train_x, train_y, val_x, val_y, plot_option=False):
    B_tree = arbol_decision(DecisionTreeClassifier(max_depth=2))
    B_tree.train_tree(train_x, train_y)
    y_pred = B_tree.test_tree(val_x)
    
    if plot_option==True:
        B_tree.plot_save_tree(val_y, y_pred, "B_tree.pdf")
    
    B_tree.acc_error(y_pred,train_x, train_y, val_x, val_y )        
    
def item_C(train_x, train_y, val_x, val_y, plot_option=False):
    C_tree = arbol_decision(DecisionTreeRegressor(max_depth=2))
    C_tree.train_tree(train_x, train_y)
    y_pred = C_tree.test_tree(val_x)

    if plot_option==True:
        C_tree.plot_save_tree(val_y, y_pred, "C_tree.pdf")

    C_tree.acc_error(y_pred,train_x, train_y, val_x, val_y ) 
    
    parameters = {'max_depth':range(2,20)}
    clf = GridSearchCV(DecisionTreeRegressor(), parameters)
    clf.fit(X=train_x, y=train_y)
    tree_model = clf.best_estimator_
    print("Profundidad óptima\n")
    print (clf.best_score_, clf.best_params_) 
    
def item_E(train_x, train_y, val_x, val_y, plot_option=False):
    E_tree = DecisionTreeRegressor(random_state=0)
    parameters = E_tree.cost_complexity_pruning_path(train_x, train_y)
    ccp_alphas, impurities = parameters.ccp_alphas, parameters.impurities

    regressor_forest = []
    for ccp_alpha in ccp_alphas:
        regressor_tree = DecisionTreeRegressor(random_state=0, ccp_alpha=ccp_alpha)
        regressor_tree.fit(train_x, train_y)
        regressor_forest.append(regressor_tree)
    
    nodo_per_tree= [arbol.tree_.node_count for arbol in regressor_forest]
    max_depth = [arbol.tree_.max_depth for arbol in regressor_forest]
    
    train_scores = [arbol.score(train_x, train_y) for arbol in regressor_forest]
    test_scores = [arbol.score(val_x, val_y) for arbol in regressor_forest]
    
    
    fig,ax = plt.subplots()
    ax.set_ylabel("accuracy")
    ax.set_xlabel("ccp $\\alpha$")
    ax.plot(ccp_alphas[:-2], train_scores[:-2], marker='o', label="train",drawstyle="steps-post", color="blue", alpha=0.7)
    ax.plot(ccp_alphas[:-2], test_scores[:-2], marker='s', label="val",drawstyle="steps-post", color="red", alpha=0.7)
    ax.legend(loc=0)
    ax2=ax.twinx()
    ax2.plot(ccp_alphas[:-2], max_depth[:-2], marker='^', label="profundidad",drawstyle="steps-post", color="green", alpha=0.7)
    ax2.set_ylabel("Profundidad")
    ax2.legend(loc='center right')
    
    
    E_tree_optimo = arbol_decision(DecisionTreeRegressor(max_depth=5, ccp_alpha=0.25))
    E_tree_optimo.train_tree(train_x, train_y)
    y_pred = E_tree_optimo.test_tree(val_x)

    if plot_option==True:
        E_tree_optimo.plot_save_tree(val_y, y_pred, "E_tree_optimo.pdf")
        
    plt.show()

    E_tree_optimo.acc_error(y_pred,train_x, train_y, val_x, val_y )

def item_F(train_x, train_y, val_x, val_y, plot_option=False):
    F_tree = arbol_decision(BaggingRegressor(base_estimator_= None,  n_estimators=8, random_state=0))
    
    FF_tree = arbol_decision(DecisionTreeRegressor(max_depth=1, random_state=0))
    F_tree.train_tree(train_x, train_y)
    y_pred = F_tree.test_tree(val_x)

    FF_tree.train_tree(train_x, train_y)
    yFF_pred = FF_tree.test_tree(val_x)

    plt.figure(5)
    plt.xlabel("Dato")
    plt.ylabel("Predicho")
    plt.scatter(val_y, y_pred, alpha=0.6)
    plt.scatter(val_y, yFF_pred, c='red', alpha=0.6)
    
    plt.plot(val_y, val_y, color='black', alpha=0.6)
    plt.savefig("F_tree.pdf")
    plt.show()
                 
    print("Error <F_tree>: {} errores de {}. {}%, acc:{}, val_acc:{}".format(                                                                  
            np.sum(val_y-y_pred), len(val_y), 
            100*np.sum(val_y-y_pred)/len(val_y),
            F_tree.tree.score(train_x, train_y),
            F_tree.tree.score(val_x, val_y)))
    
    print("Error <FF_tree>: {} errores de {}. {}%, acc:{}, val_acc:{}".format(                                                                  
        np.sum(val_y-yFF_pred), len(val_y), 
        100*np.sum(val_y-yFF_pred)/len(val_y),
        FF_tree.tree.score(train_x, train_y),
        FF_tree.tree.score(val_x, val_y)))
  
def item_G(train_x, train_y, val_x, val_y, plot_option=False):
    E_tree = RandomForestRegressor(random_state=0)
    parameters = E_tree.cost_complexity_pruning_path(train_x, train_y)
    ccp_alphas, impurities = parameters.ccp_alphas, parameters.impurities

    regressor_forest = []
    for ccp_alpha in ccp_alphas:
        regressor_tree = DecisionTreeRegressor(random_state=0, ccp_alpha=ccp_alpha)
        regressor_tree.fit(train_x, train_y)
        regressor_forest.append(regressor_tree)
    
    nodo_per_tree= [arbol.tree_.node_count for arbol in regressor_forest]
    max_depth = [arbol.tree_.max_depth for arbol in regressor_forest]
    
    train_scores = [arbol.score(train_x, train_y) for arbol in regressor_forest]
    test_scores = [arbol.score(val_x, val_y) for arbol in regressor_forest]
    
    
    fig,ax = plt.subplots()
    ax.set_ylabel("accuracy")
    ax.set_xlabel("ccp $\\alpha$")
    ax.plot(ccp_alphas[:-2], train_scores[:-2], marker='o', label="train",drawstyle="steps-post", color="blue", alpha=0.7)
    ax.plot(ccp_alphas[:-2], test_scores[:-2], marker='s', label="val",drawstyle="steps-post", color="red", alpha=0.7)
    ax.legend(loc=0)
    ax2=ax.twinx()
    ax2.plot(ccp_alphas[:-2], max_depth[:-2], marker='^', label="profundidad",drawstyle="steps-post", color="green", alpha=0.7)
    ax2.set_ylabel("Profundidad")
    ax2.legend(loc='center right')

    G_tree =  arbol_decision(RandomForestRegressor())
    G_tree.train_tree(train_x, train_y)
    y_pred = G_tree.test_tree(val_x)
    
    if plot_option==True:
        G_tree.plot_save_tree(val_y, y_pred, "G_tree.pdf")
    
    G_tree.acc_error(y_pred,train_x, train_y, val_x, val_y )  
    
    pass
          
def main():    
    data = data_loading()        
    train, val = split_data(0.8,0.2, data)

    train_split, train_sales, train_high = separate_data(train)
    val_split, val_sales, val_high = separate_data(val)
    
    print("Tree B")
    item_B(train_split, train_high, val_split, val_high)
    
    print("Tree C")
    # item_C(train_split, train_sales, val_split, val_sales )
    
    print("Tree E")
    # item_E(train_split, train_sales, val_split, val_sales)
    
    print("Tree F")
    # item_F(train_split, train_sales, val_split, val_sales)
    
    print("Tree G")
    #item_G(train_split, train_sales, val_split, val_sales, True)
    

if __name__ == '__main__':
    main()
    