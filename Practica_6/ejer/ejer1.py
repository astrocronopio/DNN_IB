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

cmap = plt.get_cmap('viridis',5)

class arbol_decision(object):
    def __init__(self, type_tree):
        self.tree = type_tree
        
    def train_tree(self, X, y):
        self.tree=self.tree.fit(X,y)
        
    def test_tree(self, X_test):
        return self.tree.predict(X_test)
    
    def error(self, x, y):
        y_pred = self.tree.predict(x)
        error=np.sum((y-y_pred)**2)/len(y)**2
        return error
    
    def acc_error(self, y_pred,train_x, train_y, val_x, val_y ):
        self.acc= self.tree.score(train_x, train_y)
        self.val_acc= self.tree.score(val_x, val_y)
        error=self.error(train_x, train_y)
        val_error=self.error(val_x, val_y)
        
        print("Error <tree>: error:{}/val_error:{}  de {} ejemplos. {}%, acc:{}, val_acc:{}".format(                                                                  
                error, val_error, len(val_y), 100*np.sum(val_y-y_pred)/len(val_y),
                self.acc, self.val_acc))    
        print( "Profundidad: {} \t Leaves: {}".format(self.tree.get_depth(), self.tree.get_n_leaves()))
    
    def plot_save_tree(self, val_y, y_pred, plot_path):
        names= ["CompPrice","Income","Advertising",
        "Population","Precio","Posición en vitrina","Edad",
        "Education","Urban","US"]
        plt.figure(12)
        plot_tree(self.tree, filled=True, feature_names=names, rounded=True, fontsize=21,
        class_names=["Por encima  de 8", "Por debajo de 8"], max_depth=2)
        
        
        fig = mpl.pyplot.gcf()
        fig.set_size_inches(13, 11.5)
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

class bosque_decision(arbol_decision):
    """
    un bosque xD
    """
    pass

    def feat_import(self):
        self.feature_importances = np.mean(
        [tree.feature_importances_ for tree in self.tree.estimators_ ]
        , axis=0)
        return self.feature_importances
    

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
    B_tree = arbol_decision(DecisionTreeClassifier())
    B_tree.train_tree(train_x, train_y)
    y_pred = B_tree.test_tree(val_x)
    
    if plot_option==True:
        B_tree.plot_save_tree(val_y, y_pred, "B_tree.pdf")
    
    B_tree.acc_error(y_pred,train_x, train_y, val_x, val_y )        
    
def item_C(train_x, train_y, val_x, val_y, plot_option=False):
    C_tree = arbol_decision(DecisionTreeRegressor())
    C_tree.train_tree(train_x, train_y)
    y_pred = C_tree.test_tree(val_x)

    if plot_option==True:
        C_tree.plot_save_tree(val_y, y_pred, "C_tree.pdf")

    C_tree.acc_error(y_pred,train_x, train_y, val_x, val_y ) 

def item_E_CV(train_x, train_y, val_x, val_y, plot_option=False):
    parameters = {'max_depth':range(1,20), 'ccp_alpha':np.arange(0,1,0.02)}
    E_tree = GridSearchCV(DecisionTreeRegressor(), parameters)
    E_tree.fit(X=train_x, y=train_y)
    tree_model = E_tree.best_estimator_
    print("Profundidad óptima / Ccp óptimo\n")
    print (E_tree.best_score_, E_tree.best_params_)    
    
def item_E_prunning(train_x, train_y, val_x, val_y, plot_option=False):
    E_tree = DecisionTreeRegressor(random_state=0)
    parameters = E_tree.cost_complexity_pruning_path(train_x, train_y)
    ccp_alphas, impurities = parameters.ccp_alphas, parameters.impurities

    regressor_forest = []
    for ccp_alpha in ccp_alphas:
        regressor_tree = arbol_decision(DecisionTreeRegressor(random_state=0, ccp_alpha=ccp_alpha))
        regressor_tree.train_tree(train_x, train_y)
        regressor_forest.append(regressor_tree)
    
    nodo_per_tree= [arbol.tree.tree_.node_count for arbol in regressor_forest]
    max_depth    = [arbol.tree.tree_.max_depth for arbol in regressor_forest]
    
    train_scores = [arbol.error(train_x, train_y) for arbol in regressor_forest]
    test_scores = [arbol.error(val_x, val_y) for arbol in regressor_forest]
    
    
    fig,ax = plt.subplots()
    ax.set_ylabel("error")
    ax.set_xlabel("ccp $\\alpha$")
    ax.plot(ccp_alphas[:-2], train_scores[:-2], marker='o', label="train",drawstyle="steps-post", color=cmap(1), alpha=0.7)
    ax.plot(ccp_alphas[:-2], test_scores[:-2], marker='s', label="val",drawstyle="steps-post", color=cmap(2), alpha=0.7)
    ax.legend(loc='center')
    
    ax2=ax.twinx()
    ax2.plot(ccp_alphas[:-2], max_depth[:-2], marker='^', label="profundidad",drawstyle="steps-post", color=cmap(3), alpha=0.7)
    ax2.set_ylabel("Profundidad")
    ax2.legend(loc='center right')
    
    
    E_tree_optimo = arbol_decision(DecisionTreeRegressor(max_depth=5, ccp_alpha=0.2))
    E_tree_optimo.train_tree(train_x, train_y)
    y_pred = E_tree_optimo.test_tree(val_x)

    if plot_option==True:
        E_tree_optimo.plot_save_tree(val_y, y_pred, "E_tree_optimo.pdf")
        
    plt.show()

    E_tree_optimo.acc_error(y_pred,train_x, train_y, val_x, val_y )

def item_F(train_x, train_y, val_x, val_y, plot_option=False):
    F_tree = DecisionTreeRegressor()
    F_tree.fit(train_x, train_y)
    feature_simple =F_tree.feature_importances_
    error_simple = F_tree.error(val_x, val_y)
    
    F_bosque = bosque_decision(BaggingRegressor(DecisionTreeRegressor(), n_estimators=25))
    F_bosque.fit(train_x, train_y)
    feature_importances = F_bosque.feat_import()
    error = F_bosque.error(val_x, val_y)
    
    print("Simple Tree: ", feature_simple)
    print("Simple Error: ", error_simple)
    print("\n=======\nBagging: ",feature_importances)
    print("Bagging Error: ",error)
  
def item_G_H(train_x, train_y, val_x, val_y, bosque_type):
    G_bosque = bosque_decision(RandomForestRegressor(n_estimators=25))
    i=0
    if bosque_type=="AdaBoost":
        G_bosque = bosque_decision(RandomForestRegressor(n_estimators=25))
        print(bosque_type)
        i=2
        
    G_bosque.train_tree(train_x, train_y)
    
    feature_importances = G_bosque.feat_import()
    error = G_bosque.error(val_x, val_y)
    
    print(bosque_type+": ",feature_importances)
    print(bosque_type+" Error: ",error)
    
    max_feat = np.arange(1,10,1)
    max_de= np.arange(1,20,1)
    
    regressor_forest_de = []
    error_de=np.array([])
    
    regressor_forest_feat = []
    error_feat=np.array([])
    
    for max_f in max_feat:
        regressor_tree = bosque_decision(RandomForestRegressor(max_features=max_f))
        regressor_tree.train_tree(train_x, train_y)
        regressor_forest_feat.append(regressor_tree)
        error_feat = np.append(error_feat, regressor_tree.error(val_x, val_y))
        
    for max_d in max_de:
        regressor_tree = bosque_decision(RandomForestRegressor(max_depth=max_d))
        regressor_tree.train_tree(train_x, train_y)
        regressor_forest_de.append(regressor_tree)
        error_de = np.append(error_de, regressor_tree.error(val_x, val_y))
    
    plt.figure(3)
    plt.plot(max_feat, error_feat, c=cmap(0+i),label="Features: "+bosque_type)
    plt.legend(loc=0)
    
    plt.figure(4)   
    plt.plot(max_de, error_de, c=cmap(2+i),label="Profundidad: "+bosque_type)
    plt.legend(loc=0)
          
def main():    
    data = data_loading()        
    train, val = split_data(0.8,0.2, data)

    train_split, train_sales, train_high = separate_data(train)
    val_split, val_sales, val_high = separate_data(val)
    
    print("Tree B")
    # item_B(train_split, train_high, val_split, val_high, True)
    
    print("Tree C")
    item_C(train_split, train_sales, val_split, val_sales)
    
    print("Tree E")
    # item_E_CV(train_split, train_sales, val_split, val_sales)
    # item_E_prunning(train_split, train_sales, val_split, val_sales)
    
    print("Tree F")
    # item_F(train_split, train_sales, val_split, val_sales)
    
    print("Tree G")
    # item_G_H(train_split, train_sales, val_split, val_sales, "Random")
    
    print("Tree H")
    # item_G_H(train_split, train_sales, val_split, val_sales, "AdaBoost")
    
    plt.show()


if __name__ == '__main__':
    main()
    