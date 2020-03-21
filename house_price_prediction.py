import pandas as pd
import numpy as np
import fit

def train(iter):
    # Load data from CSV file
    df_train = pd.read_csv('data/train.csv')

    # Print 5 rows
    # print(df_train.head(5))

    # Print price, our Y
    Y = np.array(df_train['SalePrice'])
    # print("Price Y :", Y) 

    # Feature 1 : X1 = LotArea
    X1 = np.array(df_train['LotArea'])

    # Feature 2 : X2 = 1stFlrSF
    X2 = np.array(df_train['1stFlrSF'])

    # Feature 3 : X3 = 2ndFlrSF
    X3 = np.array(df_train['2ndFlrSF'])

    # Feature 4 : X4 = GrLivArea
    X4 = np.array(df_train['GrLivArea'])

    X1 = (X1 - np.min(X1)) / (np.max(X1) - np.min(X1))
    X2 = (X2 - np.min(X2)) / (np.max(X2) - np.min(X2))
    X3 = (X3 - np.min(X3)) / (np.max(X3) - np.min(X3))
    X4 = (X4 - np.min(X4)) / (np.max(X4) - np.min(X4))
    Y = (Y - np.min(Y)) / (np.max(Y) - np.min(Y))

    print(X1, X2, X3, X4)

    X = pd.DataFrame({"X1" : X1, "X2" : X2, "X3" : X3, "X4" : X4})



    Weights, cost = fit.fit(X, Y, iter)

    df = pd.DataFrame(Weights)
    df.to_csv(r'data/FinalWeights.csv')
    print("Weights saved to data/FinalWeights.csv")
    
    return cost


def predict(x):
    # Load weightd 
    Weights_df = pd.read_csv('data/FinalWeights.csv')
    Weights = np.array(Weights_df['0'])
    
    print("Price for proparty is : $", fit.predict(x, Weights))
    
    return fit.predict(x, Weights)

def minmax():
    # Test prediction
    df_test = pd.read_csv('data/train.csv')

    x0 = 1
    
    # Feature 1 : X1 = LotArea 
    minx1 = np.min(df_test['LotArea'])
    maxx1 = np.max(df_test['LotArea'])

    x1 = np.array(df_test.at[10,'LotArea'])

    # Feature 2 : X2 = 1stFlrSF
    minx2 = np.min(df_test['1stFlrSF'])
    maxx2 = np.max(df_test['1stFlrSF'])

    x2 = np.array(df_test.at[10, '1stFlrSF'])

    # Feature 3 : x3 = 2ndFlrSF
    minx3 = np.min(df_test['2ndFlrSF'])
    maxx3 = np.max(df_test['2ndFlrSF'])

    x3 = np.array(df_test.at[10, '2ndFlrSF'])

    # Feature 4 : x4 = GrLivArea
    minx4 = np.min(df_test['GrLivArea'])
    maxx4 = np.max(df_test['GrLivArea'])

    x4 = np.array(df_test.at[10, 'GrLivArea'])

    minx = np.array([minx1, minx2, minx3, minx4])
    
    maxx = np.array([maxx1, maxx2, maxx3, maxx4])
    
    x = np.array([x0, x1, x2, x3, x4])
    
    return minx, maxx
