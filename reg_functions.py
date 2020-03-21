# Copyright All Rights Recieved By Sujith S, AiBlocks India Private Limited

# Linear regression implementation, Relation between 2 variables 

#           List of functions
#
# 1. Varience of a destribution from median
# 2. Find the Standard deviation of a distribution
# 3. Input function
# 4. mean Function
# 5. Correlation Function
# 6. Regression line plot function
# 7. Line_plot
# 8. Calculate predicted value of Y from Y and X, Y' = bX + A  
# 9. Sum of Squared difference of Y from Mean. y = (Y - m) ** 2

import math
import numpy as np
import matplotlib.pyplot as plt


# 1. Varience of a destribution from mean
def variance(distribution):
    
    mean = np.sum(distribution) / len(distribution)
    diff_sq = 0
    diff = 0
    variance = 0
    
    for i in range(0, len(distribution)):
        
        diff = distribution[i] - mean
        diff_sq = diff ** 2
        # print("{} {}".format(diff_sq,len(distribution)))
        variance += diff_sq
        # print(variance)
    
    return variance / len(distribution)


# 2. Find the Standard deviation of a distribution

def std_deviation(distribution):
    
    return math.sqrt(variance(distribution))
    


# 3. input x and y

def inputs():
    
    x = []
    y = []
    
    n = int(input("Enter number of Elements: "))
    print("\n\nEnter X Values\n")
    
    for i in range(n):
        m = float(input("\nEnter X[{}]:".format(i)))
        x.append(m)
    print("\n\nEnter Y Values\n")
    
    for i in range(n):
        m = float(input("\nEnter y[{}]:".format(i)))
        y.append(m)
    return x, y

# 4. calculate mean of x and y


def mean(x, y):
    
    # x, y = inputs()

    X = np.asarray(x)
    Y = np.asarray(y)

    x = np.sum(X)/len(x)
    y = np.sum(Y)/len(y)

    return x, y


# 5. Calculate Correlation
def corr(X, Y):
    
    x = 0
    y = 0

    X = np.asarray(X)              # X = [1, 3, 3, 1 ]
    Y = np.asarray(Y)              # Y = [2, 4, 4, 2]

    x, y = mean(X, Y)              # x = 2, y = 3
    x = np.full((1, len(X)), x)    # x = [2, 2, 2, 2]
    y = np.full((1, len(Y)), y)    # y = [3, 3, 3, 3]

    x = X - x
    y = Y - y

    x2 = x ** 2
    y2 = y ** 2
    xy = x * y

    div1 = np.sum(xy)
    a = np.sum(x2) * np.sum(y2)
    b = math.sqrt(a)
    corr = div1 / b

    return x, y, corr

# 6. line through all y'

def Regression_line(X, Y, B, A):   
        
    #   Yd = bX + A 
    Yd = []

    for i in range(0,len(X)):
        
        Yd.append(B * X[i] + A)
    
    # print(Yd)
    
    plt.scatter(X, Y)
    plt.plot(X, Yd)
    # plt.scatter(X, Yd)


    
    plt.show()
    

# 7. Line_plot
def lplot(a,b):
    
    plt.plot(a,b)
    plt.show()
    

#  8. Calculate predicted value of Y from Y and X, Y' = bX + A  
def y_dash(X, Y):
    
    #   Yd = bX + A 
    
    # X, Y = inputs()
    Mx, My = mean(X, Y)

    sx = std_deviation(X)
    # print(sx)
    sy = std_deviation(Y)
    # print(sy)
    
    x, y, corrl = corr(X, Y)

    r = corrl

    b = r * ( sy / sx )
    A = My - ( b * Mx )
    # print(b, A)
    Yd = []

    for i in range(0,len(X)):
        Yd.append(b * X[i] + A)

    return Yd

    
# 9. Sum of Squared difference of Y from Mean. y = (Y - m) ** 2
def y2(array):
    
    mean2 = np.sum(array)
    mean = mean2 / len(array)
    
    # print("Mean : {}\n".format(mean))
    
    y, sq_diff, row = 0, 0, 0

    for Y in array:
        rowa = Y - mean
        row += rowa
        sq_diff = (Y - mean) ** 2
        y = y + sq_diff
        
    return y

# 10. ( Y-Y') ** 2

def y_yd2(Y, Yd):
    
    Y_Ydd, Sq_Y_Ydd = 0, 0
    for i in range(0, len(Yd)):
        Y_Yd = Y[i] - Yd[i]
        Y_Ydd = Y_Yd + Y_Ydd
        
        Sq_Y_Yd = Y_Yd**2 
        Sq_Y_Ydd += Sq_Y_Yd
    return Sq_Y_Ydd

# 11. Standard Error

def stderr(X, Y):
    
    # Predicted value Y'
    Yd = y_dash(X, Y)
    
    # the sum of squares predicted (SSY') 
    SSY_d = y2(Yd)
    
    # The sum of squares error is the sum of the squared errors of prediction.
    SSE = y_yd2(Y, Yd)
    
    # the standard error of the estimate is sqrt( sum((Y - Y') ** 2) / N )
    
    stder = math.sqrt(SSE / (len(Y) - 2))
    
    return stder
    
    






    
