import math
import pandas as pd
import numpy as np
import reg_functions as r
import matplotlib.pyplot as plt
import time
import pylab

import random

# hypothesis h = theta_1 * x + theta_0
def hypothesis(x, theta):
    
    h = np.transpose(np.dot(np.transpose(x), theta))
    return h
    
# Calculate cost function J(theta) = (1/2m) * (sum(h(theta[i]) - y[i])) ** 2 
def cal_cost(theta, x, y):
    h = hypothesis(x, theta)
    
    cost = (1/2) * ((h - y) ** 2)
    cost = np.sum(cost[0])
    
    return cost
    

# Calculate error radient with respect to weights
def err_grad(x, y, theta):
    Y = y
    Yp = np.transpose(np.dot(np.transpose(x), theta))
    er_gr_a = -(Y - Yp)
    er_gr_b = -(Y - Yp) * x[0]
    er_gr_c = -(Y - Yp) * x[1]
    er_gr_d = -(Y - Yp) * x[2]
    er_gr_e = -(Y - Yp) * x[3]
    return er_gr_a, er_gr_b, er_gr_c, er_gr_d, er_gr_e
    
    
def fit(X, Y, iter):
    
    
    
    df = pd.DataFrame(X)
    
    # Add an extra [1, 1, ..., 1] to the data so that we can do
    # multiplication with b values of theta 
    # y' = t1 * x1 + t2 * x0 or (X.T * theta) 
    
    arr1 = np.full(len(X), 1.00)
    
    
    
    X = df.assign(X0 = arr1)
    # X = X.to_numpy()
    X = np.transpose(X)
    
    # print(X)
    
    # Y = df.assign(b0 = arr1)
  
    # X = np.vstack((X,  np.full(len(X), 1.00)))
    # Y = np.vstack((Y,  np.full(len(Y), 1.00)))
    
    X = np.array(X)
    Y = np.array(Y)
    
    # Print Value of X and Y
    print("X: {}\n".format(X))
    print("y: {}\n".format(Y))
    
    # Initializing theta at random numbers and make an array for vector operations
    theta = []
    th = []
    for i in range(5):
        # rnum = random.uniform(0.05, 100.0)
        rnum = i * 0.003
        th.append(rnum)
        theta.append(np.full(len(X),rnum))
    
    
    th_0 = th[0]
    th_1 = th[1]
    th_2 = th[2]
    th_3 = th[3]
    th_4 = th[4]
    
    
    # theta_0 = np.full(len(X), th_0) # b = 0.75
    # theta_1 = np.full(len(X), th_1) # a = 0.45
    # theta_2 = np.full(len(X), th_2) # a = 0.09
    
    # Theta to array (b, a)
    # theta = np.array([theta_0, theta_1, theta_2])
    
    # Print all the values with Initial theta values
    print("\nInitial theta ( weights b and a ) : \n", [th_0, th_1, th_2, th_3, th_4])
    print("\nInitial hypothesis : \n",hypothesis(X, theta))
    print("\nInitial Cost  \n: ", cal_cost(theta, X, Y))
    
    # Before training
    print("\n\nStart teraining.................\n")
    
    # Learning parameters 
    lr = 0.0001
    iteration = 0
    Full_weight_th_0 = []
    Full_weight_th_1 = []
    Full_weight_th_2 = []
    Full_weight_th_3 = []
    Full_weight_th_4 = []
    cost_150_iter = []
    
    while True:
        iteration = iteration + 1
        
        
        err_gr_a, err_gr_b, err_gr_c, err_gr_d, err_gr_e = err_grad(X, Y, theta)
        
        sum_ergr_a = np.sum(err_gr_a[0])
        sum_ergr_b = np.sum(err_gr_b[0])
        sum_ergr_c = np.sum(err_gr_c[0])
        sum_ergr_d = np.sum(err_gr_d[0])
        sum_ergr_e = np.sum(err_gr_e[0])
        # print(sum_ergr_a, sum_ergr_b)
        
        # Theta 2 is w3
        th_4 = th_4 - lr * (1/5) * sum_ergr_e
        
        # Theta 2 is w3
        th_3 = th_3 - lr * (1/5) * sum_ergr_d
        
        # Theta 2 is w3
        th_2 = th_2 - lr * (1/5) * sum_ergr_c
        
        # Theta 1 is a
        th_1 = th_1 - lr * (1/5) * sum_ergr_a
        
        # Theta 0 is b
        th_0 = th_0 - lr * (1/5) * sum_ergr_b
        
        Full_weight_th_0.append(th_0)
        Full_weight_th_1.append(th_1)
        Full_weight_th_2.append(th_2)
        Full_weight_th_3.append(th_3)
        Full_weight_th_4.append(th_4)
        
        
        theta_0 = np.full(len(X), th_0) # b 
        theta_1 = np.full(len(X), th_1) # a
        theta_2 = np.full(len(X), th_2) # c
        theta_3 = np.full(len(X), th_3) # a 
        theta_4 = np.full(len(X), th_4) # d 
        
        # Theta to array (b, a)
        theta = np.array([theta_0, theta_1, theta_2, theta_3, theta_4])
        
        # Cost after nth iteration
        
        costt = cal_cost(theta, X, Y)
        
        cost_150_iter.append(costt)
        print("\nIteration: {}; New Weights: {}, {}, {}, {}, {}; Cost : {}\n".format(iteration, th_0, th_1, th_2, th_3, th_4, costt))
        
        plt.ylabel('Error')
        plt.xlabel('Iterations')
        plt.suptitle('Training progress')
        plt.scatter(iteration, costt)
        # plt.pause(0.001)
        plt.savefig(r'static/images/out.png')
        
        if iteration > iter:
            print("Training Over")
            plt.show(block=False)
            plt.close()
            break
        

    # plt.close()
    # plt.show()
    
    
    Final_weight_th_0 = Full_weight_th_0[cost_150_iter.index(min(cost_150_iter))]
    
    Final_weight_th_1 = Full_weight_th_1[cost_150_iter.index(min(cost_150_iter))]
    
    Final_weight_th_2 = Full_weight_th_2[cost_150_iter.index(min(cost_150_iter))]
    
    Final_weight_th_3 = Full_weight_th_3[cost_150_iter.index(min(cost_150_iter))]
    
    Final_weight_th_4 = Full_weight_th_4[cost_150_iter.index(min(cost_150_iter))]
    
    Final_weihts = [Final_weight_th_0, Final_weight_th_1, Final_weight_th_2, Final_weight_th_3, Final_weight_th_4]
    
    print("\n After Iteration: {}; Final Weight : {}, {}, {}, {}, {} ; Minimum Cost : {}\n".format(iteration,Final_weight_th_0, Final_weight_th_1, Final_weight_th_2, Final_weight_th_3, Final_weight_th_4, min(cost_150_iter)))
    
    cost =  min(cost_150_iter)
    
    return Final_weihts, cost
    
    
def predict(x, weights):
    return 10 * hypothesis(x, weights)    
    

# main
if __name__ == '__main__':
    # Our data, X and Y, They can have multiple dimentions
    X0 = [0.00, 0.22, 0.24, 0.33, 0.37, 0.44, 0.44, 0.57, 0.93, 1.00]
    X1 = [0.00, 0.22, 0.24, 0.33, 0.37, 0.44, 0.44, 0.57, 0.93, 1.00]
    
    X = pd.DataFrame({"X0" : X0, "X1" : X1})
    
    # print(X)
    
    Y = [0.00, 0.22, 0.58, 0.20, 0.55, 0.39, 0.54, 0.53, 1.00, 0.61]
    # X = [1.00, 2.00, 3.00, 4.00, 5.00]
    # Y = [1.00, 2.00, 1.30, 3.75, 2.25]
    fit(X, Y, 300)
    
