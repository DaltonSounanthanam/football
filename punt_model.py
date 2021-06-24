#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 18:48:12 2021

@author: dalton
"""

from matplotlib import pyplot as plt
import numpy as np

### build the model ###


def position_x(t, V0, theta):
    return V0*np.cos(theta)*t 

def position_y(t, V0, theta):
    return -0.5*9.81*t**2 + V0*np.sin(theta)*t + 1

def velocity_x(t, V0, theta):
    return V0*np.cos(theta)

def velocity_y(t, V0, theta):
    return -9.81*t + V0*np.sin(theta)

def acceleration_x(t):
    return 0

def acceleration_y(t):
    return -9.81
    


def simulation(V0, theta, step = 0.1):
    X = []
    Y = []
    xt = 0
    yt = 1
    t = 0
    while yt > 0:
        xt = position_x(t, V0, theta)
        X.append(xt)
        yt = position_y(t, V0, theta)
        Y.append(yt)
        t = t + step
    return X, Y, t
         
X, Y, T = simulation(30, 3*np.pi/7)
plt.plot(X,Y, c = 'brown')
    
    