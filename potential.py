import numpy as np

def PotentialSphere(X, Xo, h, theta):
    v = 50 * ((X - Xo) * (np.cos(theta * np.pi / 180)) - h * (np.sin(theta * np.pi / 180))) / ((X - Xo)**2 + h**2)**1.5
    return v

def PotentialCylinder(X, Xo, h, theta):
    v = 50 * ((X - Xo) * (np.cos(theta * np.pi / 180)) - h * (np.sin(theta * np.pi / 180))) / ((X - Xo)**2 + h**2)
    return v

def AnalyticalSignal(X, Xo, h):
    anSig = 50/((X-Xo)**2+h**2)
    return anSig
