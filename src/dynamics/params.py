import sympy as sp
from sympy import *
import numpy as np
import pandas as pd
from typing import Callable
from enum import Enum
import os

class Method(Enum):
    lin_int = "linear interpolation"
    poly_fit = "polynomial fit"
    geo_mod = "geometric model"

class Model:
    def __init__(self, var: str, method: Method, basis: np.ndarray): 
        self.var = var # Name of variable being modeled
        self.method = method # Numerical method being used
        self.basis = basis # Basis of method
        self.coeffs: np.ndarray # Coefficients for model
        self.coeffs = np.zeros(len(self.basis))
    def getVal(self):
        val = [x * y for x, y in zip(self.coeffs, self.basis)]
        return val

    def getMethod(self):
        return self.method
    
    def getBasis(self):
        return self.basis

    def getCoeffs(self):
        return self.coeffs

class Params():
    def __init__(self, rocket: str):
        self.paramname = ""
        self.rockname = rocket
        self.dvars = np.array([])
        self.ivars = np.array([])
        self.models = np.array([])
        self.alldata = np.array([])
        self.exclude = np.array([])
        self.ranges = np.array([])
        self.data = np.array([])

    def read(self):
        pass
    def write(self):

        # Write Independant Variables
        newline = "<indep_var><"
        for i in (range)((len)(self.ivars)):
            newline = newline + (str)(self.ivars[i]) + ','
        newline = newline[:-1] + ">"
        self.alldata = np.append(self.alldata, newline)
        newline = ""

        # Write Numerical Models
        for i in (range)((len)(self.dvars)):
            newline = "<" + self.dvars[i] + "><numeric-method><" + (self.models[i]).getMethod().value + "><numeric-model><coeffs><"
            for j in (range)((len)((self.models[i]).getCoeffs())):
                newline = newline + (str)(self.models[i].getCoeffs()[j]) + ","
            newline = newline[:-1] + "><basis><"
            for j in (range)((len)((self.models[i]).getBasis())):
                newline = newline + (str)(self.models[i].getBasis()[j]) + ","
            newline = newline[:-1] + ">"
            self.alldata = np.append(self.alldata, newline)
            newline = ""

        # Write Data
        self.alldata = np.append(self.alldata, "<data-start>")
        for i in range((len)(self.ivars)):
            if ((str)(self.ivars[i]) not in self.exclude):
                newline = newline + "<" + (str)(self.ivars[i]) + ">"
        for i in range((len)(self.dvars)):
            newline = newline + "<" + (str)(self.dvars[i]) + ">"
        self.alldata = np.append(self.alldata, newline)
        newline = ""
        for i in range((len)(self.data)):
            for j in range((len)(self.data[i])):
                newline = newline + (str)(self.data[i][j]) + ","
            self.alldata = np.append(self.alldata, newline)
            newline = ""

        # Note: FV-Controls MUST be cloned within LRI for code compliance.
        current_dir = os.getcwd() # Uses OS package to get current working directory (cwd)

        while (current_dir[-3:] != 'LRI'): # Checks last three letters to ensure you are in your LRI directory
            current_dir = os.getcwd()
            os.chdir('..')
        path = "FV-Controls/rockets/" + self.rockname
        relative_path = os.path.join(current_dir, path)
        
        os.chdir(relative_path)
        os.mkdir("def")
        os.chdir("def/")

        filename = self.paramname + ".txt"
        for i in range((len)(self.alldata)):
            with open(filename, "a") as f:
                f.write(self.alldata[i])
                f.write("\n")

class Inertia(Params):
    def __init__(self, rocket: str):
        super().__init__(rocket)
        self.paramname = "Inertia"
        self.dvars = np.array(["I1", "I2", "I3", "CG", "m"]) # Longitudinal MOIs, Roll MOI, Center of Gravity, Mass

        t, u = sp.symbols(['t', 'u'])
        self.ivars = np.array([t, u]) # Independent variables for all inertia parameters
        self.models = np.array([])
        for i in (range)((len)(self.dvars)):
            self.models = np.append(self.models, [Model(self.dvars[i], Method.lin_int, self.ivars)])
        
        ### TODO ###
        self.coeffs = np.array([1, 2])
        self.exclude = np.array([(str)(u)])
    
inertia = Inertia("Maurice 2")
inertia.write()