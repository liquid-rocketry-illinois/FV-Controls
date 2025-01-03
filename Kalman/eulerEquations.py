import numpy as np
import csv

class state:
    def __init__(self, angular, inertias, torques, filePath):
        #Define all the class variables
        self.angular = np.array(angular)
        self.inertias = np.array(inertias)
        self.torques = np.array(torques)
        self.filePath = filePath

    def parseData(self):
        #take data from rocket and extract angular velocity, inertial components, and torque components
        self.times = []
        self.verticalVelocities = []
        self.masses = []
        self.thrusts = []
        self.dragForces = []
        self.dragCoefficients = []
        self.airTemperatures = []
        self.airPressures = []
        with open(self.filePath, "r", newline="") as file:
            reader = csv.reader(file, delimiter=",")
            for row in reader:
                if row[0][0] != '#':
                    self.times.append(row[0])
                    self.verticalVelocities.append(row[1])
                    self.masses.append(row[2])
                    self.thrusts.append(row[3])
                    self.dragForces.append(row[4])
                    self.dragCoefficients.append(row[5])
                    self.airTemperatures.append(row[6])
                    self.airPressures.append(row[7])
            return self.times, self.verticalVelocities, self.masses, self.thrusts, self.dragForces, self.dragCoefficients, self.airTemperatures, self.airPressures
    
    def calculateDrag_v(self, time):
        GASCONSTANT = 287.052874
        index = self.times.index(time)
        verticalVelocity = self.verticalVelocities[index]
        dragForce = self.dragForces[index]
        dragCoefficient = self.dragCoefficients[index]
        airTemperature = self.airTemperatures[index]
        airPressure = self.airPressures[index]
        
        airDensity = airPressure/(airTemperature*GASCONSTANT)
        area = (2*dragForce)/(airDensity*dragCoefficient*(verticalVelocity**2))
        drag_v = airDensity*verticalVelocity*dragCoefficient*area
        return drag_v

    def calculateA(self, time):
        #Cols of A: 
        #[w_1, w_2, w_3, v_x, v_y, v_z,]
        
        index = self.times.index(time)
    
        mass = self.masses[index]
        
        Fg = mass * 9.81
        thrust = self.thrusts[index]
        drag_v = self.calculateDrag_v(time)

        A = [[0, -((self.inertias[0]-self.inertias[2])*self.angular[2])/self.inertias[1], -((self.inertias[1]-self.inertias[0])*self.angular[1])/self.inertias[2], 0, 0, 0],
             [-((self.inertias[2]-self.inertias[1])*self.angular[2])/self.inertias[0], 0, -((self.inertias[1]-self.inertias[0])*self.angular[0])/self.inertias[2], 0, 0 ,0],
             [-((self.inertias[2]-self.inertias[1])*self.angular[1])/self.inertias[0], -((self.inertias[0]-self.inertias[2])*self.angular[0])/self.inertias[1], 0, 0, 0, (thrust-drag_v-Fg)/mass]]

        return A
    
    def calculateB(self, time):
        c = 0.1
        B = np.zeros((4, 4))
        B[0][0] = c
        return B

    def getState(self, time):
        index = self.times.index(time)
        verticalVelocity = self.verticalVelocities[index]
        
        currentState = [
            self.angular[0],
            self.angular[1],
            self.angular[2],
            0,
            0,
            verticalVelocity
            ]
        return currentState

angular = [0, 0, 0]
torques = [0, 0, 0]
inertias = [0, 0, 0]
filePath = "flightVehicleDataV3.csv"

rocket = state(angular, torques, inertias, filePath)