import numpy as np
import matplotlib.pyplot as plt


def AllanDeviation(dataArr: np.ndarray, fs: float, maxNumM: int=100):
    """Compute the Allan deviation (sigma) of time-series data.

    Algorithm obtained from Mathworks:
    https://www.mathworks.com/help/fusion/ug/inertial-sensor-noise-analysis-using-allan-variance.html

    Args
    ----
        dataArr: 1D data array
        fs: Data sample frequency in Hz
        maxNumM: Number of output points
    
    Returns
    -------
        (taus, allanDev): Tuple of results
        taus (numpy.ndarray): Array of tau values
        allanDev (numpy.ndarray): Array of computed Allan deviations
    """
    ts = 1.0 / fs
    N = len(dataArr)
    Mmax = 2**np.floor(np.log2(N / 2))
    M = np.logspace(np.log10(1), np.log10(Mmax), num=maxNumM)
    M = np.ceil(M)  # Round up to integer
    M = np.unique(M)  # Remove duplicates
    taus = M * ts  # Compute 'cluster durations' tau

    # Compute Allan variance
    allanVar = np.zeros(len(M))
    for i, mi in enumerate(M):
        twoMi = int(2 * mi)
        mi = int(mi)
        allanVar[i] = np.sum(
            (dataArr[twoMi:N] - (2.0 * dataArr[mi:N-mi]) + dataArr[0:N-twoMi])**2
        )
    
    allanVar /= (2.0 * taus**2) * (N - (2.0 * M))
    return (taus, np.sqrt(allanVar))  # Return deviation (dev = sqrt(var))


allData = []
with open("/Users/dsong/Library/CloudStorage/OneDrive-UniversityofIllinois-Urbana/Club Stuff/LRI/FV-Controls/Control/dataTest1072025.txt", "r") as file:
    curList = []
    for line in file:
        line = str(line)
        line = line.replace("\x00", "")
        ar = line.split("|")
        # if "sensor" in line:
        #     allData.append(curList)
        #     curList = []
        if(len(ar) > 5):
            gryos = ar[4]
            gryoList = gryos.split(",")
            newList = []
            for val in gryoList:
                newList.append(float(val) * 57.2958)
            curList.append(newList)
    allData.append(curList)

ts = 0.1
test = allData[0]
test = np.array(test)

xTest = np.cumsum(test[:,0]) * ts
yTest = np.cumsum(test[:,1]) * ts
zTest = np.cumsum(test[:,2]) * ts

tausX, devX = AllanDeviation(xTest, 10, 1000)
tausY, devY = AllanDeviation(yTest, 10, 1000)
tausZ, devZ = AllanDeviation(zTest, 10, 1000)

index = 0
minDist = 10000
for i in range(len(tausX)):
    dist = (tausX[i] -1)**2
    if(dist < minDist):
        minDist = dist
        index = i
print(index)


plt.plot(tausX, devX, label="Wx")
plt.plot(tausY, devY, label="Wy")
plt.plot(tausZ, devZ, label="Wz")
# plt.plot(testingtau, outputs, label= "-0.5 slope")
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.show()

print("The deviation for Wx is " + str(devX[index] * 60))
print("The deviation for Wy is " + str(devY[index] * 60))
print("The deviation for Wz is " + str(devZ[index] * 60))

factor = 5421.68674699
print("The bias instability for Wx is " + str(devX[-1] * factor))
print("The bias instability for Wy is " + str(devY[-1] * factor))
print("The bias instability for Wz is " + str(devZ[-1] * factor))
