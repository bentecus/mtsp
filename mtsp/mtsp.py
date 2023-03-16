import numpy as np
import time
import random

from typing import Callable, Union

from mtsp.distances import euclidean

class MTSP():
    def __init__(self, cityCoordinates: dict, tspCoordinates: dict, 
                distanceFunc: Union[
                    Callable[[np.ndarray, np.ndarray], float], str] = "euclidean",
                seed: int = 12345):
        """
        Solving Multiple Traveling Salesman Problem (MTSP)

        Parameters:
            cityCoordinates (dict): x and y coordinates of all salesman cities.
                                    Style: {"Name": (x, y)}
            tspCoordinates (dict): x and y coordinates of all traveling salesman.
            distanceFunc (Callable[[np.ndarray, np.ndarray], float]|str):
                                    Function or method to calculate distances.
                                    Pre-implemented options: 
                                        - euclidean
            seed (int): Random seed. 
        """
        self.cityCoordinates = cityCoordinates
        self.tspCoordinates = tspCoordinates
        self.distanceFunc = distanceFunc
        
        self.numCities = len(cityCoordinates)
        self.cityNames = list(cityCoordinates.keys())
        self.numTSP = len(tspCoordinates)
        self.tspNames = list(tspCoordinates.keys())
        self.rng = np.random.default_rng(seed)
   
        self.distances = self.calcDistances(self.cityCoordinates, self.tspCoordinates, 
                                            self.distanceFunc)


    def calcDistances(self, cityCoordinates: dict, tspCoordinates: dict,
                      distanceFunc: Union[
                        Callable[[np.ndarray, np.ndarray], float], str]
                     ) -> dict:
        """
        Calculate city-to-city and tsp-to-city distances.

        Parameters: 
            cityCoordinates (dict): x and y coordinates of all salesman cities.
            tspCoordinates (dict): x and y coordinates of all traveling salesman.
            distanceFunc (Callable[[np.ndarray, np.ndarray], float] | str):
                            Function or method to calculate distances.

        Returns:
            dict: City-to-city and tsp-to-city distances.
        
        Raises: 
            ValueError: Unsupported distance function.
        """
        if isinstance(distanceFunc, str):
            if distanceFunc == "euclidean":
                distanceFunc = euclidean
            else:
                raise ValueError(
                    "The provided distance function {} is not supported.".format(
                    distanceFunc))
        
        allCoordinates = cityCoordinates.copy()
        allCoordinates.update(tspCoordinates)
        distances = dict()

        for name in (self.cityNames + self.tspNames):
            distances[name] = dict()
            for k, v in cityCoordinates.items():
                distances[name][k] = distanceFunc(allCoordinates[name], v)
        
        return distances


if __name__ == '__main__':
    rospy.init_node("MTSP", anonymous=True)
    numTags = rospy.get_param('/tagCount')
    pubRob1 = rospy.Publisher("/r1d1DrivePlan", DrivePlan, queue_size=10)
    pubRob2 = rospy.Publisher("/r2d2DrivePlan", DrivePlan, queue_size=10)

    mtsp = MTSP(numCities=numTags, numTSP=2, seed=12345)

    rng = np.random.default_rng(12345)
    popSize = 10
    pop = mtsp.createRandomPop(popSize, rng)
    num_iter = 200
    pm = 0.05
    resultPop, bestChromo, fitnessDynamics = mtsp.optimize(pop, num_iter, pm)

    assignedTags = bestChromo[0]
    cutPoint = assignedTags[-1]
    tagsRob1 = assignedTags[:cutPoint]
    tpRob1 = DrivePlan()
    tpRob1.drivePlan.extend(list(tagsRob1))
    tagsRob2 = assignedTags[cutPoint:-1]
    tpRob2 = DrivePlan()
    tpRob2.drivePlan.extend(list(tagsRob2))

    rospy.loginfo("Publishing drive plans for robots.")
    pubRob1.publish(tpRob1)
    pubRob2.publish(tpRob2)
    rospy.loginfo("Published.")