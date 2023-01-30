#!/usr/bin/env python3

import numpy as np
import time
import random
import matplotlib.pyplot as plt
import rospy
from pathcost_calculation.msg import PathCosts
from pathcost_calculation.msg import PathCost
from pathcost_calculation.msg import DrivePlan

class MTSP():
    def __init__(self, cityCoordinates, tspCoordinates, seed=12345):
        self.cityCoordinates = cityCoordinates
        self.numCities = len(cityCoordinates)
        self.tspCoordinates = tspCoordinates
        self.numTSP = len(tspCoordinates)
        self.rng = np.random.default_rng(seed)
   
        self.distances = {}
        for pc in data.costs:
            self.distances[pc.id] = pc.cost
        for pc in self.robotDistances:
            self.distances[pc.id] = pc.cost
        for pc in self.tagDistances:
            self.distances[pc.id] = pc.cost
        

    def optimize(self, pop, num_iter, pm, numOffsprings=2):
        print("Solving MTSP.")
        self.popSize = len(pop)
        self.num_iter = num_iter
        self.pm = pm
        self.numOffsprings = numOffsprings

        #initial fitness evaluation
        self.pop = [(p, self._evalTotalDistance(p)) for p in pop]
        indBest = np.argmax([x[1] for x in self.pop])
        self.fitnessDynamics = [self.pop[indBest][1]]
        self.bestChromo = self.pop[indBest]

        for i in range(self.num_iter):
            self.offsprings = []

            #tournament selection and crossover
            for o in range(self.numOffsprings):
                parentOne, parentTwo = self._binaryTournamentSelect()
                self.offsprings.append(self._crossoverHGA(parentOne[0], parentTwo[0]))
            
            #offspring mutation and evaluation
            for offspring in self.offsprings:
                if self.rng.uniform() <= self.pm:
                    if self.rng.uniform() <= 0.5:
                        offspring = self._mutateReverse(offspring)
                    else:
                        offspring = self._mutateTransposeFragments(offspring)
                self.pop.append((offspring, self._evalTotalDistance(offspring)))
            
            #sort population by fitness and select new population
            sortedPop = sorted(self.pop, key=lambda x: x[1])
            self.pop = sortedPop[:self.popSize]
            self.bestChromo = self.pop[0]
            self.fitnessDynamics.append(self.pop[0][1])
            #print("Iteration %i done." % (i+1))
        
        return self.pop, self.bestChromo, self.fitnessDynamics
        

    def _evalTotalDistance(self, chromo):
        cutPoint = chromo[-1]
        tagsRobOne = chromo[:cutPoint].copy()
        tagsRobTwo = chromo[cutPoint:-1].copy()
        totalDistance = 0
        for ind, robotTags in zip(['r1d1', 'r2d2'], [tagsRobOne, tagsRobTwo]):
            for counter, tag in enumerate(robotTags):
                if counter == 0:
                    totalDistance += self._getDistance(ind, tag)
                else:
                    totalDistance += self._getDistance(robotTags[counter-1], tag)
        return totalDistance

    def _getDistance(self, a, b):
        if type(a) != str:
            return self.distances['%i_%i' % (min(a,b),max(a,b))]
        else:
            return self.distances['%s_%i' % (a,b)]

    def _binaryTournamentSelect(self):
        internalPop = self.pop.copy()
        selections = []
        for _ in range(2):
            cumFitness = sum([x[1] for x in internalPop])
            probs = [x[1]/cumFitness for x in internalPop]
            currentProb = 0
            selectProb = self.rng.uniform()
            for counter, prob in enumerate(probs):
                if selectProb >= currentProb and selectProb < currentProb+prob:
                    selections.append(internalPop[counter])
                    internalPop.pop(counter)
                    break
                else:
                    currentProb += prob
        return selections

    def _crossoverHGA(self, pa, pb):
        endings = [pa[-1], pb[-1]]
        pa = pa[:-1]
        pb = pb[:-1]
        length = len(pa)
        k = self.rng.integers(1, length)
        direction = self.rng.integers(0, 2)
        child = [k]

        while length > 1:
            if direction == 0:
                x = pa[pa.index(k)+1] if pa.index(k)+1 != len(pa) else pa[0] 
                y = pb[pb.index(k)+1] if pb.index(k)+1 != len(pb) else pb[0]
            else:
                x = pa[pa.index(k)-1]
                y = pb[pb.index(k)-1]
            pa.remove(k)
            pb.remove(k)
            dx = self._getDistance(k, x)
            dy = self._getDistance(k, y)
            if dx < dy:
                k = x
            else:
                k = y
            child.append(k)
            length -= 1
        child.append(endings[self.rng.integers(2)])
        return child

    def _mutateReverse(self, chromo):
        cutOne = self.rng.integers(0, len(chromo)-self.numTSP-1-1)
        cutTwo = self.rng.integers(cutOne, len(chromo)-self.numTSP-1)

        revChromo = chromo[cutOne:cutTwo].copy()
        revChromo.reverse()
        child = chromo[:cutOne] + revChromo + chromo[cutTwo:]
        child[-self.numTSP+1:] = self.rng.integers(1, len(chromo)-self.numTSP-1, self.numTSP-1)

        return child
    
    def _mutateTransposeFragments(self, chromo):
        cutOne = self.rng.integers(0, len(chromo)-self.numTSP-1-1)
        cutTwo = self.rng.integers(cutOne, len(chromo)-self.numTSP-1)

        child = chromo[cutOne:cutTwo] + chromo[:cutOne] + chromo[cutTwo:]
        child[-self.numTSP+1:] = self.rng.integers(1, len(chromo)-self.numTSP-1, self.numTSP-1)

        return child

    def createRandomPop(self, popSize, rng):
        '''
        create population
        '''
        pop = []
        for _ in range(popSize):
            l = list(range(numTags))
            random.shuffle(l)
            l.append(rng.integers(1, numTags-1))
            pop.append(l)
        return pop

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