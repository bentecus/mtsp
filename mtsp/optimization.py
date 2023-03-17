import numpy as np
import random

from typing import List

from mtsp.mtsp import MTSP

def genAlgo(mtsp: MTSP, pop: list = None, numIter: int = 200, pm: float = 0.05,
            numOffsprings: int = 2) -> (dict, float):
    """
    NSGA-II genetic algorithm implementation inspired by
    https://www.tandfonline.com/doi/full/10.1080/21642583.2019.1674220.

    Parameters:
        mtsp (MTSP): The multiple traveling salesman problem to be solved.
        pop (list): Initial population.
        numIter (int): Number of optimization iterations.
        pm (float): Mutation probability.
        numOffsprings (int): Number of generated offsprings per iteration.

    Returns:
        dict: Assigned cities per salesman.
        float: Resulting total distance.
    """
    if pop is None:
        pop = createRandomPop(mtsp.cityNames, mtsp.tspNames, mtsp.rng)
    popSize = len(pop)

    # Initial fitness evaluation
    pop = [(p, _evalTotalDistance(p)) for p in pop]
    indBest = np.argmax([x[1] for x in self.pop])
    self.fitnessDynamics = [self.pop[indBest][1]]
    self.bestChromo = self.pop[indBest]

    for i in range(self.numIter):
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


def _evalTotalDistance(chromo: list, tspNames: List[str], numTSP: int,
                       distances: dict) -> float:
    """
    Evaluate the total distance of the chromosome.

    Parameters:
        chromo (list): Chromosome under evaluation.
        tspNames (List[str]): Names of traveling salesmen.
        numTSP (int): Number of traveling salesman.
        distances (dict): City-to-City and TSP-to-City distances.

    Returns:
        float: Total distance achieved by chromosome.
    """
    cutPoints = chromo[-(numTSP-1):]
    chromoCities = chromo[:-(numTSP-1)]
    totalDistance = 0

    for tspInd, (cut, tsp) in enumerate(zip(cutPoints, tspNames)):
        if tspInd == 0:
            currentTSPCities = chromoCities[:cut]
        else:
            currentTSPCities = chromoCities[cutPoints[tspInd-1]:cut]
        for counter, city in enumerate(currentTSPCities):
            if counter == 0:
                totalDistance += distances[tsp][city]
            else:
                totalDistance += distances[currentTSPCities[counter-1]][city]

    # Add cities assigned to last TSP
    # TODO: Incorporate following implementation in previous for loop
    lastCities = chromoCities[-(len(chromoCities) - cutPoints[-1]):]
    for counter, city in enumerate(lastCities):
        if counter == 0:
            totalDistance += distances[tspNames[-1]][city]
        else:
            totalDistance += distances[lastCities[counter-1]][city]

    return totalDistance


def _binaryTournamentSelect(pop: list, rng: np.random.Generator) -> list:
    """
    Perform binary tournament selection.

    Parameters:
        pop (list): Current population.
        rng (np.random.Generator): Random number generator.

    Returns:
        list: Selected individuals.
    """
    internalPop = pop.copy()
    selections = []
    for _ in range(2):
        cumFitness = sum([x[1] for x in internalPop])
        probs = [x[1]/cumFitness for x in internalPop]
        cumProb = 0
        selectProb = rng.uniform()
        for counter, prob in enumerate(probs):
            if selectProb >= cumProb and selectProb < cumProb+prob:
                selections.append(internalPop[counter])
                internalPop.pop(counter)
                break
            else:
                cumProb += prob
    return selections


def _crossoverHGA(pa: list, pb: list, numCuts: int, distances: dict, rng: np.random.Generator) -> list:
    """
    Perform crossover operation.

    Parameters:
        pa (list): Parent A.
        pb (list): Parent B.
        numCuts (int): Number of cutting points (numTSP - 1).
        distances (dict): City-to-city and TSP-to-city distances.
        rng (np.random.Generator): Random number generator.

    Returns:
        list: Child.
    """
    endings = [pa[-numCuts:], pb[-numCuts:]]
    pa = pa[:-numCuts]
    pb = pb[:-numCuts]
    length = len(pa)
    k = rng.integers(1, length)
    direction = rng.integers(0, 2)
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
        dx = distances[k][x]
        dy = distances[k][y]
        if dx < dy:
            k = x
        else:
            k = y
        child.append(k)
        length -= 1
    child + endings[rng.integers(2)]
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


def createRandomPop(cityNames: List[str], tspNames: List[str], rng: np.random.Generator,
                    popSize: int = 10) -> List[list]:
    """
    Create random population.

    Parameters:
        cityNames (List[str]): Number of cities.
        tspNames (List[str]): Number of traveling salesmen.
        rng (np.random.Generator): Random number generator.
        popSize (int): Population size.

    Returns:
        List[list]: Population.

    Raises:
        Exception: Too many Salesmen.
    """
    if len(tspNames) > len(cityNames):
        raise Exception("Number of salesmen must not exceed the number of cities.")
    pop = []
    for _ in range(popSize):
        l = cityNames.copy()
        random.shuffle(l)
        cutPoints = rng.integers(1, len(cityNames)-1, len(tspNames)-1)
        for cut in cutPoints:
            l.append(int(cut))
        pop.append(l)
    return pop
