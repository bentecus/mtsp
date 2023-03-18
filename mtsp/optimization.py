import numpy as np
import random

from typing import List

from mtsp.mtsp import MTSP


def genAlgo(
    mtsp: MTSP,
    pop: list = None,
    numIter: int = 200,
    pm: float = 0.05,
    numOffsprings: int = 2,
) -> (dict, float):
    """
    One-objective NSGA-II genetic algorithm implementation inspired by
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
    pop = [
        (p, _evalTotalDistance(p, mtsp.tspNames, mtsp.numTSP, mtsp.distances))
        for p in pop
    ]
    indBest = np.argmax([x[1] for x in pop])
    fitnessDynamics = [pop[indBest][1]]
    bestChromo = pop[indBest]

    for i in range(numIter):
        offsprings = []

        # Tournament selection and crossover
        for o in range(numOffsprings):
            parentOne, parentTwo = _binaryTournamentSelect(pop, mtsp.rng)
            offsprings.append(
                _crossoverHGA(
                    parentOne[0],
                    parentTwo[0],
                    mtsp.numTSP - 1,
                    mtsp.distances,
                    mtsp.rng,
                )
            )

        # Offspring mutation and evaluation
        for offspring in offsprings:
            if mtsp.rng.uniform() <= pm:
                mutationType = "reverse" if mtsp.rng.uniform() <= 0.5 else "transpose"
                offspring = _mutateChromosome(
                    offspring, mtsp.numTSP - 1, mutationType, mtsp.rng
                )
            pop.append(
                (
                    offspring,
                    _evalTotalDistance(
                        offspring, mtsp.tspNames, mtsp.numTSP, mtsp.distances
                    ),
                )
            )

        # Sort population by fitness and select new population
        sortedPop = sorted(pop, key=lambda x: x[1])
        pop = sortedPop[:popSize]
        bestChromo = pop[0]
        fitnessDynamics.append(pop[0][1])

    return pop, bestChromo, fitnessDynamics


def _evalTotalDistance(
    chromo: list, tspNames: List[str], numTSP: int, distances: dict
) -> float:
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
    cutPoints = chromo[-(numTSP - 1) :]
    chromoCities = chromo[: -(numTSP - 1)]
    totalDistance = 0

    for tspInd, (cut, tsp) in enumerate(zip(cutPoints, tspNames)):
        if tspInd == 0:
            currentTSPCities = chromoCities[:cut]
        else:
            currentTSPCities = chromoCities[cutPoints[tspInd - 1] : cut]
        for counter, city in enumerate(currentTSPCities):
            if counter == 0:
                totalDistance += distances[tsp][city]
            else:
                totalDistance += distances[currentTSPCities[counter - 1]][city]

    # Add cities assigned to last TSP
    # TODO: Incorporate following implementation in previous for loop
    lastCities = chromoCities[-(len(chromoCities) - cutPoints[-1]) :]
    for counter, city in enumerate(lastCities):
        if counter == 0:
            totalDistance += distances[tspNames[-1]][city]
        else:
            totalDistance += distances[lastCities[counter - 1]][city]

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
        probs = [x[1] / cumFitness for x in internalPop]
        cumProb = 0
        selectProb = rng.uniform()
        for counter, prob in enumerate(probs):
            if selectProb >= cumProb and selectProb < cumProb + prob:
                selections.append(internalPop[counter])
                internalPop.pop(counter)
                break
            else:
                cumProb += prob
    return selections


def _crossoverHGA(
    pa: list, pb: list, numCuts: int, distances: dict, rng: np.random.Generator
) -> list:
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
    k = pa[rng.integers(1, length)]
    direction = rng.integers(0, 2)
    child = [k]

    while length > 1:
        if direction == 0:
            x = pa[pa.index(k) + 1] if pa.index(k) + 1 != len(pa) else pa[0]
            y = pb[pb.index(k) + 1] if pb.index(k) + 1 != len(pb) else pb[0]
        else:
            x = pa[pa.index(k) - 1]
            y = pb[pb.index(k) - 1]
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
    child += endings[rng.integers(2)]
    return child


def _mutateChromosome(
    chromo: list, numCuts: int, mutType: str, rng: np.random.Generator
) -> list:
    """
    Mutate chromosome. Equal probability for reverse or transpose fragment mutation.

    Parameters:
        chromo (list): Original chromosome to be mutated.
        numCuts (int): Number of cutting points (numTSP - 1).
        mutType (str): Mutation type. Possible options: reverse, transpose
        rng (np.random.Generator): Random number generator.

    Returns:
        list: Mutated chromosome.
    """
    mutated = chromo[:-numCuts].copy()
    cutOne = rng.integers(
        0, len(chromo) - numCuts - 1
    )  # First cut can't be last entry.
    cutTwo = rng.integers(cutOne, len(chromo) - numCuts)

    if mutType == "reverse":
        # Reverse mutation operation
        revChromo = chromo[cutOne:cutTwo].copy()
        revChromo.reverse()
        mutated = mutated[:cutOne] + revChromo + mutated[cutTwo:]
    elif mutType == "transpose":
        # Transpose fragment mutation operation
        mutated = mutated[cutOne:cutTwo] + mutated[:cutOne] + mutated[cutTwo:]
    else:
        raise ValueError("Unknown mutation type: {}".format(mutType))
    mutated += sorted(list(rng.integers(1, len(chromo) - numCuts, numCuts)))

    return mutated


def createRandomPop(
    cityNames: List[str],
    tspNames: List[str],
    rng: np.random.Generator,
    popSize: int = 10,
) -> List[list]:
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
        cutPoints = rng.integers(1, len(cityNames) - 1, len(tspNames) - 1)
        for cut in cutPoints:
            l.append(int(cut))
        pop.append(l)
    return pop
