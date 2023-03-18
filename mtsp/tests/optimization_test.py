import pytest
import numpy as np
import random

from typing import List

from mtsp.optimization import (
    createRandomPop,
    _evalTotalDistance,
    _binaryTournamentSelect,
    _crossoverHGA,
    _mutateChromosome,
)


@pytest.fixture
def dummyCities() -> List[str]:
    return ["A", "B", "C", "D", "E"]


@pytest.fixture
def dummySalesmen() -> List[str]:
    return ["Man", "Woman"]


@pytest.fixture
def dummyRNG() -> np.random.Generator:
    return np.random.default_rng(123)


@pytest.fixture()
def dummyDistances(dummyCities, dummySalesmen) -> dict:
    distances = dict()
    for start in dummyCities + dummySalesmen:
        distances[start] = dict()
        for destination in dummyCities + dummySalesmen:
            distances[start][destination] = 1.0
    return distances


def test_createRandomPop_happy_path(dummyCities, dummySalesmen, dummyRNG):
    result = createRandomPop(dummyCities, dummySalesmen, dummyRNG)

    assert len(result) == 10
    assert isinstance(result[0][0], str)
    assert isinstance(result[0][-1], int)


def test_createRandomPop_too_many_salesmen(dummyCities, dummyRNG):
    manySalesmen = ["Man{}".format(i) for i in range(len(dummyCities) + 1)]

    with pytest.raises(Exception):
        result = createRandomPop(dummyCities, manySalesmen, dummyRNG)


def test__evalTotalDistance_happy_path(dummyCities, dummySalesmen, dummyDistances):
    dummyChromo = dummyCities.copy()
    dummyChromo.append(2)

    # Explanation: 3 Cities, 2 TSP:
    # TSP1 -> City1, City1 -> City2, TSP2 -> City3
    expectation = len(dummyCities)

    result = _evalTotalDistance(
        dummyChromo, dummySalesmen, len(dummySalesmen), dummyDistances
    )

    assert expectation == result


def test__binaryTournamentSelect_happy_path(dummyCities, dummyRNG):
    tmpCities = dummyCities.copy()
    dummyPop = []
    for _ in range(10):
        random.shuffle(tmpCities)
        dummyPop.append((tmpCities, 1.0))

    selection = _binaryTournamentSelect(dummyPop, dummyRNG)

    assert len(selection) == 2
    assert all([chromo in dummyPop for chromo in selection])


def test__crossoverHGA_happy_path(dummyCities, dummyDistances, dummySalesmen, dummyRNG):
    parentA = dummyCities.copy()
    parentB = dummyCities.copy()
    random.shuffle(parentB)
    parentA.append(1)
    parentB.append(2)

    result = _crossoverHGA(
        parentA, parentB, len(dummySalesmen) - 1, dummyDistances, dummyRNG
    )

    assert len(result) == len(parentA)
    assert isinstance(result[0], str)
    assert isinstance(result[-1], int)


@pytest.mark.parametrize(("mutationType"), ["reverse", "transpose"])
def test__mutateChromosome_happy_path(
    mutationType, dummyCities, dummySalesmen, dummyRNG
):
    dummyChromo = dummyCities.copy()
    dummyChromo += [2]

    result = _mutateChromosome(
        dummyChromo, len(dummySalesmen) - 1, mutationType, dummyRNG
    )

    assert len(dummyChromo) == len(result)
    assert dummyChromo != result
    assert all(city in result for city in dummyChromo[:-1])


def test__mutateChromosome_unknown_mutation_type(dummyCities, dummySalesmen, dummyRNG):
    dummyChromo = dummyCities.copy()
    dummyChromo += [2]
    unknownMutationType = "crazyMutation"

    with pytest.raises(ValueError):
        _ = _mutateChromosome(
            dummyChromo, len(dummySalesmen) - 1, unknownMutationType, dummyRNG
        )
