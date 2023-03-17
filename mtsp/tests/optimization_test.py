import pytest
import numpy as np
import random

from typing import List

import sys
import os
sys.path.append(os.getcwd())

from mtsp.optimization import (
    createRandomPop,
    _evalTotalDistance,
    _binaryTournamentSelect
)


@pytest.fixture
def dummyCities() -> List[str]:
    return ["A", "B", "C"]


@pytest.fixture
def dummySalesmen() -> List[str]:
    return ["Man", "Woman"]


@pytest.fixture
def dummyRNG() -> np.random.Generator:
    return np.random.default_rng(123)


@pytest.fixture()
def dummyDistances(dummyCities, dummySalesmen) -> dict:
    distances = dict()
    for start in (dummyCities + dummySalesmen):
        distances[start] = dict()
        for destination in (dummyCities + dummySalesmen):
            distances[start][destination] = 1.0
    return distances


def test_createRandomPop_happy_path(dummyCities, dummySalesmen, dummyRNG):
    result = createRandomPop(dummyCities, dummySalesmen, dummyRNG)

    assert len(result) == 10
    assert isinstance(result[0][0], str)
    assert isinstance(result[0][-1], int)


def test_createRandomPop_too_many_salesmen(dummyCities, dummyRNG):
    manySalesmen = ["Man", "Woman", "Divers", "Other"]

    with pytest.raises(Exception):
        result = createRandomPop(dummyCities, manySalesmen, dummyRNG)


def test__evalTotalDistance_happy_path(dummyCities, dummySalesmen, dummyDistances):
    dummyChromo = dummyCities.copy()
    dummyChromo.append(2)

    # Explanation: 3 Cities, 2 TSP:
    # TSP1 -> City1, City1 -> City2, TSP2 -> City3
    expectation = 3

    result = _evalTotalDistance(dummyChromo, dummySalesmen,
                                len(dummySalesmen), dummyDistances)

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
