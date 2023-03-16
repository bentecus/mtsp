import pytest
import numpy as np

from typing import List

from mtsp.optimization import (
    createRandomPop,
    _evalTotalDistance
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
def dummyDistances(dummyCities, dummySalesmen, dummyRNG) -> dict:
    distances = dict()
    pass


def test_createRandomPop_happy_path(dummyCities, dummySalesmen, dummyRNG):
    result = createRandomPop(dummyCities, dummySalesmen, dummyRNG)

    assert len(result) == 10
    assert isinstance(result[0][0], str)
    assert isinstance(result[0][-1], int)


def test_createRandomPop_too_many_salesmen(dummyCities, dummyRNG):
    manySalesmen = ["Man", "Woman", "Divers", "Other"]

    with pytest.raises(Exception):
        result = createRandomPop(dummyCities, manySalesmen, dummyRNG)
