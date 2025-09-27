# -*- coding: utf-8 -*-
"""
Test the base proposal class.
"""

import logging
import os
import pickle
from unittest.mock import MagicMock, Mock, create_autospec, patch

import numpy as np
import pytest

from nessai.livepoint import numpy_array_to_live_points
from nessai.proposal.base import Proposal


class DummyProposal(Proposal):
    def draw(self, old_param):
        pass


@pytest.fixture
def proposal():
    return create_autospec(Proposal)


def test_init(proposal):
    """Test the init method"""
    model = Mock()
    Proposal.__init__(proposal, model)
    assert proposal.model is model
    assert proposal.training_count == 0
    assert proposal._checked_population is True
    assert proposal.population_time.total_seconds() == 0


def test_init_with_draw():
    """Assert an error is raised if `draw` is not implemented."""
    model = Mock()
    with pytest.raises(TypeError) as excinfo:
        Proposal(model)
    assert "class Proposal with" in str(excinfo.value)


def test_initialised(proposal):
    """Test the initialised property"""
    proposal._initialised = True
    val = Proposal.initialised.__get__(proposal)
    assert val is True


@pytest.mark.parametrize("val", [True, False])
def test_initialised_setter(proposal, val):
    """Test the setter for initialised."""
    Proposal.initialised.__set__(proposal, val)
    assert proposal._initialised is val


def test_initialise(proposal):
    """Test the initialise method"""
    Proposal.initialise(proposal)
    assert proposal.initialised is True


def test_evaluate_likelihoods(proposal):
    """Assert the correct method is called"""
    samples = numpy_array_to_live_points(np.array([[1], [2]]), ["x"])
    proposal.samples = samples
    proposal.model = MagicMock()
    proposal.model.batch_evaluate_log_likelihood = MagicMock(
        return_value=[1, 2]
    )
    Proposal.evaluate_likelihoods(proposal)
    proposal.model.batch_evaluate_log_likelihood.assert_called_once_with(
        samples
    )


def test_draw(proposal):
    """Assert an error is raised."""
    with pytest.raises(NotImplementedError):
        Proposal.draw(proposal, None)


def test_train(proposal, caplog):
    """Test the train method."""
    caplog.set_level(logging.INFO)
    Proposal.train(proposal, 1, plot=True)
    assert "This proposal method cannot be trained" in str(caplog.text)


def test_resume(proposal):
    """Test the resume method."""
    model = Mock()
    Proposal.resume(proposal, model)
    assert proposal.model is model


def test_update_output_directory(proposal):
    """Test the update_output_directory method with a specific directory."""
    test_dir = "/test/output/dir"
    Proposal.update_output_directory(proposal, test_dir)
    assert proposal.output == test_dir


def test_update_output_directory_none(proposal):
    """Test the update_output_directory method with None (should use cwd)."""
    with patch('os.getcwd', return_value='/current/working/dir'):
        Proposal.update_output_directory(proposal, None)
        assert proposal.output == '/current/working/dir'


def test_update_output_directory_existing_attribute(proposal):
    """Test updating when output attribute already exists."""
    proposal.output = "/old/dir"
    test_dir = "/new/dir"
    Proposal.update_output_directory(proposal, test_dir)
    assert proposal.output == test_dir


def test_update_output_directory_no_existing_attribute(proposal):
    """Test updating when output attribute doesn't exist."""
    # Make sure output attribute doesn't exist
    if hasattr(proposal, 'output'):
        delattr(proposal, 'output')
    
    test_dir = "/test/dir"
    Proposal.update_output_directory(proposal, test_dir)
    assert proposal.output == test_dir


def test_getstate(proposal):
    """Test the get state method called by pickle."""
    proposal.model = Mock()
    state = Proposal.__getstate__(proposal)
    assert "model" not in list(state.keys())


@pytest.mark.integration_test
def test_pickling(model):
    """Test pickling the proposal"""
    proposal = DummyProposal(model)
    d = pickle.dumps(proposal)
    out = pickle.loads(d)
    assert hasattr(out, "model") is False


@pytest.mark.integration_test
def test_update_output_directory_integration(model, tmpdir):
    """Test update_output_directory with a real DummyProposal instance."""
    proposal = DummyProposal(model)
    
    # Test setting a specific directory
    test_dir = str(tmpdir.mkdir("test_output"))
    proposal.update_output_directory(test_dir)
    assert proposal.output == test_dir
    
    # Test with None (should use cwd)
    with patch('os.getcwd', return_value='/mock/cwd'):
        proposal.update_output_directory(None)
        assert proposal.output == '/mock/cwd'
