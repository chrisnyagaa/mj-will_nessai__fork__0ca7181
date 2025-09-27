# -*- coding: utf-8 -*-
"""
Base object for all proposal classes.
"""

import datetime
import logging
import os
from abc import ABC, abstractmethod

import numpy as np

logger = logging.getLogger(__name__)


class Proposal(ABC):
    """
    Base proposal object

    Parameters
    ----------
    model: obj
        User-defined model
    """

    def __init__(self, model):
        self.model = model
        self.populated = True
        self._initialised = False
        self.training_count = 0
        self.population_acceptance = None
        self.population_time = datetime.timedelta()
        self.r = np.nan
        self.samples = []
        self.indices = []
        self._checked_population = True
        self.output = None

    @property
    def initialised(self):
        """Boolean that indicates if the proposal is initialised or not."""
        return self._initialised

    @initialised.setter
    def initialised(self, boolean):
        """Setter for initialised"""
        if boolean:
            self._initialised = boolean
        else:
            self._initialised = boolean

    def initialise(self):
        """
        Initialise the proposal
        """
        self.initialised = True

    def evaluate_likelihoods(self):
        """Evaluate the likelihoods for the pool of live points."""
        self.samples["logL"] = self.model.batch_evaluate_log_likelihood(
            self.samples
        )

    @abstractmethod
    def draw(self, old_param):
        """
        New a new point given the old point
        """
        raise NotImplementedError

    def train(self, x, **kwargs):
        """
        Train the proposal method

        Parameters
        ----------
        x: array_like
            Array of live points to use for training
        kwargs:
            Any of keyword arguments
        """
        logger.error("This proposal method cannot be trained")

    def resume(self, model):
        """
        Resume the proposal with the model
        """
        self.model = model

    def update_output_directory(self, output):
        """
        Update the output directory for saving results.

        Parameters
        ----------
        output : str
            Path to the new output directory. If None, uses the current 
            working directory. The directory will be created if it doesn't exist.
        """
        if output is None:
            output = os.getcwd()
        
        if not os.path.exists(output):
            os.makedirs(output, exist_ok=True)
            logger.debug(f"Created output directory: {output}")
        
        self.output = output
        logger.debug(f"Updated output directory to: {output}")

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["model"]
        return state
