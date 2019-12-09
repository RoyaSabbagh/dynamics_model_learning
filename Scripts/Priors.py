'''

This program contain classes to define different type of priors

Author: roya.sabbaghnovin@utah.edu

'''

# Import
import numpy as np
from Models import *
from Sampler import PYMCSampler


class ModelPrior(object):
    """
    Base class for MDP models.

    """

    def __init__(self):
        super(ModelPrior, self).__init__()

    def __copy__(self):
        raise NotImplementedError()

    def __deepcopy__(self, memo):
        raise NotImplementedError()

    def __eq__(self, other):

        raise NotImplementedError()

    def __ne__(self, other):
        return not self.eq(other)

    def __hash__(self):
        raise NotImplementedError()

    def sampleModel(self):
        raise NotImplementedError()

    def getExpectedModel(self):
        raise NotImplementedError()

    def update(self, state, action, gripper_pos, time_step, newstate):
        raise NotImplementedError()


class Prior_Base(ModelPrior):
    """
    Base class for modeling MDP transitions using arbitrary
    lazy learners.
    """

    def __init__(self):
        self.state_history = []
        self.action_history = []
        self.gripper_pos_history = []
        self.newstate_history = []
        self.time_step_history = []

    def __copy__(self):
        newobj = self.__class__()
        newobj.state_history = self.state_history
        newobj.action_history = self.action_history
        newobj.gripper_pos_history = self.gripper_pos_history
        newobj.time_step_history = self.time_step_history
        newobj.newstate_history = self.newstate_history
        return newobj

    def __deepcopy__(self, memo):
        newobj = self.__class__()
        newobj.state_history = copy.deepcopy(self.state_history, memo)
        newobj.action_history = copy.deepcopy(self.action_history, memo)
        newobj.gripper_pos_history = copy.deepcopy(self.gripper_pos_history, memo)
        newobj.time_step_history = copy.deepcopy(self.time_step_history, memo)
        newobj.newstate_history = copy.deepcopy(self.newstate_history, memo)
        return newobj

    def __eq__(self, other):
        return np.array_equal(self.state_history, other.state_history) and \
               np.array_equal(self.action_history, other.action_history) and \
               np.array_equal(self.gripper_pos_history, other.gripper_pos_history) and \
               np.array_equal(self.time_step_history, other.time_step_history) and \
               np.array_equal(self.newstate_history, other.newstate_history)

    def __hash__(self):
        return hash((str(self.state_history),
                     str(self.action_history),
                     str(self.gripper_pos_history),
                     str(self.newstate_history),
                     str(self.time_step_history)))

    def train(self):
        '''
        Trains on current data, and toggles stale bit to False
        '''
        raise NotImplementedError()

    def getExpectedModel(self):
        '''
        Returns the MAP sample from the model posterior
        '''
        raise NotImplementedError()

    def sampleModel(self):
        '''
        Returns a sample from the model posterior
        '''
        raise NotImplementedError()

    def update(self, state, action, gripper_pos, time_step, newstate):
        self.state_history.append(copy.deepcopy(state))
        self.action_history.append(copy.deepcopy(action))
        self.gripper_pos_history.append(copy.deepcopy(gripper_pos))
        self.time_step_history.append(copy.deepcopy(time_step))
        self.newstate_history.append(copy.deepcopy(newstate))

class RegressionPrior(Prior_Base):
    """
    Base class for modeling MDP transitions using independent regression
    models for each output dimension.
    """

    def __init__(self, regressionClass, reg_args=None):
        super(RegressionPrior, self).__init__()
        self.regressionClass = regressionClass
        self.reg_args = reg_args
        self.model = None

    def __copy__(self):
        newobj = super(RegressionPrior, self).__copy__(self.nominalActions)
        newobj.regressionClass = self.regressionClass
        return newobj

    def __deepcopy__(self, memo):
        newobj = super(RegressionPrior, self).__deepcopy__(self.nominalActions)
        newobj.regressionClass = self.regressionClass
        return newobj

    def train(self):
        self.model = RegressionModel(self.regressionClass)
        self.model.train(self.state_history,
                         self.action_history,
                         self.gripper_pos_history,
                         self.time_step_history,
                         self.newstate_history)
        return self.model

class BayesianPrior(Prior_Base):
    """
    Implements a generative physics model for rigid body dynamics.
    The learning method is based on MCMC inference using the
    PYMCSampler class.

    """

    def __init__(self, dynamic_model, max_hist=50, sigma_retrain_thresh=0.05,
                 niter=2e4, burn=1e0, thin=10, tune_interval=1000):
        """
        Initializes a physics prior for modeling the dynamics
        of a single object.
        :param max_hist: The maximum number of training points to hold
        :param sigma_retrain_thresh: If model.sigma is below sigma_retrain_thresh, don't bother re-training
        :param niter: Number of MCMC iterations
        :param burn: Number of MCMC burn-in steps
        :param thin: Interval for acquiring MCMC samples
        :param tune_interval: Interval for tuning (pymc.AdaptiveMetropolis)
        """
        super(BayesianPrior, self).__init__()
        self.model = BayesianModel()
        self.max_hist = max_hist
        self.niter = niter
        self.burn = burn
        self.thin = thin
	self.dynamic_model = dynamic_model
        self.tune_interval = tune_interval
        self.sigma_retrain_thresh = sigma_retrain_thresh

    def update(self, *args, **kwargs):
        if len(self.state_history) < self.max_hist:
            return Prior_Base.update(self, *args, **kwargs)

    def train(self):
        states = []
        actions = []
        gripper_poses = []
        time_steps = []
        newstates = []
        for i in xrange(len(self.state_history)):
            state = self.state_history[i]
            newstate = self.newstate_history[i]
            gripper_pos = self.gripper_pos_history[i]
            action = self.action_history[i]
            time_step = self.time_step_history[i]

            states.append(state)
            actions.append(action)
            gripper_poses.append(gripper_pos)
            time_steps.append(time_step)
            newstates.append(newstate)
        self.sampler = PYMCSampler(states, actions, gripper_poses, time_steps, newstates, self.dynamic_model)
        (inertia_mu, friction_mu, inertia_std, friction_std) = self.sampler.run(niter=self.niter,
                                                     burn=self.burn,
                                                     thin=self.thin,
                                                     tune_interval=self.tune_interval)
        self.model.params = [inertia_mu, friction_mu]
	print("inertia_mu, friction_mu, inertia_std, friction_std")
        print(inertia_mu, friction_mu, inertia_std, friction_std)
        return self.model




