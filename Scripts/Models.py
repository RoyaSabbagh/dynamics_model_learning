'''

This program to define various classes for different type of models.

Author: roya.sabbaghnovin@utah.edu

'''

# Import
import numpy as np
import copy
from regression import RegressionBase
from numpy.linalg import inv


class MDPModel(object):
    """
    An abstract type for MDP models. This class represents transition model in
    the MDP and supports sampling new state from the transition probability
    given the present state and action.

    Transition model T(s, a, s') is a probability of entering new state s' given
    present state s and the action s.
    """

    def sampleTransition(self, state, action, *args, **kwargs):
        """
        Given present state s and action a, samples new state from
        transition model.

        """
        raise NotImplementedError("can not sample from abstract MDPModel")


class RegressionModel(MDPModel):
    '''
    An MDPModel object for MDPs with continuous state and action
    spaces, where transition probabilities are modeled by
    a user-provided regression function.
    This class can model the state-space dynamics of ONE object at
    a time.
    '''

    def __init__(self, regressionClass=None):
        self.regressionClass = regressionClass
        self.model = []  # list of low-level regression models

    def train(self, states, actions, newstates):
        '''
        trains a regression model
        :param states: An array representing states
        :param actions: An array representing actions
        :param newstates: An array representing the new states
        '''
        # data lengths should match
        assert (len(states) == len(actions) == len(newstates))
        # Horizontally stack action and state regressors and fit a model
        X = np.concatenate((states,actions),axis=1)  # note: looses state/action types here
        y = newstates
        self.model = self.regressionClass()
        self.model.fit(X, y)

    def predict(self, state, action):
        '''
        Samples a new state for the given (state,action) pair, one
        parameter at a time, according to the internal regression model
        :param state: A numpy array of states
        :param action: A numpy array of actions
        '''

        X = state.__class__(np.concatenate((np.atleast_2d(intercept),
                                             np.atleast_2d(state),
                                             np.atleast_2d(action)),
                                            axis=1), state.thresh)  # 11/18/13 will break if not ArrayElement
        try:
            newstates = self.model.predict(X)
        except:
            # just to be safe
            return state  # or return None if we implement KWiK planning
        return newstates

class BayesianModel(MDPModel):
    '''
        Implements a generative physics model for rigid body dynamics.
        '''

    def __init__(self, prior=None):
        '''
        Initializes a physics model for modeling the dynamics
        of a single object.  Also includes a mechanism for
        specifying (marginal) distributions over physical parameters
        for resampling.
        '''
        super(BayesianModel, self).__init__()
        self.prior = prior
        self.params = {}
        self.sigma = 0  # stdev of additive Gaussian noise

    def predict(self, X, U, r_F, dT, Rho, Omega, dynamic_model):
	Y_model = []
	error_d = []
	for j in range(len(X)):
		Xnew = [X[j][0]]
		error_x = []
		error_y = []
		error_phi = []
		error_xdot = []
		error_ydot = []
		error_phidot = []
		if dynamic_model == "friction_only":
			for i in range(len(X[j])-1):
				Xold = Xnew[i]
				R = [[np.cos(Xnew[i][2]), -np.sin(Xnew[i][2])], [np.sin(Xnew[i][2]), np.cos(Xnew[i][2])]]
				R2 = [[np.cos(Omega[3]), -np.sin(Omega[3])], [np.sin(Omega[3]), np.cos(Omega[3])]]
				r_U = [r_F[j][i][0] - Rho[0], r_F[j][i][1] - Rho[1]]
				Tau = np.cross(r_U, U[j][i])
				theta_dot = Tau/Omega[2]
				vel1 = -np.dot(np.dot([[0, -theta_dot],[theta_dot, 0]],R),[Rho[0], Rho[1]]) 
				vel2 = -np.dot(np.dot(np.dot(R,[[float(1)/Omega[0], 0], [0, float(1)/Omega[1]]]),inv(R)),U[j][i])			
				[x_dot,y_dot] = [vel1[0]+vel2[0], vel1[1]+vel2[1]]
				Xnew.append([Xnew[i][0]+dT[j][i]*x_dot, Xnew[i][1]+dT[j][i]*y_dot, Xnew[i][2]+dT[j][i]*theta_dot, x_dot, y_dot, theta_dot])
				aux = [Xnew[i][k] - Xold[k] for k in range(6)]
				dx = [X[j][i][k]-X[j][i+1][k] for k in range(6)]
				error_x.append(abs(aux[0]-dx[0]))
				error_y.append(abs(aux[1]-dx[1]))
				error_phi.append(abs(aux[2]-dx[2]))
				error_xdot.append(abs(aux[3]-dx[3]))
				error_ydot.append(abs(aux[4]-dx[4]))
				error_phidot.append(abs(aux[5]-dx[5])) 
		if dynamic_model == "point_mass_on_a_wheel":
			for i in range(len(X[j])-1):
				R = [[np.cos(Xnew[i][2]), -np.sin(Xnew[i][2])], [np.sin(Xnew[i][2]), np.cos(Xnew[i][2])]]
				R2 = [[np.cos(Omega[2]), -np.sin(Omega[2])], [np.sin(Omega[2]), np.cos(Omega[2])]]
				V_wheel = np.dot(inv(R2),np.dot(inv(R),([Xnew[i][3], Xnew[i][4]]+ np.dot([[0, -Xnew[i][5]],[Xnew[i][5], 0]], np.dot(R,[Rho[2],Rho[3]])))))
				F_wheel = -np.dot(R,np.dot(R2,np.dot([[Omega[0], 0], [0, Omega[1]]], V_wheel)))
				F_T = U[j][i] + F_wheel
				r_U = [r_F[j][i][0] - Rho[2], r_F[j][i][1] - Rho[3]]
				Tau = np.cross(r_U, U[j][i])
				aux = [dT[j][i]*x for x in [Xnew[i][3], Xnew[i][4], Xnew[i][5], F_T[0]/Rho[0], F_T[1]/Rho[0], Tau/Rho[1]]]
				dx = [X[j][i][k]-X[j][i+1][k] for k in range(6)]
				error_x.append(abs(aux[0]-dx[0]))
				error_y.append(abs(aux[1]-dx[1]))
				error_phi.append(abs(aux[2]-dx[2]))
				error_xdot.append(abs(aux[3]-dx[3]))
				error_ydot.append(abs(aux[4]-dx[4]))
				error_phidot.append(abs(aux[5]-dx[5])) 
				Xnew.append([Xnew[i][k] +aux[k] for k in range(6)])
		elif dynamic_model == "2_wheel":
			for i in range(len(X[j])-1):
				R = [[np.cos(Xnew[i][2]), -np.sin(Xnew[i][2])], [np.sin(Xnew[i][2]), np.cos(Xnew[i][2])]]
				V_wheel_1 = np.dot(inv(R),([Xnew[i][3], Xnew[i][4]]+ np.dot([[0, -Xnew[i][5]],[Xnew[i][5], 0]], np.dot(R,[Omega[4]+Omega[6],Omega[5]]))))
				V_wheel_2 = np.dot(inv(R),([Xnew[i][3], Xnew[i][4]]+ np.dot([[0, -Xnew[i][5]],[Xnew[i][5], 0]], np.dot(R,[Omega[4]-Omega[6],Omega[5]]))))
				F_wheel_1 = -np.dot(R,np.dot([[Omega[0], 0], [0, Omega[1]]], V_wheel_1))
				F_wheel_2 = -np.dot(R,np.dot([[Omega[2], 0], [0, Omega[3]]], V_wheel_2))
				F_T = U[j][i] + F_wheel_1 + F_wheel_2
				r_U = [r_F[j][i][0] - Rho[2], r_F[j][i][1] - Rho[3]]
				r_wheel_1 = [Omega[4]+Omega[6] - Rho[2], Omega[5] - Rho[3]]
				r_wheel_2 = [Omega[4]-Omega[6] - Rho[2], Omega[5] - Rho[3]]
				Tau = np.cross(r_U, U[j][i]) + np.cross(r_wheel_1, F_wheel_1) + np.cross(r_wheel_2, F_wheel_2)
				aux = [dT[j][i]*x for x in [Xnew[i][3], Xnew[i][4], Xnew[i][5], F_T[0]/Rho[0], F_T[1]/Rho[0], Tau/Rho[1]]]
				dx = [X[j][i][k]-X[j][i+1][k] for k in range(6)]
				error_x.append(abs(aux[0]-dx[0]))
				error_y.append(abs(aux[1]-dx[1]))
				error_phi.append(abs(aux[2]-dx[2]))
				error_xdot.append(abs(aux[3]-dx[3]))
				error_ydot.append(abs(aux[4]-dx[4]))
				error_phidot.append(abs(aux[5]-dx[5])) 
				Xnew.append([Xnew[i][k] +aux[k] for k in range(6)])

		error_d.append([error_x,error_y,error_phi,error_xdot,error_ydot,error_phidot])
		Y_model.append(Xnew)
	Y_model_fb = []
	error_d_fb = []
	for j in range(len(X)):
		Xnew = [X[j][0]]
		error_x = []
		error_y = []
		error_phi = []
		error_xdot = []
		error_ydot = []
		error_phidot = []
		if dynamic_model == "friction_only":
			for i in range(len(X[j])-1):
				if i%20 == 0:
					Xnew.append(X[j][i])
					error_x.append(0)
					error_y.append(0)
					error_phi.append(0)
					error_xdot.append(0)
					error_ydot.append(0)
					error_phidot.append(0) 
				else:	
					Xold = Xnew[i]
					R = [[np.cos(Xnew[i][2]), -np.sin(Xnew[i][2])], [np.sin(Xnew[i][2]), np.cos(Xnew[i][2])]]
					R2 = [[np.cos(Omega[3]), -np.sin(Omega[3])], [np.sin(Omega[3]), np.cos(Omega[3])]]
					r_U = [r_F[j][i][0] - Rho[0], r_F[j][i][1] - Rho[1]]
					Tau = np.cross(r_U, U[j][i])
					theta_dot = Tau/Omega[2]
					vel1 = -np.dot(np.dot([[0, -theta_dot],[theta_dot, 0]],R),[Rho[0],Rho[1]]) 
					vel2 = -np.dot(np.dot(np.dot(R,[[float(1)/Omega[0], 0], [0, float(1)/Omega[1]]]),inv(R)),U[j][i])			
					[x_dot,y_dot] = [vel1[0]+vel2[0], vel1[1]+vel2[1]]
					Xnew.append([Xnew[i][0]+dT[j][i]*x_dot, Xnew[i][1]+dT[j][i]*y_dot, Xnew[i][2]+dT[j][i]*theta_dot, x_dot, y_dot, theta_dot])
					aux = [Xnew[i][k] - Xold[k] for k in range(6)]
					dx = [X[j][i][k]-X[j][i+1][k] for k in range(6)]
					error_x.append(abs(aux[0]-dx[0]))
					error_y.append(abs(aux[1]-dx[1]))
					error_phi.append(abs(aux[2]-dx[2]))
					error_xdot.append(abs(aux[3]-dx[3]))
					error_ydot.append(abs(aux[4]-dx[4]))
					error_phidot.append(abs(aux[5]-dx[5])) 
		if dynamic_model == "point_mass_on_a_wheel":
			for i in range(len(X[j])-1):
				if i%20 == 0:
					Xnew.append(X[j][i])
					error_x.append(0)
					error_y.append(0)
					error_phi.append(0)
					error_xdot.append(0)
					error_ydot.append(0)
					error_phidot.append(0) 
				else:
					R = [[np.cos(Xnew[i][2]), -np.sin(Xnew[i][2])], [np.sin(Xnew[i][2]), np.cos(Xnew[i][2])]]
					R2 = [[np.cos(Omega[2]), -np.sin(Omega[2])], [np.sin(Omega[2]), np.cos(Omega[2])]]
					V_wheel = np.dot(inv(R2),np.dot(inv(R),([Xnew[i][3], Xnew[i][4]]+ np.dot([[0, -Xnew[i][5]],[Xnew[i][5], 0]], np.dot(R,[Rho[2],Rho[3]])))))
					F_wheel = -np.dot(R,np.dot(R2,np.dot([[Omega[0], 0], [0, Omega[1]]], V_wheel)))
					F_T = U[j][i] + F_wheel
					r_U = [r_F[j][i][0] - Rho[2], r_F[j][i][1] - Rho[3]]
					Tau = np.cross(r_U, U[j][i])
					aux = [dT[j][i]*x for x in [Xnew[i][3], Xnew[i][4], Xnew[i][5], F_T[0]/Rho[0], F_T[1]/Rho[0], Tau/Rho[1]]]
					dx = [X[j][i][k]-X[j][i+1][k] for k in range(6)]
					error_x.append(abs(aux[0]-dx[0]))
					error_y.append(abs(aux[1]-dx[1]))
					error_phi.append(abs(aux[2]-dx[2]))
					error_xdot.append(abs(aux[3]-dx[3]))
					error_ydot.append(abs(aux[4]-dx[4]))
					error_phidot.append(abs(aux[5]-dx[5])) 
					Xnew.append([Xnew[i][k] +aux[k] for k in range(6)])
		elif dynamic_model == "2_wheel":
			for i in range(len(X[j])-1):
				if i%20 == 0:
					Xnew.append(X[j][i])
					error_x.append(0)
					error_y.append(0)
					error_phi.append(0)
					error_xdot.append(0)
					error_ydot.append(0)
					error_phidot.append(0) 
				else:

					R = [[np.cos(Xnew[i][2]), -np.sin(Xnew[i][2])], [np.sin(Xnew[i][2]), np.cos(Xnew[i][2])]]
					V_wheel_1 = np.dot(inv(R),([Xnew[i][3], Xnew[i][4]]+ np.dot([[0, -Xnew[i][5]],[Xnew[i][5], 0]], np.dot(R,[Omega[4]+Omega[6],Omega[5]]))))
					V_wheel_2 = np.dot(inv(R),([Xnew[i][3], Xnew[i][4]]+ np.dot([[0, -Xnew[i][5]],[Xnew[i][5], 0]], np.dot(R,[Omega[4]-Omega[6],Omega[5]]))))
					F_wheel_1 = -np.dot(R,np.dot([[Omega[0], 0], [0, Omega[1]]], V_wheel_1))
					F_wheel_2 = -np.dot(R,np.dot([[Omega[2], 0], [0, Omega[3]]], V_wheel_2))
					F_T = U[j][i] + F_wheel_1 + F_wheel_2
					r_U = [r_F[j][i][0] - Rho[2], r_F[j][i][1] - Rho[3]]
					r_wheel_1 = [Omega[4]+Omega[6] - Rho[2], Omega[5] - Rho[3]]
					r_wheel_2 = [Omega[4]-Omega[6] - Rho[2], Omega[5] - Rho[3]]
					Tau = np.cross(r_U, U[j][i]) + np.cross(r_wheel_1, F_wheel_1) + np.cross(r_wheel_2, F_wheel_2)
					aux = [dT[j][i]*x for x in [Xnew[i][3], Xnew[i][4], Xnew[i][5], F_T[0]/Rho[0], F_T[1]/Rho[0], Tau/Rho[1]]]
					dx = [X[j][i][k]-X[j][i+1][k] for k in range(6)]
					error_x.append(abs(aux[0]-dx[0]))
					error_y.append(abs(aux[1]-dx[1]))
					error_phi.append(abs(aux[2]-dx[2]))
					error_xdot.append(abs(aux[3]-dx[3]))
					error_ydot.append(abs(aux[4]-dx[4]))
					error_phidot.append(abs(aux[5]-dx[5])) 
					Xnew.append([Xnew[i][k] +aux[k] for k in range(6)])

		error_d_fb.append([error_x,error_y,error_phi,error_xdot,error_ydot,error_phidot])
		Y_model_fb.append(Xnew)
        return Y_model, Y_model_fb, error_d, error_d_fb







