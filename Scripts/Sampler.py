'''

This program implements PYMC Sampler.

Author: roya.sabbaghnovin@utah.edu

'''

# Import
import pymc as pm
import numpy as np
from numpy.linalg import inv


class Truncnorm(object):
    """
    A truncated normal distribution.  This is a good general-purpose
    univariate distribution for physical parameters since it is
    normal-like but allows customized support.
    """

    def __init__(self, vmin=0, vmax=1, mu=0, std=1):
        self.min = vmin
        self.max = vmax
        self.mu = mu
        self.std = std

    def __str__(self, *args, **kwargs):
        return "Truncnorm(min: %f, max: %f, mu: %f, std: %f)" % \
               (self.min,
                self.max,
                self.mu,
                self.std)

    def expected(self):
        return self.mu

    def sample(self, size=1):
        a = (self.min - self.mu) / self.std if self.std > 0 else self.mu
        b = (self.max - self.mu) / self.std if self.std > 0 else self.mu
        x = truncnorm.rvs(a, b, loc=self.mu, scale=self.std, size=size)
        return x[0] if size == 1 else x

class PYMCSampler(object):
    """
    This is a sampler that leverages PYMC for inference.  The idea is to write
    out a generative model for the data, and implement the physics simulation
    in an @deterministic method.

    The basic setup is to model dynamics Y = f(X,U), where X,Y \in {x,y,t,xdot,ydot,tdot},
    and U \in {fx,fy,tau}.

    """

    def __init__(self, X, U, r_F, dT, Y, model):
        super(PYMCSampler, self).__init__()
        # extract data
        self.X = X
        self.U = U
        self.r_F = r_F
        self.dT = dT
        self.Y = []
	self.model = model
	if self.model == "friction_only":
		for i in range(len(X)):
			self.Y.append(Y[i][len(Y[i])-1][0:3])
	else:
		for i in range(len(X)):
			self.Y.append(Y[i][len(Y[i])-1])


        self.rho_best = None


    def run(self, niter=2e4, burn=1e3, thin=10, tune_interval=1000):
        """
        Returns a tuple of length 2 (params, sigma). params is list of
        PhysicalParams instances. Third element is numpy.ndarray representing
        sigma.

        :param niter: Number of iterations
        :param burn: Variables will not be tallied until this many iterations are complete.
                     If less than 1, use as proportion of niter
        :param thin: Frequency of sample extraction
        :param tune_interval: Step methods will be tuned at intervals of this many iterations.
                              If less than 1, use as proportion of niter

        """
        if burn < 1:
            burn = niter * burn
        if tune_interval < 1:
            tune_interval = niter * tune_interval

        #############################################################
        #                      Rigid body parameters                     #
        #############################################################
        if self.model == "friction_only":
		## N-dimensional prior for wheel params [x_c,y_c]
		lb = np.array([-1, -1])
		ub = np.array([1, 1])
		Rho = pm.Uniform("Rho", lb, ub)
	else:
		## N-dimensional prior for wheel params [m,I,x_c,y_c]
		lb = np.array([0.0, 0.0, -1, -1])
		ub = np.array([200.0, 200.0, 1, 1])
		Rho = pm.Uniform("Rho", lb, ub)
        #############################################################
        #                      Wheel parameters                     #
        #############################################################
	if self.model == "friction_only":
		## N-dimensional prior for wheel params [muX,muY,muT,theta_mu]
		lb = np.array([0, 0, 0, -np.pi])
		ub = np.array([100, 100, 100, np.pi])
	        Omega = pm.Uniform("Omega", lb, ub)
	if self.model == "point_mass_on_a_wheel":
		## N-dimensional prior for wheel params [muX,muY,muT]
		lb = np.array([0, 0, -np.pi])
		ub = np.array([100, 100, np.pi])
	        Omega = pm.Uniform("Omega", lb, ub)
	elif self.model == "2_wheel":
		## N-dimensional prior for wheel params [muXR,muYR,muXL,muYL,x_s,y_s,b]
		lb = np.array([0, 0, 0, 0, -5, -5, -5])
		ub = np.array([100, 100, 100, 100, 5, 5, 5])
	        Omega = pm.Uniform("Omega", lb, ub)


        @pm.deterministic
        def muY(X= self.X, U= self.U, r_F= self.r_F, dT= self.dT, Rho= Rho, Omega= Omega):
	    Y_model = []
	    if self.model == "friction_only":
		for j in range(len(X)):
			Xnew = X[j][0]
			for i in range(len(X[j])):
				R = [[np.cos(Xnew[2]), -np.sin(Xnew[2])], [np.sin(Xnew[2]), np.cos(Xnew[2])]]
				R2 = [[np.cos(Omega[3]), -np.sin(Omega[3])], [np.sin(Omega[3]), np.cos(Omega[3])]]
				r_U = [r_F[j][i][0] - Rho[0], r_F[j][i][1] - Rho[1]]
				Tau = np.cross(r_U, U[j][i])
				theta_dot = Tau/Omega[2]
				vel1 = -np.dot(np.dot([[0, -theta_dot],[theta_dot, 0]],R), [Rho[0], Rho[1]]) 
				vel2 = -np.dot(np.dot(np.dot(R,[[float(1)/Omega[0], 0], [0, float(1)/Omega[1]]]),inv(R)),U[j][i])			
				[x_dot,y_dot] = [vel1[0]+vel2[0], vel1[1]+vel2[1]]
				aux = [dT[j][i]*x for x in [x_dot, y_dot, theta_dot]]
				Xnew = [Xnew[k] +aux[k] for k in range(3)]
			Y_model.append(Xnew)
	    if self.model == "point_mass_on_a_wheel":
		for j in range(len(X)):
			Xnew = X[j][0]
			for i in range(len(X[j])):
				R = [[np.cos(Xnew[2]), -np.sin(Xnew[2])], [np.sin(Xnew[2]), np.cos(Xnew[2])]]
				R2 = [[np.cos(Omega[2]), -np.sin(Omega[2])], [np.sin(Omega[2]), np.cos(Omega[2])]]
				V_wheel = np.dot(inv(R2),np.dot(inv(R),([Xnew[3], Xnew[4]]+ np.dot([[0, -Xnew[5]],[Xnew[5], 0]], np.dot(R,[Rho[2],Rho[3]])))))
				F_wheel = -np.dot(R,np.dot(R2,np.dot([[Omega[0], 0], [0, Omega[1]]], V_wheel)))
				F_T = U[j][i] + F_wheel
				r_U = [r_F[j][i][0] - Rho[2], r_F[j][i][1] - Rho[3]]
				Tau = np.cross(r_U, U[j][i])
				aux = [dT[j][i]*x for x in [Xnew[3], Xnew[4], Xnew[5], F_T[0]/Rho[0], F_T[1]/Rho[0], Tau/Rho[1]]]
				Xnew = [Xnew[k] +aux[k] for k in range(6)]
			Y_model.append(Xnew)
	    elif self.model == "2_wheel":
		for j in range(len(X)):
			Xnew = X[j][0]
			for i in range(len(X[j])):
				R = [[np.cos(Xnew[2]), -np.sin(Xnew[2])], [np.sin(Xnew[2]), np.cos(Xnew[2])]]
				V_wheel_1 = np.dot(inv(R),([Xnew[3], Xnew[4]]+ np.dot([[0, -Xnew[5]],[Xnew[5], 0]], np.dot(R,[Omega[4]+Omega[6],Omega[5]]))))
				V_wheel_2 = np.dot(inv(R),([Xnew[3], Xnew[4]]+ np.dot([[0, -Xnew[5]],[Xnew[5], 0]], np.dot(R,[Omega[4]-Omega[6],Omega[5]]))))
				F_wheel_1 = -np.dot(R,np.dot([[Omega[0], 0], [0, Omega[1]]], V_wheel_1))
				F_wheel_2 = -np.dot(R,np.dot([[Omega[2], 0], [0, Omega[3]]], V_wheel_2))
				F_T = U[j][i] + F_wheel_1 + F_wheel_2
				r_U = [r_F[j][i][0] - Rho[2], r_F[j][i][1] - Rho[3]]
				r_wheel_1 = [Omega[4]+Omega[6] - Rho[2], Omega[5] - Rho[3]]
				r_wheel_2 = [Omega[4]-Omega[6] - Rho[2], Omega[5] - Rho[3]]
				Tau = np.cross(r_U, U[j][i]) + np.cross(r_wheel_1, F_wheel_1) + np.cross(r_wheel_2, F_wheel_2)
				aux = [dT[j][i]*x for x in [Xnew[3], Xnew[4], Xnew[5], F_T[0]/Rho[0], F_T[1]/Rho[0], Tau/Rho[1]]]
				Xnew = [Xnew[k] +aux[k] for k in range(6)]
			Y_model.append(Xnew)
            return Y_model

        sigma = pm.Uniform("sigma", 0.0, 200.0, value=3.0)  # value=10., observed=True
        y = pm.Normal("y", mu=muY, tau=1.0 / sigma ** 2, value=self.Y, observed=True)

        mdl = pm.Model([Rho, Omega, y])
        self.M = pm.MCMC(mdl)
        self.M.sample(niter, burn, thin, tune_interval)
        fit = self.M.stats()
        hpdRho = fit['Rho']['95% HPD interval']
        muRho = np.atleast_2d(fit['Rho']['mean'])
        muRho = np.clip(muRho, hpdRho[0], hpdRho[1])
        stdRho = np.atleast_2d(fit['Rho']['standard deviation'])
        hpdOmega = fit['Omega']['95% HPD interval']
        muOmega = np.atleast_2d(fit['Omega']['mean'])
        muOmega = np.clip(muOmega, hpdOmega[0], hpdOmega[1])
        stdOmega = np.atleast_2d(fit['Omega']['standard deviation'])
        inertia = Truncnorm(hpdRho[0], hpdRho[1], muRho[0], stdRho[0])
        friction = Truncnorm(hpdOmega[0], hpdOmega[1], muOmega[0], stdOmega[0])
        return (inertia.mu, friction.mu, inertia.std, friction.std)
