""""

This function is to train and test a dynamic model. The method used for learning can be simple Linear Regression or
Bayesian Regression, which should be specified by user.

Author: roya.sabbaghnovin@utah.edu

"""
# Import
from regression import LinearRegression
from Priors import RegressionPrior, BayesianPrior
from Models import BayesianModel


def train(X, U, r_F, dT, Y, learning_type, dynamic_model):
    """
    Returns a model with type specified by the model_type string trained on the
    given data.

    Args:
     A set of pre-processed data containing:
       X:  an array of all states [x y theta x_dot y_dot theta_dot]
       U:  an array of all actions [F_x F_y]
       r_f: an array of all moment vectors [r_F_x r_F_y]
       dT: an array of timesteps
       Y: an array of all new states [x y theta x_dot y_dot theta_dot]
       learning_type: (string) Type of learning algorithm for training.'LR' for linear regression and 'BR' for Bayesian Regression
       dynamic_model: (string) Type of model to train.'point_mass_on_a_wheel' or '2_wheel' 

    """

    ## Set priors
    if learning_type == 'LR':
        prior = RegressionPrior(LinearRegression)
    elif learning_type == 'BR':
        prior = BayesianPrior( dynamic_model,
			       max_hist=200,
                               sigma_retrain_thresh=0.05,
                               niter=20000,
                               burn=0,
                               thin=10,
                               tune_interval=1000)

    ## feed prior
    for i in range(len(X)):
        prior.update(X[i], U[i], r_F[i], dT[i], Y[i])
    model = prior.train()  # train the prior
    return model

def test(X, U, r_F, dT, Y, Final_model, learning_type, dynamic_model):
    """
    Returns predicted trajectory using the learned model and the error between predicted trajectory and the actual trajectory.

    Args:
     A set of pre-processed data containing:
       X:  an array of all states [x y theta x_dot y_dot theta_dot]
       U:  an array of all actions [F_x F_y]
       r_f: an array of all moment vectors [r_F_x r_F_y]
       dT: an array of timesteps
       Y: an array of all new states [x y theta x_dot y_dot theta_dot]
       Final_model: a learned model
       learning_type: (string) Type of algorithm for testing.'LR' for linear regression and 'BR' for Bayesian Regression
       dynamic_model: (string) Type of model to train.'point_mass_on_a_wheel' or '2_wheel' 

    """
    # Read dynamic parameters from model
    [Rho, Omega] = Final_model.params

    # Predict trajectory
    if learning_type == 'LR':
        print("None")
    elif learning_type == 'BR':
        Predicted_Y, Predicted_Y_fb, error_d, error_d_fb = BayesianModel.predict(Final_model, X, U, r_F, dT, Rho, Omega, dynamic_model)

    # Finding error
    Y_x = []
    Y_y = []
    Y_theta = []
    Y_xdot = []
    Y_ydot = []
    Y_thetadot = []
    for j in range(len(Y)):
	Y_x.append([])
	Y_y.append([])
	Y_theta.append([])
	Y_xdot.append([])
	Y_ydot.append([])
	Y_thetadot.append([])
	for i in range(len(Y[j])):
		Y_x[j].append(Y[j][i][0])
		Y_y[j].append(Y[j][i][1])
		Y_theta[j].append(Y[j][i][2])
		Y_xdot[j].append(Y[j][i][3])
		Y_ydot[j].append(Y[j][i][4])
		Y_thetadot[j].append(Y[j][i][5])
    Predicted_Y_x = []
    Predicted_Y_y = []
    Predicted_Y_theta = []
    Predicted_Y_xdot = []
    Predicted_Y_ydot = []
    Predicted_Y_thetadot = []
    fb_Predicted_Y_x = []
    fb_Predicted_Y_y = []
    fb_Predicted_Y_theta = []
    fb_Predicted_Y_xdot = []
    fb_Predicted_Y_ydot = []
    fb_Predicted_Y_thetadot = []
    error_x = []
    error_y = []
    error_theta = []
    error_xdot = []
    error_ydot = []
    error_thetadot = []
    error_x_fb = []
    error_y_fb = []
    error_theta_fb = []
    error_xdot_fb = []
    error_ydot_fb = []
    error_thetadot_fb = []
    for j in range(len(Predicted_Y)):
	Predicted_Y_x.append([])
	Predicted_Y_y.append([])
	Predicted_Y_theta.append([])
	Predicted_Y_xdot.append([])
	Predicted_Y_ydot.append([])
	Predicted_Y_thetadot.append([])
	fb_Predicted_Y_x.append([])
	fb_Predicted_Y_y.append([])
	fb_Predicted_Y_theta.append([])
	fb_Predicted_Y_xdot.append([])
	fb_Predicted_Y_ydot.append([])
	fb_Predicted_Y_thetadot.append([])
	error_x.append([])
	error_y.append([])
	error_theta.append([])
	error_xdot.append([])
	error_ydot.append([])
	error_thetadot.append([])
	error_x_fb.append([])
	error_y_fb.append([])
	error_theta_fb.append([])
	error_xdot_fb.append([])
	error_ydot_fb.append([])
	error_thetadot_fb.append([])
	for i in range(len(Predicted_Y[j])-1):
		Predicted_Y_x[j].append(Predicted_Y[j][i][0])
		Predicted_Y_y[j].append(Predicted_Y[j][i][1])
		Predicted_Y_theta[j].append(Predicted_Y[j][i][2])
		Predicted_Y_xdot[j].append(Predicted_Y[j][i][3])
		Predicted_Y_ydot[j].append(Predicted_Y[j][i][4])
		Predicted_Y_thetadot[j].append(Predicted_Y[j][i][5])
		fb_Predicted_Y_x[j].append(Predicted_Y_fb[j][i][0])
		fb_Predicted_Y_y[j].append(Predicted_Y_fb[j][i][1])
		fb_Predicted_Y_theta[j].append(Predicted_Y_fb[j][i][2])
		fb_Predicted_Y_xdot[j].append(Predicted_Y_fb[j][i][3])
		fb_Predicted_Y_ydot[j].append(Predicted_Y_fb[j][i][4])
		fb_Predicted_Y_thetadot[j].append(Predicted_Y_fb[j][i][5])
		error_x[j].append(Predicted_Y_x[j][i] - Y_x[j][i])
		error_y[j].append(Predicted_Y_y[j][i] - Y_y[j][i])
		error_theta[j].append(Predicted_Y_theta[j][i] - Y_theta[j][i])
		error_xdot[j].append(Predicted_Y_xdot[j][i] - Y_xdot[j][i])
		error_ydot[j].append(Predicted_Y_ydot[j][i] - Y_ydot[j][i])
		error_thetadot[j].append(Predicted_Y_thetadot[j][i] - Y_thetadot[j][i])
		error_x_fb[j].append(fb_Predicted_Y_x[j][i] - Y_x[j][i])
		error_y_fb[j].append(fb_Predicted_Y_y[j][i] - Y_y[j][i])
		error_theta_fb[j].append(fb_Predicted_Y_theta[j][i] - Y_theta[j][i])
		error_xdot_fb[j].append(fb_Predicted_Y_xdot[j][i] - Y_xdot[j][i])
		error_ydot_fb[j].append(fb_Predicted_Y_ydot[j][i] - Y_ydot[j][i])
		error_thetadot_fb[j].append(fb_Predicted_Y_thetadot[j][i] - Y_thetadot[j][i])
    error = [error_x, error_y, error_theta, error_xdot, error_ydot, error_thetadot]
    error_fb = [error_x_fb, error_y_fb, error_theta_fb, error_xdot_fb, error_ydot_fb, error_thetadot_fb]
    Pred_Y = [Predicted_Y_x, Predicted_Y_y, Predicted_Y_theta, Predicted_Y_xdot, Predicted_Y_ydot, Predicted_Y_thetadot]
    Pred_Y_fb = [fb_Predicted_Y_x, fb_Predicted_Y_y, fb_Predicted_Y_theta, fb_Predicted_Y_xdot, fb_Predicted_Y_ydot, fb_Predicted_Y_thetadot]
    Actual_Y = [Y_x, Y_y, Y_theta, Y_xdot, Y_ydot, Y_thetadot]
    return [Pred_Y, Pred_Y_fb, Actual_Y, error, error_fb, error_d, error_d_fb]


