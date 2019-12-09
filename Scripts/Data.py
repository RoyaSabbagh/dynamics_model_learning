'''

This program is to read data from files and return them as processed dataset.

Author: roya.sabbaghnovin@utah.edu

'''

from xlrd import open_workbook
import math
import numpy as np
from scipy import signal
import os


def butter_filter(sig):

    #Creation of the filter
    sf = 10
    cutOff =1 # Cutoff frequency
    nyq = 0.5 * sf
    N  = 6    # Filter order
    fc = cutOff / nyq # Cutoff frequency normal
    a,b = signal.butter(N, fc, btype="lowpass", analog=False)
    filtered = signal.filtfilt(a,b, sig)
    return filtered


def generate_data():
    """"
     Generates some fake 2d data for testing. used while developing the code.

    """
    u = np.array([[0.0, 0.0], [0.5,0.5], [1.0, 1.0], [1.5, 1.5], [2.0, 2.0], [2.1, 2.1], [2.2, 2.2]])
    U = np.repeat(u, 10, axis=0)
    X= np.ones((70,2))
    for i in range(69):
        X[i+1,:] = X[i,:] + 0.8*U[i,:]

    U_train = U[:-1]
    X_train = X[:-1]
    Y_train = X[1:]
    dt_train = 0.1*np.ones(len(U_train))

    return [X_train, U_train, dt_train, Y_train]


def read_data(target_object, mode):
    """"
    Reads train/test data from Excel files
    Returns all data as [X, U, gripper_pos,  dT, Y], containing states, actions, new states, gipper position, timesteps.

    """
    dirname = os.path.dirname(os.path.dirname(__file__))
    filename = os.path.join(dirname, 'Data/%s_%s.xlsx'%(target_object,mode))
    wb = open_workbook(filename)
    r_gripper = 0.26
    X = []
    Y = []
    gripper_pos = []
    dT = []
    U = []
    j = 0
    for sheet in wb.sheets():
	X.append([])
	Y.append([])
	gripper_pos.append([])
	dT.append([])
	U.append([])
	force = []
	PAM_phi = []
        number_of_rows = sheet.nrows
        for row in range(1,number_of_rows-1):
		PAM_x = sheet.cell(row, 1).value
		PAM_y = sheet.cell(row, 2).value
		PAM_phi.append(sheet.cell(row, 3).value)
		object_x = sheet.cell(row, 7).value
		object_y = sheet.cell(row, 8).value
		object_phi = sheet.cell(row, 9).value
		object_x_dot = sheet.cell(row, 10).value
		object_y_dot = sheet.cell(row, 11).value
		object_phi_dot = sheet.cell(row, 12).value
		object_x_next = sheet.cell(row+1, 7).value
		object_y_next = sheet.cell(row+1, 8).value
		object_phi_next = sheet.cell(row+1, 9).value
		object_x_dot_next = sheet.cell(row+1, 10).value
		object_y_dot_next = sheet.cell(row+1, 11).value
		object_phi_dot_next = sheet.cell(row+1, 12).value
		force.append(sheet.cell(row, 13).value)
		gripper_x = PAM_x + r_gripper * np.cos(PAM_phi[row-1])
		gripper_y = PAM_y + r_gripper * np.sin(PAM_phi[row-1])
		dT[j].append(sheet.cell(row+1, 0).value-sheet.cell(row, 0).value)
		X[j].append([object_x, object_y, object_phi, object_x_dot, object_y_dot, object_phi_dot])
		Y[j].append([object_x_next, object_y_next, object_phi_next, object_x_dot_next, object_y_dot_next, object_phi_dot_next])
		gripper_pos[j].append([gripper_x-object_x, gripper_y-object_y])

	f = butter_filter(force)
	U[j]=[[-f[i] * math.cos(PAM_phi[i]), -f[i] * math.sin(PAM_phi[i])] for i in range(len(f))]
        j += 1
    #Force.append(4.45*24.096*sheet.cell(row, col).value) # loadcell calibration
    return [X, U, gripper_pos,  dT, Y]
