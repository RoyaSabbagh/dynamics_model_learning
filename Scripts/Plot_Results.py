""""

This function is to plot final results

Author: roya.sabbaghnovin@utah.edu

"""
import matplotlib.pyplot as plt
import numpy as np
import math
from Learning import train, test
from Data import read_data, butter_filter
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from numpy.linalg import inv
from matplotlib.collections import PatchCollection
from Models import *

def setBoxColors(bp, edge_color, fill_color, mid_color):
		for element in ['boxes', 'whiskers', 'fliers', 'caps']:
			plt.setp(bp[element], color=edge_color)
		plt.setp(bp['medians'], color=mid_color)
		for patch in bp['boxes']:
			patch.set(facecolor=fill_color)

def plot_results(Pred_Y_walker_2, Pred_Y_fb_walker_2, Actual_Y_walker, error_walker, error_fb_walker, error_walker_2, error_fb_walker_2, error_walker_3, error_fb_walker_3, Pred_Y_cart, Pred_Y_fb_cart, Actual_Y_cart, error_cart, error_fb_cart, error_cart_2, error_fb_cart_2, error_cart_3, error_fb_cart_3, Pred_Y_blue_chair, Pred_Y_fb_blue_chair, Actual_Y_blue_chair, error_blue_chair, error_fb_blue_chair, error_blue_chair_2, error_fb_blue_chair_2, error_blue_chair_3, error_fb_blue_chair_3, Pred_Y_gray_chair, Pred_Y_fb_gray_chair, Actual_Y_gray_chair, error_gray_chair, error_fb_gray_chair, error_gray_chair_2, error_fb_gray_chair_2, error_gray_chair_3, error_fb_gray_chair_3):
    """"
    This function plots the final predicted trajectory and errors in two separate figures.

    """

    std = []
    mean = []
    std_fb = []
    mean_fb = []
    std_2 = []
    mean_2 = []
    std_fb_2 = []
    mean_fb_2 = []
    std_3 = []
    mean_3 = []
    std_fb_3 = []
    mean_fb_3 = []
    l = 0
    s = 1
    plt.figure(2, figsize=(15,6))
    blue_circle = dict(markerfacecolor='b', marker='o')
    red_circle = dict(markerfacecolor='r', marker='o')
    width = 0.3
    i=-1
    bp = [[[],[],[],[],[],[]],[[],[],[],[],[],[]],[[],[],[],[],[],[]],[[],[],[],[],[],[]]]
    rmse = [[],[],[],[]]
    rmse_fb = [[],[],[],[]]
    rmse_2 = [[],[],[],[]]
    rmse_fb_2 = [[],[],[],[]]
    rmse_3 = [[],[],[],[]]
    rmse_fb_3 = [[],[],[],[]]
    for obj  in ["walker", "blue_chair", "gray_chair", "cart"]:
        i+=1
        for traj in range(len(globals()["error_{0}".format(obj)][0])):
	           rmse[i].append(np.sqrt(abs(globals()["error_{0}".format(obj)][l][traj][50])**2+abs(globals()["error_{0}".format(obj)][s][traj][50])**2))
	           rmse_fb[i].append(np.sqrt(abs(globals()["error_fb_{0}".format(obj)][l][traj][50])**2+abs(globals()["error_fb_{0}".format(obj)][s][traj][50])**2))
	           rmse_2[i].append(np.sqrt(abs(globals()["error_{0}_2".format(obj)][l][traj][50])**2+abs(globals()["error_{0}_2".format(obj)][s][traj][50])**2))
	           rmse_fb_2[i].append(np.sqrt(abs(globals()["error_fb_{0}_2".format(obj)][l][traj][50])**2+abs(globals()["error_fb_{0}_2".format(obj)][s][traj][50])**2))
	           rmse_3[i].append(np.sqrt(abs(globals()["error_{0}_3".format(obj)][l][traj][50])**2+abs(globals()["error_{0}_3".format(obj)][s][traj][50])**2))
	           rmse_fb_3[i].append(np.sqrt(abs(globals()["error_fb_{0}_3".format(obj)][l][traj][50])**2+abs(globals()["error_fb_{0}_3".format(obj)][s][traj][50])**2))
		#rmse.append(np.sqrt(np.mean([globals()["error_{0}".format(obj)][0][traj][t]**2 for t in range(len(globals()["error_{0}".format(obj)][0][traj]))])))
		#rmse_fb.append(np.sqrt(np.mean([globals()["error_fb_{0}".format(obj)][0][traj][t]**2 for t in range(len(globals()["error_fb_{0}".format(obj)][0][traj]))])))
	    # std.append(np.std(rmse))
	    # mean.append(np.mean(rmse))
	    # std_fb.append(np.std(rmse_fb))
	    # mean_fb.append(np.mean(rmse_fb))
	    # std_2.append(np.std(rmse_2))
	    # mean_2.append(np.mean(rmse_2))
	    # std_fb_2.append(np.std(rmse_fb_2))
	    # mean_fb_2.append(np.mean(rmse_fb_2))
	    # std_3.append(np.std(rmse_3))
	    # mean_3.append(np.mean(rmse_3))
	    # std_fb_3.append(np.std(rmse_fb_3))
	    # mean_fb_3.append(np.mean(rmse_fb_3))


    # N = 4
    # ind = np.arange(N)
    # width = 0.1
    # print mean
    # print mean_2
    # print mean_3
    # plt.grid(True, 'major', 'y', ls='--', lw=.5, c='k', alpha=.3)
    # i = 0


        bp[i][0] = plt.boxplot(rmse[i], positions = [i*2.5+0.5], widths = width, patch_artist=True, flierprops=blue_circle)
        setBoxColors(bp[i][0], 'black', (0.93,0.6,0.26, 1), 'black')

        bp[i][1] = plt.boxplot(rmse_fb[i], positions = [i*2.5+0.3+0.5], widths = width, patch_artist=True, flierprops=blue_circle)
        setBoxColors(bp[i][1], 'black', (0.4, 0.73, 0.73, 1), 'black')

        bp[i][2] = plt.boxplot(rmse_2[i], positions = [i*2.5+0.6+0.5], widths = width, patch_artist=True, flierprops=blue_circle)
        setBoxColors(bp[i][2], 'black', (0.73,0.33,0.33, 1), 'black')

        bp[i][3] = plt.boxplot(rmse_fb_2[i], positions = [i*2.5+0.9+0.5], widths = width, patch_artist=True, flierprops=blue_circle)
        setBoxColors(bp[i][3], 'black', (0.6, 0.73, 0.33, 1), 'black')

        bp[i][4] = plt.boxplot(rmse_3[i], positions = [i*2.5+1.2+0.5], widths = width, patch_artist=True, flierprops=blue_circle)
        setBoxColors(bp[i][4], 'black', (0.33,0.33,0.4, 1), 'black')

        bp[i][5] = plt.boxplot(rmse_fb_3[i], positions = [i*2.5+1.5+0.5], widths = width, patch_artist=True, flierprops=blue_circle)
        setBoxColors(bp[i][5], 'black', (0.33, 0.53, 0.73, 1), 'black')
    #

    ax = plt.axes()
    ax.set_ylim((0,1.5))
    ax.set_xlim((0,10))
    ax.set_ylabel('$\mathrm{Final\; Position\; Error \;(m)}$', fontsize=20)
    ax.set_xticklabels(['$\mathrm{Walker}$', '$\mathrm{Blue Chair}$', '$\mathrm{Gray Chair}$', '$\mathrm{Rack}$'], fontsize=18)
    ax.set_yticklabels(['$\mathrm{0.0}$', '$\mathrm{0.5}$', '$\mathrm{1.0}$', '$\mathrm{1.5}$', '$\mathrm{2.0}$', '$\mathrm{2.5}$'], fontsize=16)
    plt.xticks([1.25, 3.75, 6.25, 8.75])
    plt.yticks([0, 0.5, 1 , 1.5])
    ax.legend([bp[0][0]["boxes"][0], bp[0][2]["boxes"][0], bp[0][4]["boxes"][0], bp[0][1]["boxes"][0], bp[0][3]["boxes"][0], bp[0][5]["boxes"][0]], ['$\mathrm{Model\;1\;w/o \;feedback}$', '$\mathrm{Model\;2\;w/o feedback}$', '$\mathrm{Model\;3\;w/o \;feedback}$', '$\mathrm{Model\;1\;w/ \;feedback}$', '$\mathrm{Model\;2\;w/ \;feedback}$', '$\mathrm{Model\;3\;w/ \;feedback}$'], loc='upper center', ncol=2, fontsize=14)

    ax.yaxis.grid(True)
    plt.savefig("/home/roya/model_comparison.pdf", dpi =300)
    # plt.bar(ind + width, mean, width, yerr=std, align='center', ecolor='black', capsize=6, color=(0.93,0.6,0.26, 1), label='Model 1, w/o feedback', edgecolor = (0.93,0.6,0.26, 1))
    # plt.bar(ind + 2*width+0.04, mean_fb, width, yerr=std_fb, align='center', ecolor='black', capsize=6, color=(0.4, 0.73, 0.73, 1), label='Model 1, w/ feedback', edgecolor = (0.4, 0.73, 0.73, 1))
    # plt.bar(ind + 3*width+0.08, mean_2, width, yerr=std_2, align='center', ecolor='black', capsize=6, color=(0.73,0.33,0.33, 1), label='Model 2, w/o feedback', edgecolor = (0.73,0.33,0.33, 1))
    # plt.bar(ind + 4*width+0.12, mean_fb_2, width, yerr=std_fb_2, align='center', ecolor='black', capsize=6, color=(0.6, 0.73, 0.33, 1), label='Model 2, w/ feedback', edgecolor = (0.6, 0.73, 0.33, 1))
    # plt.bar(ind + 5*width+0.16, mean_3, width, yerr=std_3, align='center', ecolor='black', capsize=6, color=(0.33,0.33,0.4, 1), label='Model 3, w/o feedback', edgecolor = (0.33,0.33,0.4, 1))
    # plt.bar(ind + 6*width+0.2, mean_fb_3, width, yerr=std_fb_3, align='center', ecolor='black', capsize=6, color=(0.33, 0.53, 0.73, 1), label='Model 3, w/ feedback', edgecolor = (0.33, 0.53, 0.73, 1))
    #
    # plt.ylabel('$\mathrm{RMSE(m)}$', fontsize=20)
    #
    # plt.xticks(ind+ 5*width, ('$\mathrm{Gray Chair}$', '$\mathrm{Rack}$', '$\mathrm{Blue Chair}$', '$\mathrm{Walker}$'), fontsize=20)
    #
    # plt.legend(loc=2)
    #
    # plt.ylim((0,1))


    # plt.figure(1,figsize=(12, 12))
    # f = 0
    # for j in [8,0,11,14]:
	#     plt.subplot(4,4,f+1)
	#     m1 = 0
	#     m2 = 50
	#     plt.plot(Pred_Y_gray_chair[0][j][m1:m2], Pred_Y_gray_chair[1][j][m1:m2], 'r' , linestyle=':', linewidth = 2)
	#     plt.plot(Pred_Y_fb_gray_chair[0][j][m1:m2], Pred_Y_fb_gray_chair[1][j][m1:m2], 'b--', linewidth = 3)
	#     plt.plot(Actual_Y_gray_chair[0][j][m1:m2], Actual_Y_gray_chair[1][j][m1:m2], 'g', linewidth = 2, alpha= 1)
	#     for k in range(1,5):
	# 	i = k*10
	# 	plt.arrow(Pred_Y_fb_gray_chair[0][j][i], Pred_Y_fb_gray_chair[1][j][i], 0.1*np.cos(Pred_Y_fb_gray_chair[2][j][i]+np.pi/2), 0.1*np.sin(Pred_Y_fb_gray_chair[2][j][i]+np.pi/2),
	# 	      head_width=0.02, length_includes_head=True, color='b', alpha= 1)
	# 	plt.arrow(Actual_Y_gray_chair[0][j][i], Actual_Y_gray_chair[1][j][i], 0.1*np.cos(Actual_Y_gray_chair[2][j][i]+np.pi/2), 0.1*np.sin(Actual_Y_gray_chair[2][j][i]+np.pi/2),
	# 	      head_width=0.02, length_includes_head=True, color='g', alpha= 1)
	#     if f==0:
	# 	plt.xlim((2, 2.8))
	#     	plt.ylim((1.6,2.4))
	#     elif f==1:
	# 	plt.xlim((1.6, 2.4))
	#     	plt.ylim((1,1.8))
	#     elif f==2:
	# 	plt.xlim((1.8, 2.6))
	#     	plt.ylim((1.4,2.2))
	#     elif f==3:
	# 	plt.xlim((2, 2.8))
	#     	plt.ylim((0.6,1.4))
	#     f += 1
    #
    # for j in [11,1,5,9]:
	#     plt.subplot(4,4,f+1)
	#     m1 = 0
	#     m2 = 50
	#     plt.plot(Pred_Y_cart[0][j][m1:m2], Pred_Y_cart[1][j][m1:m2], 'r' , linestyle=':', linewidth = 2)
	#     plt.plot(Pred_Y_fb_cart[0][j][m1:m2], Pred_Y_fb_cart[1][j][m1:m2], 'b--', linewidth = 3)
	#     plt.plot(Actual_Y_cart[0][j][m1:m2], Actual_Y_cart[1][j][m1:m2], 'g', linewidth = 2, alpha= 1)
	#     if f==5:
	# 	    for k in range(1,5):
	# 		i = k*10
	# 		plt.arrow(Pred_Y_fb_cart[0][j][i], Pred_Y_fb_cart[1][j][i], 0.1*np.cos(Pred_Y_fb_cart[2][j][i]-np.pi/2), 0.1*np.sin(Pred_Y_fb_cart[2][j][i]-np.pi/2),
	# 		      head_width=0.02, length_includes_head=True, color='b', alpha= 1)
	# 		plt.arrow(Actual_Y_cart[0][j][i], Actual_Y_cart[1][j][i], 0.1*np.cos(Actual_Y_cart[2][j][i]-np.pi/2), 0.1*np.sin(Actual_Y_cart[2][j][i]-np.pi/2),
	# 		      head_width=0.02, length_includes_head=True, color='g', alpha= 1)
	#     elif f==7:
	# 	    for k in range(1,(len(Pred_Y_fb_cart[0][j])-1)/10):
	# 		i = k*10
	# 		plt.arrow(Pred_Y_fb_cart[0][j][i], Pred_Y_fb_cart[1][j][i], 0.1*np.cos(Pred_Y_fb_cart[2][j][i]), 0.1*np.sin(Pred_Y_fb_cart[2][j][i]),
	# 		      head_width=0.02, length_includes_head=True, color='b', alpha= 1)
	# 		plt.arrow(Actual_Y_cart[0][j][i], Actual_Y_cart[1][j][i], 0.1*np.cos(Actual_Y_cart[2][j][i]), 0.1*np.sin(Actual_Y_cart[2][j][i]),
	# 		      head_width=0.02, length_includes_head=True, color='g', alpha= 1)
	#     else:
	# 	    for k in range(1,(len(Pred_Y_fb_cart[0][j])-1)/10):
	# 		i = k*10
	# 		plt.arrow(Pred_Y_fb_cart[0][j][i], Pred_Y_fb_cart[1][j][i], 0.1*np.cos(Pred_Y_fb_cart[2][j][i]+np.pi/2), 0.1*np.sin(Pred_Y_fb_cart[2][j][i]+np.pi/2),
	# 		      head_width=0.02, length_includes_head=True, color='b', alpha= 1)
	# 		plt.arrow(Actual_Y_cart[0][j][i], Actual_Y_cart[1][j][i], 0.1*np.cos(Actual_Y_cart[2][j][i]+np.pi/2), 0.1*np.sin(Actual_Y_cart[2][j][i]+np.pi/2),
	# 		      head_width=0.02, length_includes_head=True, color='g', alpha= 1)
   	#     if f==4:
	# 	plt.xlim((2.6, 3.4))
	#     	plt.ylim((0.2,1))
	#     elif f==5:
	# 	plt.xlim((2.6, 3.4))
	#     	plt.ylim((1.2,2))
	#     elif f==6:
	# 	plt.xlim((2.4, 3.2))
	#     	plt.ylim((0.6,1.4))
	#     elif f==7:
	# 	plt.xlim((2, 2.8))
	#     	plt.ylim((0.2,1))
	#     f += 1
    #
    # for j in [7,12,2,0]:
	#     plt.subplot(4,4,f+1)
	#     m1 = 0
	#     m2 = 50
	#     plt.plot(Pred_Y_blue_chair[0][j][m1:m2], Pred_Y_blue_chair[1][j][m1:m2], 'r' , linestyle=':', linewidth = 2)
	#     plt.plot(Pred_Y_fb_blue_chair[0][j][m1:m2], Pred_Y_fb_blue_chair[1][j][m1:m2], 'b--', linewidth = 3)
	#     plt.plot(Actual_Y_blue_chair[0][j][m1:m2], Actual_Y_blue_chair[1][j][m1:m2], 'g', linewidth = 2, alpha= 1)
    # 	    for k in range(1,5):
	# 	i = k*10
	# 	plt.arrow(Pred_Y_fb_blue_chair[0][j][i], Pred_Y_fb_blue_chair[1][j][i], 0.1*np.cos(Pred_Y_fb_blue_chair[2][j][i]+np.pi/2), 0.1*np.sin(Pred_Y_fb_blue_chair[2][j][i]+np.pi/2),
	# 	      head_width=0.02, length_includes_head=True, color='b', alpha= 1)
	# 	plt.arrow(Actual_Y_blue_chair[0][j][i], Actual_Y_blue_chair[1][j][i], 0.1*np.cos(Actual_Y_blue_chair[2][j][i]+np.pi/2), 0.1*np.sin(Actual_Y_blue_chair[2][j][i]+np.pi/2),
	# 	      head_width=0.02, length_includes_head=True, color='g', alpha= 1)
    #         if f==8:
	# 	plt.xlim((1.6, 2.4))
	#     	plt.ylim((1,1.8))
	#     elif f==9:
	# 	plt.xlim((1.4, 2.2))
	#     	plt.ylim((1.2,2))
	#     elif f==10:
	# 	plt.xlim((1,1.8))
	#     	plt.ylim((1.2,2))
	#     elif f==11:
	# 	plt.xlim((1.4, 2.2))
	#     	plt.ylim((1.2,2))
	#     f += 1
    #
    # for j in [13,11,6,2]:
	#     plt.subplot(4,4,f+1)
	#     m1 = 0
	#     m2 = 50
	#     plt.plot(Pred_Y_walker[0][j][m1:m2], Pred_Y_walker[1][j][m1:m2], 'r' , linestyle=':', linewidth = 2)
	#     plt.plot(Pred_Y_fb_walker[0][j][m1:m2], Pred_Y_fb_walker[1][j][m1:m2], 'b--', linewidth = 3)
	#     plt.plot(Actual_Y_walker[0][j][m1:m2], Actual_Y_walker[1][j][m1:m2], 'g', linewidth = 2, alpha= 1)
	#     if f==12:
	# 	    for k in range(1,5):
	# 		i = k*10
	# 		plt.arrow(Pred_Y_fb_walker[0][j][i], Pred_Y_fb_walker[1][j][i], 0.1*np.cos(Pred_Y_fb_walker[2][j][i]), 0.1*np.sin(Pred_Y_fb_walker[2][j][i]),
	# 		      head_width=0.02, length_includes_head=True, color='b', alpha= 1)
	# 		plt.arrow(Actual_Y_walker[0][j][i], Actual_Y_walker[1][j][i], 0.1*np.cos(Actual_Y_walker[2][j][i]), 0.1*np.sin(Actual_Y_walker[2][j][i]),
	# 		      head_width=0.02, length_includes_head=True, color='g', alpha= 1)
	#     elif f==15:
	# 	    for k in range(1,(len(Pred_Y_fb_walker[0][j])-1)/10):
	# 		i = k*10
	# 		plt.arrow(Pred_Y_fb_walker[0][j][i], Pred_Y_fb_walker[1][j][i], 0.1*np.cos(Pred_Y_fb_walker[2][j][i]+np.pi), 0.1*np.sin(Pred_Y_fb_walker[2][j][i]+np.pi),
	# 		      head_width=0.02, length_includes_head=True, color='b', alpha= 1)
	# 		plt.arrow(Actual_Y_walker[0][j][i], Actual_Y_walker[1][j][i], 0.1*np.cos(Actual_Y_walker[2][j][i]+np.pi), 0.1*np.sin(Actual_Y_walker[2][j][i]+np.pi),
	# 		      head_width=0.02, length_includes_head=True, color='g', alpha= 1)
	#     else:
	# 	    for k in range(1,(len(Pred_Y_fb_walker[0][j])-1)/10):
	# 		i = k*10
	# 		plt.arrow(Pred_Y_fb_walker[0][j][i], Pred_Y_fb_walker[1][j][i], 0.1*np.cos(Pred_Y_fb_walker[2][j][i]+np.pi/2), 0.1*np.sin(Pred_Y_fb_walker[2][j][i]+np.pi/2),
	# 		      head_width=0.02, length_includes_head=True, color='b', alpha= 1)
	# 		plt.arrow(Actual_Y_walker[0][j][i], Actual_Y_walker[1][j][i], 0.1*np.cos(Actual_Y_walker[2][j][i]+np.pi/2), 0.1*np.sin(Actual_Y_walker[2][j][i]+np.pi/2),
	# 		      head_width=0.02, length_includes_head=True, color='g', alpha= 1)
   	#     if f==12:
	# 	plt.xlim((2, 2.8))
	#     	plt.ylim((-0.6,0.4))
	#     elif f==13:
	# 	plt.xlim((2.2, 3))
	#     	plt.ylim((0.6,1.4))
	#     elif f==14:
	# 	plt.xlim((2.8, 3.6))
	#     	plt.ylim((-0.2,0.6))
	#     elif f==15:
	# 	plt.xlim((2.4, 3.2))
	#     	plt.ylim((1.2,2))
	#     f += 1


    plt.show()


if __name__ == '__main__':

	# blue_chair, 20000 samples, clean data:
	Rho_blue_chair = [ 69.41451775,  93.99466481,  -0.44036373,   0.29626516]
	Omega_blue_chair = [ 36.44746592,  36.61569852,   0.02108361]

	#blue_chair, 50000 samples, clean data, model 2:
	Rho_blue_chair_2 = [ 181.81275796,  157.91114061,   -0.91095963,    0.65619819]
	Omega_blue_chair_2 = [ 10.29493421,  27.4847522 ,  24.01881465,   1.82628059,     2.46654727,   0.46269262,  -2.60026588]

	#blue_chair, 50000 samples, clean data, model 3:
	Rho_blue_chair_3 = [ 0.40258775,  0.03340228]
	Omega_blue_chair_3 = [ 84.81778851,  67.47640435,  81.85900011,   1.58828871]

	# gray_chair, 20000 samples, clean data:
	Rho_gray_chair = [  6.29082319e+01,   1.09790119e+02,   7.18271637e-02,   9.71824506e-02]
	Omega_gray_chair = [ 27.39445965,  27.32128562,  -0.08073383]

	#gray_chair, 50000 samples, clean data, model 2:
	Rho_gray_chair_2 = [ 88.41454103,  127.35524659,   -0.60981765,   -0.19581444]
	Omega_gray_chair_2 = [  2.65727732e+01,   1.96523171e+01,   2.52874559e+01,   2.14380029e+01,   5.75052356e-02,   3.01197473e-01,  -2.89359969e-03]

	#gray_chair, 50000 samples, clean data, model 3:
	Rho_gray_chair_3 = [ 0.40258775,  0.03340228]
	Omega_gray_chair_3 = [ 84.81778851,  67.47640435,  81.85900011,   1.58828871]

	# walker, 20000 samples, clean data:
	Rho_walker = [  8.31562175e+01,   7.46498361e+01,   9.11737583e-02,   7.34392901e-02]
	Omega_walker = [ 46.14655612,  45.3535728 ,  -1.4426394]

	#walker, 50000 samples, clean data, model 2:
	Rho_walker_2 = [ 80.36407651,   100.37032702,    0.82370979,    0.20066161]
	Omega_walker_2 = [  1.34559205e+00,   6.35589823e+01,   4.15842935e+01,   3.64209927e+01,   3.56998798e-02,  -2.13576757e-01,   2.69359326e-01]

	#walker, 50000 samples, clean data, model 3:
	Rho_walker_3 = [ 0.40258775,  0.03340228]
	Omega_walker_3 = [ 84.81778851,  67.47640435,  81.85900011,   1.58828871]

	# cart, 20000 samples, clean data:
	Rho_cart = [  86.58925901,  183.18027128,    0.53349051,    0.38911036]
	Omega_cart = [  2.59474801e+01,   2.64456098e+01,  -1.93996588e-02]

	#cart, 50000 samples, clean data, model 2:
	Rho_cart_2 = [ 267.13793947,  282.76702281,    0.81227457,    0.6449838 ]
	Omega_cart_2 = [ 15.26349935,  17.1323213 ,  31.64002716,  11.44320843,   -0.42572011,   0.67113536,   0.2065445 ]

	#cart, 50000 samples, clean data, model 3:
	Rho_cart_3 = [ 0.40258775,  0.03340228]
	Omega_cart_3 = [ 84.81778851,  67.47640435,  81.85900011,   1.58828871]

	# cart, 20000 samples, clean data:
	Rho_walker2 = [  86.58925901,  183.18027128,    0.53349051,    0.38911036]
	Omega_walker2 = [  2.59474801e+01,   2.64456098e+01,  -1.93996588e-02]

	#cart, 50000 samples, clean data, model 2:
	Rho_walker2_2 = [ 267.13793947,  282.76702281,    0.81227457,    0.6449838 ]
	Omega_walker2_2 = [ 15.26349935,  17.1323213 ,  31.64002716,  11.44320843,   -0.42572011,   0.67113536,   0.2065445 ]

	#cart, 50000 samples, clean data, model 3:
	Rho_walker2_3 = [ 0.40258775,  0.03340228]
	Omega_walker2_3 = [ 84.81778851,  67.47640435,  81.85900011,   1.58828871]

	for obj in ["walker", "cart", "blue_chair", "gray_chair"]:

		## Read test data from files
		[globals()["X_test_{0}".format(obj)], globals()["U_test_{0}".format(obj)], globals()["r_F_test_{0}".format(obj)], globals()["dT_test_{0}".format(obj)], globals()["Y_test_{0}".format(obj)]] = read_data(obj, "test")

		# Learn Model
		Rho = globals()["Rho_{0}".format(obj)]
		Omega = globals()["Omega_{0}".format(obj)]
		Rho_2 = globals()["Rho_{0}_2".format(obj)]
		Omega_2 = globals()["Omega_{0}_2".format(obj)]
		Rho_3 = globals()["Rho_{0}_3".format(obj)]
		Omega_3 = globals()["Omega_{0}_3".format(obj)]
		Final_model = BayesianModel()
		Final_model.params = [Rho, Omega]
		Final_model_2 = BayesianModel()
		Final_model_2.params = [Rho_2, Omega_2]
		Final_model_3 = BayesianModel()
		Final_model_3.params = [Rho_3, Omega_3]

		# Obtain predictions
		[globals()["Pred_Y_{0}_2".format(obj)], globals()["Pred_Y_fb_{0}_2".format(obj)], globals()["Actual_Y_{0}_2".format(obj)], globals()["error_{0}_2".format(obj)], globals()["error_fb_{0}_2".format(obj)], globals()["error_d_{0}_2".format(obj)], globals()["error_d_fb_{0}_2".format(obj)]] = test(globals()["X_test_{0}".format(obj)], globals()["U_test_{0}".format(obj)], globals()["r_F_test_{0}".format(obj)], globals()["dT_test_{0}".format(obj)], globals()["Y_test_{0}".format(obj)], Final_model_2, learning_type='BR', dynamic_model='2_wheel')

		[globals()["Pred_Y_{0}".format(obj)], globals()["Pred_Y_fb_{0}".format(obj)], globals()["Actual_Y_{0}".format(obj)], globals()["error_{0}".format(obj)], globals()["error_fb_{0}".format(obj)], globals()["error_d_{0}".format(obj)], globals()["error_d_fb_{0}".format(obj)]] = test(globals()["X_test_{0}".format(obj)], globals()["U_test_{0}".format(obj)], globals()["r_F_test_{0}".format(obj)], globals()["dT_test_{0}".format(obj)], globals()["Y_test_{0}".format(obj)], Final_model, learning_type='BR', dynamic_model='point_mass_on_a_wheel')

		[globals()["Pred_Y_{0}_3".format(obj)], globals()["Pred_Y_fb_{0}_3".format(obj)], globals()["Actual_Y_{0}_3".format(obj)], globals()["error_{0}_3".format(obj)], globals()["error_fb_{0}_3".format(obj)], globals()["error_d_{0}_3".format(obj)], globals()["error_d_fb_{0}_3".format(obj)]] = test(globals()["X_test_{0}".format(obj)], globals()["U_test_{0}".format(obj)], globals()["r_F_test_{0}".format(obj)], globals()["dT_test_{0}".format(obj)], globals()["Y_test_{0}".format(obj)], Final_model_3, learning_type='BR', dynamic_model='friction_only')

	# Plot Results
	plot_results(Pred_Y_walker_2, Pred_Y_fb_walker_2, Actual_Y_walker, error_walker, error_fb_walker, error_walker_2, error_fb_walker_2, error_walker_3, error_fb_walker_3, Pred_Y_cart, Pred_Y_fb_cart, Actual_Y_cart, error_cart, error_fb_cart, error_cart_2, error_fb_cart_2, error_cart_3, error_fb_cart_3, Pred_Y_blue_chair, Pred_Y_fb_blue_chair, Actual_Y_blue_chair, error_blue_chair, error_fb_blue_chair, error_blue_chair_2, error_fb_blue_chair_2, error_blue_chair_3, error_fb_blue_chair_3, Pred_Y_gray_chair, Pred_Y_fb_gray_chair, Actual_Y_gray_chair, error_gray_chair, error_fb_gray_chair, error_gray_chair_2, error_fb_gray_chair_2, error_gray_chair_3, error_fb_gray_chair_3)
