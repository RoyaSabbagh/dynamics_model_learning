'''

This is the main function to read data, learn a model, test the model and report results.
Author: roya.sabbaghnovin@utah.edu

'''

# Import
from Learning import train, test
from sklearn import datasets
import numpy as np
from Data import read_data
from Plot_Results import plot_results


# target_object = "blue_chair"
#target_object = "gray_chair"
#target_object = "walker"
#target_object = "cart"
target_object = "walker_2"

# Load dataset
## Read train data from files
[X_train, U_train, r_F_train, dT_train, Y_train] = read_data(target_object, "train")

## Read test data from files
[X_test, U_test, r_F_test, dT_test, Y_test] = read_data(target_object, "test")

# Learn Model
Final_model = train(X_train, U_train, r_F_train, dT_train, Y_train, learning_type='BR',  dynamic_model="point_mass_on_a_wheel") #Options for learning_type are 'LR' or 'BR' and for dynamic_model are 'point_mass_on_a_wheel', 'friction_only' and '2_wheel'

# find test Results
[Pred_Y, Pred_Y_fb, Actual_Y, error, error_fb, error_d, error_d_fb] = test(X_test, U_test, r_F_test, dT_test, Y_test, Final_model, learning_type='BR',  dynamic_model="point_mass_on_a_wheel")

# Plot Results
#plot_results(Pred_Y, Pred_Y_fb, Actual_Y, error, dT_test)
