######################################## BM489E HW NO.1 ##############################################
##                                       ALi KARATANA                                               ##
##                                         121180043                                                ##
######################################################################################################


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math


float_formatter = lambda x: "%.3f" % x  # creating float formatter to use when showing float data
np.set_printoptions(formatter={'float_kind': float_formatter})  # setting up the formatter we created

data = np.loadtxt("magic_04.txt", delimiter=",", usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)) # getting data from .txt and creating matrix
print("OUr data------------------------\n",data)

col_number = np.size(data, axis=1)  # finding column number of dataset matrix
row_number = np.size(data, axis=0)  # finding row number of dataset matrix

mean_vector = np.mean(data, axis=0).reshape(col_number, 1)  # computing the mean vector
print("Mean Vector--------------------\n", mean_vector, "\n")
t_mean_vector = np.transpose(mean_vector)

centered_data_matrix = data - (1 * t_mean_vector)  # computing the centered data matrix
print("Centered Data Matrix-------------------\n", centered_data_matrix, "\n")

t_centered_data_matrix = np.transpose(centered_data_matrix) # computing the transpose of the centered data matrix

covariance_matrix_inner = (1 / row_number) * np.dot(t_centered_data_matrix, centered_data_matrix) # description below in print function as a string
print(
    "The sample covariance matrix as inner products between the columns of the centered data matrix ----------------------------------------\n",
    covariance_matrix_inner, "\n")


def sum_of_centered_points():    # finding the sum of centered data points
    sum = np.zeros(shape=(col_number, col_number))

    for i in range(0, row_number):
        sum += np.dot(np.reshape(t_centered_data_matrix[:, i], (-1, 1)),
                      np.reshape(centered_data_matrix[i, :], (-1, col_number)))
    return sum


covariance_matrix_outer = (1 / row_number) * sum_of_centered_points() # description below in print function as a string
print(
    "The sample covariance matrix as outer product between the centered data points ----------------------------------------\n",
    covariance_matrix_outer, "\n")

vector1 = np.array(centered_data_matrix[:, 1])
vector2 = np.array(centered_data_matrix[:, 2])


def unit_vector(vector):    # computing unit vectors for attribute vectors
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):   # calculating the angle between to attribute vectors
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


correlation = math.cos(angle_between(vector1, vector2))    # computing th correlation between attributes
print(" The correlation between Attributes 1 and 2: %.5f" % correlation, "\n")

variance_vector = np.var(data, axis=0)   # creating variance vector
max_var = np.max(variance_vector)        # finding max variance
min_var = np.min(variance_vector)        # finding min variance

for i in range(0, col_number):           # finding index of max variance
    if variance_vector[i] == max_var:
        max_var_index = i;

for i in range(0, col_number):           # finding index of min variance
    if variance_vector[i] == min_var:
        min_var_index = i

print(" Max variance = %.3f ( Attribute %d )" % (max_var, max_var_index))
print(" Min variance = %.3f ( Attribute %d )\n" % (min_var, min_var_index))

covariance_matrix = np.cov(data, rowvar=False)      # computing covariance matrix
max_cov=np.max(covariance_matrix)       # finding max value in covariance matrix
min_cov=np.min(covariance_matrix)       # finding min value in covariance matrix

for i in range(0, col_number):    # the loop to find the index of max and min values
    for j in range(0, col_number):
        if covariance_matrix[i, j] == max_cov:
            max_cov_atrr1=i
            max_cov_attr2=j

for i in range(0, col_number):   # the loop to find the index of max and min values
    for j in range(0, col_number):
        if covariance_matrix[i, j] == min_cov:
            min_cov_atrr1 = i
            min_cov_attr2 = j


print("Max Covariance = %.3f (Between Attribute %d and %d)" %(max_cov,max_cov_atrr1,max_cov_attr2))      # finding index of max covariance
print("Min Covariance = %.3f (Between Attribute %d and %d)\n" %(min_cov,min_cov_atrr1,min_cov_attr2))      # finding index of min covariance

df = pd.DataFrame(data[:, 1])   # creating data frame for plotting

plt.show(plt.scatter(data[:, 1], data[:, 2], c=("red", "yellow")))       # plotting the scatter plot between attributes
plt.show(df.plot(kind='density'))                # plotting probability density function

