

# Introduction to PyTorch

#___________________________________________________________________________________
##  Part I Introduction to PyTorch
#___________________________________________________________________________________

#1)Creating tensors in PyTorch
# Random tensors are very important in neural networks. Parameters of the neural networks typically are initialized with random weights (random tensors).
# Let us start practicing building tensors in PyTorch library. As you know, tensors are arrays with an arbitrary number of dimensions, corresponding to NumPy's ndarrays. You are going to create a random tensor of sizes 3 by 3 and set it to variable your_first_tensor. Then, you will need to print it. Finally, calculate its size in variable tensor_size and print its value.
# NB: In case you have trouble solving the problems, you can always refer to slides in the bottom right of the screen.
# Instructions
# Import PyTorch main library.
# Create the variable your_first_tensor and set it to a random torch tensor of size 3 by 3.
# Calculate its shape (dimension sizes) and set it to variable tensor_size.
# Print the values of your_first_tensor and tensor_size.

# Import torch
import torch
# Create random tensor of size 3 by 3
your_first_tensor = torch.rand(3, 3)
# Calculate the shape of the tensor
tensor_size = your_first_tensor.shape
# Print the values of the tensor and its shape
print(your_first_tensor)
print(tensor_size)


#2) Matrix multiplication
# There are many important types of matrices which have their uses in neural networks. 
# Some important matrices are matrices of ones (where each entry is set to 1) and the 
# identity matrix (where the diagonal is set to 1 while all other values are 0). 
# The identity matrix is very important in linear algebra: any matrix multiplied with 
# identity matrix is simply the original matrix.

# Let us experiment with these two types of matrices. You are going to build a matrix 
# of ones with shape 3 by 3 called tensor_of_ones and an identity matrix of the same shape,
# called identity_tensor. We are going to see what happens when we multiply these two 
# matrices, and what happens if we do an element-wise multiplication of them.

#Create a matrix of ones with shape 3 by 3, and put it on variable tensor_of_ones.
#Create an identity matrix with shape 3 by 3, and put it on variable identity_tensor.
#Do a matrix multiplication of tensor_of_ones with identity_tensor and print its value.
#Do an element-wise multiplication of tensor_of_ones with identity_tensor and print its value.

# Create a matrix of ones with shape 3 by 3
tensor_of_ones = torch.ones(3, 3)

# Create an identity matrix with shape 3 by 3
identity_tensor = torch.eye(3)

# Do a matrix multiplication of tensor_of_ones with identity_tensor
matrices_multiplied = torch.matmul(tensor_of_ones, identity_tensor)
print(matrices_multiplied)

# Do an element-wise multiplication of tensor_of_ones with identity_tensor
element_multiplication = tensor_of_ones * identity_tensor
print(element_multiplication)


#3)Forward pass
# Let's have something resembling more a neural network. The computational graph has been 
# given below. You are going to initialize 3 large random tensors, and then do the operations 
# as given in the computational graph. The final operation is the mean of the tensor, given 
# by torch.mean(your_tensor).

# Instructions
# Initialize random tensors x, y and z, each having shape (1000, 1000).
# Multiply x with y, putting the result in tensor q.
# Do an elementwise multiplication of tensor z with tensor q, putting the results in f

# Initialize tensors x, y and z
x = torch.rand(1000, 1000)
y = torch.rand(1000, 1000)
z = torch.rand(1000, 1000)
# Multiply x with y
q = x * y
# Multiply elementwise z with q
f = z * q
mean_f = torch.mean(f)
print(mean_f)


#4) Backpropagation by hand
# Possible Answers

# The Derivative of x is 5, the derivative of y is 5, the derivative of z is 1.
# correct one

# The Derivative of x is 5, the derivative of y is 5, the derivative of z is 5.

# The Derivative of x is 8, the derivative of y is -3, the derivative of z is 0.

# Derivatives are lame, integrals are cool.


#5)Backpropagation using PyTorch
# Here, you are going to use automatic differentiation of PyTorch in order to compute 
# the derivatives of x, y and z from the previous exercise.

# Instructions
# Initialize tensors x, y and z to values 4, -3 and 5.
#Put the sum of tensors x and y in q, put the product of q and z in f.
# Calculate the derivatives of the computational graph.
# Print the gradients of the x, y and z tensors.

# Initialize x, y and z to values 4, -3 and 5
x = torch.tensor(4., requires_grad=True)
y = torch.tensor(-3., requires_grad=True)
z = torch.tensor(5., requires_grad=True)
# Set q to sum of x and y, set f to product of q with z
q = x + y
f = q * z
# Compute the derivatives
f.backward()
# Print the gradients
print("Gradient of x is: " + str(x.grad))
print("Gradient of y is: " + str(y.grad))
print("Gradient of z is: " + str(z.grad))


#6)Calculating gradients in PyTorch
# Remember the exercise in forward pass? Now that you know how to calculate derivatives, 
# let's make a step forward and start calculating the gradients (derivatives of tensors) 
# of the computational graph you built back then. We have already initialized for you three 
# random tensors of shape (1000, 1000) called x, y and z. 
# First, we multiply tensors x and y, then we do an elementwise multiplication of their product
# with tensor z, and then we compute its mean. In the end, we compute the derivatives.

# The main difference from the previous exercise is the scale of the tensors. 
# While before, tensors x, y and z had just 1 number, now they each have 1 million numbers.

# Instructions (already initialized)
# Multiply tensors x and y, put the product in tensor q.
# Do an elementwise multiplication of tensors z with q.
# Calculate the gradients.
# Multiply tensors x and y
# q = torch.matmul(x, y)
# Elementwise multiply tensors z with q
# f = z * q
# mean_f = torch.mean(f)
# Calculate the gradients
# mean_f.backward()


#7) Your first neural network
# You are going to build a neural network in PyTorch, using the hard way. 
# Your input will be images of size (28, 28), so images containing 784 pixels. 
# Your network will contain an input_layer (provided for you), a hidden layer with 200 units, 
# and an output layer with 10 classes. The input layer has already been created for you. 
# You are going to create the weights, and then do matrix multiplications, getting the results 
# from the network.

# Instructions
# Initialize with random numbers two matrices of weights, called weight_1 and weight_2.
# Set the result of input_layer times weight_1 to hidden_1. Set the result of hidden_1 times 
# weight_2 to output_layer.

# Initialize the weights of the neural network
weight_1 = torch.rand(784, 10)
weight_2 = torch.rand(10, 200)
input_layer = torch.tensor([0.4087, 0.1997, 0.4769, 0.4709, 0.4936, 0.8699, 0.5515, 0.1656, 0.6059,
        0.8637, 0.9249, 0.3971, 0.4959, 0.8645, 0.0441, 0.8361, 0.5468, 0.6230,
        0.7156, 0.5483, 0.7324, 0.4919, 0.7736, 0.8201, 0.4088, 0.8573, 0.4806,
        0.6304, 0.6929, 0.9365, 0.5100, 0.7313, 0.8077, 0.0041, 0.5494, 0.3742,
        0.0825, 0.1222, 0.5785, 0.8966, 0.6954, 0.8519, 0.8770, 0.1329, 0.5745,
        0.0668, 0.3913, 0.0817, 0.0822, 0.9487, 0.4057, 0.7227, 0.5898, 0.5573,
        0.5351, 0.7718, 0.6524, 0.6225, 0.7171, 0.7160, 0.1205, 0.1004, 0.7437,
        0.1252, 0.3028, 0.0524, 0.9171, 0.8566, 0.9846, 0.5991, 0.1867, 0.4677,
        0.1099, 0.6558, 0.5568, 0.2086, 0.2061, 0.1903, 0.1728, 0.5155, 0.4852,
        0.5820, 0.8144, 0.2209, 0.0446, 0.3139, 0.5663, 0.1442, 0.7929, 0.8366,
        0.2016, 0.4877, 0.9757, 0.7782, 0.3551, 0.6471, 0.0597, 0.4101, 0.5364,
        0.1393, 0.2610, 0.5672, 0.6615, 0.8876, 0.1534, 0.1249, 0.5920, 0.4861,
        0.1060, 0.7428, 0.4631, 0.8913, 0.5201, 0.1549, 0.5788, 0.4665, 0.4266,
        0.9622, 0.8182, 0.9057, 0.4761, 0.2751, 0.6738, 0.3420, 0.0207, 0.4382,
        0.8040, 0.0240, 0.8293, 0.8188, 0.1581, 0.0906, 0.6927, 0.9537, 0.5079,
        0.1537, 0.9020, 0.8779, 0.7824, 0.4996, 0.2310, 0.7874, 0.1343, 0.1197,
        0.5376, 0.5261, 0.4303, 0.8152, 0.2990, 0.8618, 0.9347, 0.2527, 0.9238,
        0.1558, 0.9855, 0.1489, 0.2282, 0.2997, 0.3835, 0.4175, 0.1890, 0.1764,
        0.0346, 0.8769, 0.3371, 0.5508, 0.5341, 0.2268, 0.1378, 0.6518, 0.9058,
        0.3591, 0.6751, 0.2383, 0.2342, 0.4190, 0.2637, 0.0521, 0.5546, 0.2108,
        0.2845, 0.9217, 0.2804, 0.3842, 0.9855, 0.4167, 0.7223, 0.8035, 0.6947,
        0.2993, 0.2005, 0.6404, 0.1998, 0.0355, 0.8519, 0.2023, 0.7048, 0.6818,
        0.9605, 0.0336, 0.2138, 0.6745, 0.9008, 0.7409, 0.0826, 0.8648, 0.2849,
        0.8509, 0.8815, 0.0866, 0.9190, 0.7103, 0.5046, 0.0785, 0.1618, 0.7292,
        0.7089, 0.4477, 0.0879, 0.7692, 0.2794, 0.4137, 0.0748, 0.0174, 0.3932,
        0.7290, 0.9742, 0.2320, 0.1189, 0.5049, 0.5384, 0.5752, 0.9857, 0.1370,
        0.2943, 0.8934, 0.9695, 0.5466, 0.0747, 0.7343, 0.5883, 0.7226, 0.4341,
        0.2567, 0.6041, 0.0955, 0.9769, 0.1448, 0.4154, 0.3164, 0.2974, 0.2944,
        0.1425, 0.5457, 0.3341, 0.6386, 0.8526, 0.6114, 0.4141, 0.9997, 0.6396,
        0.7283, 0.6342, 0.8659, 0.8241, 0.6825, 0.4625, 0.1645, 0.6022, 0.5037,
        0.0986, 0.6231, 0.4962, 0.8049, 0.2839, 0.6977, 0.3071, 0.1337, 0.2064,
        0.3599, 0.4253, 0.6585, 0.5649, 0.3123, 0.9926, 0.7426, 0.1429, 0.9566,
        0.9471, 0.7524, 0.5857, 0.7152, 0.8081, 0.1825, 0.0584, 0.9914, 0.8519,
        0.9394, 0.0818, 0.1600, 0.2165, 0.7708, 0.6624, 0.7142, 0.6240, 0.3641,
        0.9761, 0.0450, 0.6984, 0.8463, 0.1695, 0.8905, 0.5969, 0.9112, 0.6867,
        0.9798, 0.5436, 0.8189, 0.5066, 0.9602, 0.7946, 0.6999, 0.0381, 0.5073,
        0.8515, 0.3367, 0.6397, 0.9555, 0.8126, 0.5168, 0.7044, 0.7861, 0.2718,
        0.3028, 0.6037, 0.9717, 0.6349, 0.2758, 0.0444, 0.5958, 0.5212, 0.0757,
        0.2268, 0.7862, 0.5820, 0.0231, 0.6945, 0.3756, 0.7973, 0.3090, 0.2278,
        0.0300, 0.8427, 0.4903, 0.3304, 0.8507, 0.7588, 0.8737, 0.8488, 0.4428,
        0.5040, 0.8161, 0.4690, 0.3782, 0.7344, 0.9206, 0.3064, 0.7079, 0.7518,
        0.7286, 0.1645, 0.2736, 0.2468, 0.9494, 0.1933, 0.9416, 0.2884, 0.2299,
        0.2048, 0.9035, 0.0619, 0.1400, 0.9010, 0.1083, 0.5329, 0.0815, 0.1013,
        0.8091, 0.6830, 0.5867, 0.6425, 0.7093, 0.3875, 0.1312, 0.7361, 0.2282,
        0.2185, 0.1382, 0.6745, 0.3547, 0.3806, 0.7362, 0.5328, 0.2599, 0.5914,
        0.0305, 0.8225, 0.0944, 0.5973, 0.7927, 0.8478, 0.0334, 0.7768, 0.6117,
        0.6160, 0.1170, 0.5759, 0.2825, 0.4012, 0.1751, 0.4239, 0.1643, 0.5133,
        0.4622, 0.9612, 0.8385, 0.8982, 0.7341, 0.3731, 0.2795, 0.4386, 0.5728,
        0.5302, 0.5069, 0.7485, 0.7447, 0.5230, 0.0732, 0.6534, 0.9379, 0.9208,
        0.8895, 0.2467, 0.8365, 0.7269, 0.4686, 0.9635, 0.3148, 0.3255, 0.5113,
        0.7892, 0.9105, 0.3757, 0.2983, 0.2517, 0.5445, 0.2168, 0.7303, 0.9069,
        0.0758, 0.7940, 0.0420, 0.1270, 0.6238, 0.4400, 0.6372, 0.1418, 0.4939,
        0.4792, 0.0517, 0.1842, 0.8044, 0.7751, 0.9392, 0.7418, 0.3370, 0.2232,
        0.7890, 0.5724, 0.4802, 0.8122, 0.6218, 0.8626, 0.8842, 0.2538, 0.1288,
        0.7903, 0.9914, 0.2069, 0.2344, 0.6561, 0.7509, 0.7539, 0.8282, 0.8231,
        0.0636, 0.3983, 0.9733, 0.8925, 0.8159, 0.7264, 0.7670, 0.4797, 0.2713,
        0.6417, 0.7108, 0.9323, 0.6879, 0.0832, 0.1254, 0.5788, 0.5314, 0.8200,
        0.5556, 0.6042, 0.7068, 0.5495, 0.0525, 0.4970, 0.4178, 0.7047, 0.5613,
        0.7910, 0.6168, 0.2146, 0.9594, 0.2098, 0.2194, 0.3575, 0.8423, 0.5420,
        0.9070, 0.8532, 0.9862, 0.8242, 0.0726, 0.4063, 0.1403, 0.1846, 0.4290,
        0.9815, 0.7456, 0.4768, 0.4513, 0.8797, 0.3593, 0.9641, 0.6871, 0.4184,
        0.5268, 0.9882, 0.5236, 0.6552, 0.5485, 0.0163, 0.7818, 0.5353, 0.2730,
        0.1333, 0.0082, 0.9015, 0.3494, 0.5273, 0.6121, 0.3223, 0.7258, 0.1785,
        0.1108, 0.5567, 0.7655, 0.2080, 0.7048, 0.1930, 0.3902, 0.2976, 0.0143,
        0.8721, 0.8625, 0.7449, 0.8929, 0.5287, 0.8267, 0.3062, 0.7486, 0.9272,
        0.6238, 0.6148, 0.1178, 0.9361, 0.9397, 0.8782, 0.2020, 0.7409, 0.5296,
        0.7618, 0.5381, 0.9244, 0.9842, 0.1871, 0.6271, 0.0423, 0.4588, 0.3982,
        0.1031, 0.8504, 0.8672, 0.9028, 0.4920, 0.3841, 0.9631, 0.6674, 0.3871,
        0.6497, 0.4594, 0.2951, 0.9417, 0.6727, 0.8998, 0.4110, 0.8827, 0.1578,
        0.5702, 0.8692, 0.7678, 0.0135, 0.5735, 0.9355, 0.1671, 0.3242, 0.5710,
        0.6307, 0.0631, 0.6796, 0.1538, 0.1980, 0.1402, 0.7507, 0.2236, 0.0223,
        0.2001, 0.4889, 0.8540, 0.2156, 0.4319, 0.0780, 0.7426, 0.3724, 0.8992,
        0.1358, 0.4117, 0.0634, 0.3825, 0.3535, 0.6583, 0.7189, 0.0158, 0.7902,
        0.5463, 0.8652, 0.5157, 0.1718, 0.6729, 0.3937, 0.7248, 0.4058, 0.8826,
        0.7427, 0.6369, 0.0346, 0.1368, 0.0688, 0.0308, 0.2464, 0.2408, 0.4806,
        0.6114, 0.1931, 0.8346, 0.4002, 0.1442, 0.5597, 0.4541, 0.6105, 0.9752,
        0.6950, 0.2729, 0.2677, 0.3533, 0.5120, 0.9714, 0.7756, 0.2931, 0.0952,
        0.2616, 0.5266, 0.6267, 0.4118, 0.3918, 0.4945, 0.4323, 0.1562, 0.7972,
        0.5842, 0.4449, 0.9630, 0.3089, 0.5482, 0.6561, 0.8684, 0.4669, 0.4765,
        0.3126, 0.5595, 0.8834, 0.9500, 0.4904, 0.9767, 0.2860, 0.1644, 0.1796,
        0.7509, 0.5943, 0.7843, 0.8429, 0.1072, 0.5543, 0.7442, 0.5912, 0.3650,
        0.4735, 0.6901, 0.2902, 0.1465, 0.5502, 0.4275, 0.0736, 0.6347, 0.3760,
        0.6535, 0.3149, 0.5881, 0.8508, 0.9198, 0.1839, 0.6380, 0.3136, 0.7573,
        0.5169, 0.4649, 0.6362, 0.2776, 0.8661, 0.2827, 0.4443, 0.5030, 0.9502,
        0.5843, 0.8689, 0.3207, 0.0045, 0.1192, 0.7140, 0.6612, 0.8260, 0.4318,
        0.9792, 0.4161, 0.6854, 0.4569, 0.8272, 0.4222, 0.9866, 0.9125, 0.1910,
        0.3976, 0.2391, 0.4065, 0.7811, 0.9730, 0.1828, 0.6999, 0.5569, 0.8982,
        0.5463])
# Multiply input_layer with weight_1
hidden_1 = torch.matmul(input_layer, weight_1) # input_layer given
# Multiply hidden_1 with weight_2
output_layer = torch.matmul(hidden_1, weight_2)
print(output_layer)


#8)Your first PyTorch neural network
# You are going to build the same neural network you built in the previous exercise, 
# but now using the PyTorch way. As a reminder, you have 784 units in the input layer, 
# 200 hidden units and 10 units for the output layer.

# Instructions
# Instantiate two linear layers calling them self.fc1 and self.fc2. 
# Determine their correct dimensions.
# Implement the .forward() method, using the two layers you defined and returning x.

import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # Instantiate all 2 linear layers  
        self.fc1 = nn.Linear(784, 200)
        self.fc2 = nn.Linear(200, 10)

    def forward(self, x):
      
        # Use the instantiated layers and return x
        x = self.fc1(x)
        x = self.fc2(x)
        return x


#___________________________________________________________________________________
##  Part II Artificial Neural Networks
#___________________________________________________________________________________

 
#1)Neural networks
# Let us see the differences between neural networks which apply ReLU and those which 
# do not apply ReLU. We have already initialized the input called input_layer, and 
# three sets of weights, called weight_1, weight_2 and weight_3.

# We are going to convince ourselves that networks with multiple layers which do not 
# contain non-linearity can be expressed as neural networks with one layer.
# The network and the shape of layers and weights is shown below.

# Instructions
# Calculate the first and second hidden layer by multiplying the appropriate inputs 
# with the corresponding weights.
# Calculate and print the results of the output.
# Set weight_composed_1 to the product of weight_1 with weight_2, then set weight to 
# the product of weight_composed_1 with weight_3.
# Calculate and print the output.

# Data provided
input_layer = torch.tensor([[ 0.0401, -0.9005,  0.0397, -0.0876]])
weight_1 = torch.tensor([[-0.1094, -0.8285,  0.0416, -1.1222],
        [ 0.3327, -0.0461,  1.4473, -0.8070],
        [ 0.0681, -0.7058, -1.8017,  0.5857],
        [ 0.8764,  0.9618, -0.4505,  0.2888]])
weight_2 = torch.tensor([[ 0.6856, -1.7650,  1.6375, -1.5759],
        [-0.1092, -0.1620,  0.1951, -0.1169],
        [-0.5120,  1.1997,  0.8483, -0.2476],
        [-0.3369,  0.5617, -0.6658,  0.2221]])
weight_3 = torch.tensor([[ 0.8824,  0.1268,  1.1951,  1.3061],
        [-0.8753, -0.3277, -0.1454, -0.0167],
        [ 0.3582,  0.3254, -1.8509, -1.4205],
        [ 0.3786,  0.5999, -0.5665, -0.3975]])      
# Calculate the first and second hidden layer
hidden_1 = torch.matmul(input_layer, weight_1)
hidden_2 = torch.matmul(hidden_1, weight_2)
# Calculate the output
print(torch.matmul(hidden_2, weight_3))
# Calculate weight_composed_1 and weight
weight_composed_1 = torch.matmul(weight_1, weight_2)
weight = torch.matmul(weight_composed_1, weight_3)
# Multiply input_layer with weight
print(torch.matmul(input_layer, weight))



#2) ReLU activation
# In this exercise, we have the same settings as the previous exercise. 
# In addition, we have instantiated the ReLU activation function called relu().

# Now we are going to build a neural network which has non-linearity and by doing so, 
# we are going to convince ourselves that networks with multiple layers and non-linearity 
# functions cannot be expressed as a neural network with one layer.

##   Instructions (using previous exercise data)
# Apply non-linearity on hidden_1 and hidden_2.
# Apply non-linearity in the product of first two weight.
# Multiply the result of the previous step with weight_3.
# Multiply input_layer with weight and print the results.
import torch.nn as nn
relu = nn.ReLU()
# Apply non-linearity on hidden_1 and hidden_2
hidden_1_activated = relu(torch.matmul(input_layer, weight_1))
hidden_2_activated = relu(torch.matmul(hidden_1_activated, weight_2))
print(torch.matmul(hidden_2_activated, weight_3))
# Apply non-linearity in the product of first two weights. 
weight_composed_1_activated = relu(torch.matmul(weight_1, weight_2))
# Multiply `weight_composed_1_activated` with `weight_3
weight = torch.matmul(weight_composed_1_activated, weight_3)
# Multiply input_layer with weight
print(torch.matmul(input_layer, weight))



#3)ReLU activation again (create NN with different number of neurons in each layer)
# Neural networks don't need to have the same number of units in each layer. 
# Here, you are going to experiment with the ReLU activation function again, 
# but this time we are going to have a different number of units in the layers 
# of the neural network. The input layer will still have 4 features, but then 
# the first hidden layer will have 6 units and the output layer will have 2 units.

# Instructions
# Instantiate the ReLU() activation function as relu (the function is part of nn module).
# Initialize weight_1 and weight_2 with random numbers.
# Multiply the input_layer with weight_1, storing results in hidden_1.
# Apply the relu activation function over hidden_1, and then multiply the output of it with
# weight_2.

# Instantiate ReLU activation function as relu
relu = nn.ReLU()
# Initialize weight_1 and weight_2 with random numbers
weight_1 = torch.rand(4, 6)
weight_2 = torch.rand(6, 2)
# Multiply input_layer with weight_1
hidden_1 = torch.matmul(input_layer, weight_1)
# Apply ReLU activation function over hidden_1 and multiply with weight_2
hidden_1_activated = relu(hidden_1)
print(torch.matmul(hidden_1_activated, weight_2))


#4) Calculating loss function by hand
# Let's start the exercises by calculating the loss function by hand. Don't do this 
# exercise in PyTorch, it is important to first do it using only pen and paper 
# (and a calculator).

# We have the same example as before but now our object is actually a frog, and the 
# predicted scores are -1.2 for class 0 (cat), 0.12 for class 1 (car) and 4.8 for 
# class 2 (frog).

# What is the result of the softmax cross-entropy loss function?
# Class	Predicted Score
# Cat	-1.2
# Car	0.12
# Frog	4.8

# solution
from numpy import log as ln
from scipy import exp
e_cat = exp(-1.2)
e_car = exp(0.12)
e_frog = exp(4.8)
# Probability estimation, 
prob_cat = e_cat / (e_cat + e_car + e_frog)
print(prob_cat)
prob_car = e_car / (e_cat + e_car + e_frog)
print(prob_car)
prob_frog = e_frog / (e_cat + e_car + e_frog)
print(prob_frog)
# Logarithmic calculations
log_pcat = -ln(prob_cat); print('logpcat :', log_pcat)
log_pcar = -ln(prob_car); print('logpcar :', log_pcar)
log_pfrog = -ln(prob_frog); print('logpfrog :', log_pfrog)

# Possible Answers

# 6.0117

# 4.6917

# 0.0117 (correct one)

# Score for frog is high, so loss is 0.


#5) Calculating loss function in PyTorch
# You are going to code the previous exercise, and make sure that we computed the loss correctly. Predicted scores are -1.2 for class 0 (cat), 0.12 for class 1 (car) and 4.8 for class 2 (frog). The ground truth is class 2 (frog). Compute the loss function in PyTorch.
#Class	Predicted Score
# Cat	-1.2
# Car	0.12
# Frog	4.8

#Instructions
# Initialize the tensor of scores with numbers [[-1.2, 0.12, 4.8]], and the tensor of 
# ground truth [2].
# Instantiate the cross-entropy loss and call it criterion.
# Compute and print the loss.

# Initialize the scores and ground truth
logits = torch.tensor([[-1.2, 0.12, 4.8]])
ground_truth = torch.tensor([2])
# Instantiate cross entropy loss
criterion = nn.CrossEntropyLoss()
# Compute and print the loss
loss = criterion(logits, ground_truth)
print(loss)


#6) Loss function of random scores
# If the neural network predicts random scores, what would be its loss function? 
# Let's find it out in PyTorch. The neural network is going to have 1000 classes, 
# each having a random score. For ground truth, it will have class 111. 
# Calculate the loss function.

# Instructions
# Import torch and torch.nn as nn
# Initialize logits with a random tensor of shape (1, 1000) and ground_truth with a 
# tensor containing the number 111.
# Instantiate the cross-entropy loss in a variable called criterion.
# Calculate and print the loss function.

# Import torch and torch.nn
import torch
import torch.nn as nn
# Initialize logits and ground truth
logits = torch.rand(1,1000)
ground_truth = torch.tensor([111])
# Instantiate cross-entropy loss
criterion = nn.CrossEntropyLoss()
# Calculate and print the loss
loss = criterion(logits, ground_truth)
print(loss)



#7) Preparing MNIST dataset
# You are going to prepare dataloaders for MNIST training and testing set. 
# As we explained in the lecture, MNIST has some differences to CIFAR-10, with the main 
# difference being that MNIST images are grayscale (1 channel based) instead of RGB 
# (3 channels).

# Instructions
# Transform the data to torch tensors and normalize it, mean is 0.1307 while std is 0.3081.
# Prepare the trainset and the testset.
# Prepare the dataloaders for training and testing so that only 32 pictures are processed 
# at a time.

# loading configutarion needed
import torch
import torchvision
import torch.utils.data
import torchvision.transforms as transforms

# Transform the data to torch tensors and normalize it 
transform = transforms.Compose([transforms.ToTensor(),
								transforms.Normalize((0.1307), ((0.3081)))])
# Prepare training set and testing set
trainset = torchvision.datasets.MNIST('mnist', train=True, 
									  download=True, transform=transform)
testset = torchvision.datasets.MNIST('mnist', train=False, 
									  download=True, transform=transform)
# Prepare training loader and testing loader
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                          shuffle=True, num_workers=0)
testloader = torch.utils.data.DataLoader(testset, batch_size=32,
										 shuffle=False, num_workers=0)

print('trainloader :', trainloader)
print('testloader :', testloader)

#Nice! Now you know how to prepare datasets in PyTorch. 
# You are very close to reaching the grand goal, training neural networks!



#8)Inspecting the dataloaders
# Now you are going to explore a bit the dataloaders you created in the previous exercise. 
# In particular, you will compute the shape of the dataset in addition to the minibatch size.

# Instructions
# Compute the shapes of the trainset and testset.
# Print the computed values.
# Compute the size of the minibatch for both trainset and testset.
# Print the minibatch size.

# Compute the shape of the training set and testing set
trainset_shape = trainloader.dataset.train_data.shape
testset_shape = testloader.dataset.test_data.shape
# Print the computed shapes
print(trainset_shape, testset_shape)
# Compute the size of the minibatch for training set and testing set
trainset_batchsize = trainloader.batch_size
testset_batchsize = testloader.batch_size
# Print sizes of the minibatch
print(trainset_batchsize, testset_batchsize)



#9) Training NN
# Building a neural network - again
# You haven't created a neural network since the end of the first chapter, 
# so this is a good time to build one (practice makes perfect). 
# Build a class for a neural network which will be used to train on the MNIST dataset. 
# The dataset contains images of shape (28, 28, 1), so you should deduct the size of 
# the input layer. 
# For hidden layer use 200 units, while for output layer use 10 units (1 for each class). 
# For activation function, use relu in a functional way (nn.Functional is already imported as F).

# For context, the same net will be trained and used to make predictions in the next 
# two exercises.

# Instructions
# Define the class called Net which inherits from nn.Module.
# In the __init__() method, define the parameters for the two fully connected layers.
# In the .forward() method, do the forward step

#F = nn.Functional()
import torch.nn.functional as F
# Define the class Net
class Net(nn.Module):
    def __init__(self):    
    	# Define all the parameters of the net
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28 * 1, 200)
        self.fc2 = nn.Linear(200, 10)

    def forward(self, x):   
    	# Do the forward pass
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x



#10)Training a neural network
# Given the fully connected neural network (called model) which you built in the previous 
# exercise and a train loader called train_loader containing the MNIST dataset 
# (which we created for you), you're to train the net in order to predict the classes of digits. 
# You will use the Adam optimizer to optimize the network, and considering that this is a 
# classification problem you are going to use cross entropy as loss function.

# Instructions
# Instantiate the Adam optimizer with learning rate 3e-4 and instantiate Cross-Entropy as 
# loss function.
# Complete a forward pass on the neural network using the input data.
# Using backpropagation, compute the gradients of the weights, and then change the weights 
# using the Adam optimizer.
'''
import torch_optimizer as optim

# Instantiate the Adam optimizer and Cross-Entropy loss function
model = Net()
# Adam optimizer gives error. options: Adamp, AdamW...  
optimizer = optim.AdamP(model.parameters(), lr=3e-4)  
criterion = nn.CrossEntropyLoss()
# FROM THE PREVIOUS EXERCISE, NOT THE DATACAMP FILE.
train_loader = trainloader

for batch_idx, data_target in enumerate(train_loader):
    data = data_target[0]
    target = data_target[1]
    data = data.view(-1, 28 * 28)
    optimizer.zero_grad()

    # Complete a forward pass
    output = model(data)

    # Compute the loss, gradients and change the weights
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
'''

#11)Using the network to make predictions
# Now that you have trained the network, use it to make predictions for the data in the testing set. 
# The network is called model (same as in the previous exercise), and the loader is called test_loader. 
# We have already initialized variables total and correct to 0.

# Instructions
# Set the network in testing (eval) mode.
# Put each image into a vector using inputs.view(-1, number_of_features) where the number of features 
# should be deducted by multiplying spatial dimensions (shape) of the image.
# Do the forward pass and put the predictions in output variable.
'''
# Set the model in eval mode
test_loader = testloader # not the real data of datacamp
correct, total = 0, 0
predictions = []
model.eval()

for i, data in enumerate(test_loader, 0):
    inputs, labels = data
    
    # Put each image into a vector
    inputs = inputs.view(-1, 28 * 28) # for an image of 28*28 pixeles
    
    # Do the forward pass and get the predictions
    outputs = model(inputs)
    _, outputs = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (outputs == labels).sum().item()

# gives 95 % with this data    
print('The testing set accuracy of the network is: %d %%' % (100 * correct / total))
'''
# <script.py> output:
#    The testing set accuracy of the network is: 82 %

# Congratulations, you are able to use neural networks to make predictions. 
# NB: The accuracy of the net is too low compared to what we can do with neural networks. 
# We used only a subpart of the dataset in order for the training to happen fast. 
# You can achieve a much higher accuracy by using the entire dataset, and by using a larger neural network.


#___________________________________________________________________________________
##  Part III Convolutional Neural Networks (CNNs)
#___________________________________________________________________________________

 
#1)Convolution operator - OOP way
# Let's kick off this chapter by using convolution operator from the torch.nn package. 
# You are going to create a random tensor which will represent your image and random 
# filters to convolve the image with. Then you'll apply those images.
# The torch library and the torch.nn package have already been imported for you.

# Instructions
# Create 10 images with shape (1, 28, 28).
# Build 6 convolutional filters of size (3, 3) with stride set to 1 and padding set to 1.
# Apply the filters in the image and print the shape of the feature map. 

#Create 10 random images of shape (1, 28, 28)
images = torch.rand(10, 1, 28, 28)
# Build 6 conv. filters
conv_filters = torch.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3, stride=1, padding=1)
# Convolve the image with the filters 
output_feature = conv_filters(images)
print('CNN OOP way:', output_feature.shape)


#2) Convolution operator - Functional way
# While I and most of PyTorch practitioners love the torch.nn package (OOP way), other practitioners
#  prefer building neural network models in a more functional way, using torch.nn.functional. 
# More importantly, it is possible to mix the concepts and use both libraries at the same time 
# (we have already done it in the previous chapter). You are going to build the same neural network 
# you built in the previous exercise, but this time using the functional way.

# As before, we have already imported the torch library and torch.nn.functional as F.

# Instructions
# Create 10 random images with shape (1, 28, 28).
# Create 6 random filters with shape (1, 3, 3).
# Convolve the images with the filters.

# Create 10 random images
image = torch.rand(10, 1, 28, 28)
# Create 6 filters
filters = torch.rand(6, 1, 3, 3)
# Convolve the image with the filters
output_feature = F.conv2d(image, filters, stride=1, padding=1)
print('CNN Functional way:', output_feature.shape)


#3)Max-pooling operator
# Here, you are going to practice using max-pooling in both OOP and functional way, 
# and see for yourself that the produced results are the same. We have already created 
# and printed the image for you, and imported torch library in addition to torch.nn and 
# torch.nn.Functional as F packages.

# Instructions
# Build a max-pooling operator with size 2.
# Apply the max-pooling operator in the image (loaded as im).
# Use a max-pooling operator in functional way in the image.
# Print the results of both cases.

im = torch.tensor([[[[ 8.,  1.,  2.,  5.,  3.,  1.],
          [ 6.,  0.,  0., -5.,  7.,  9.],
          [ 1.,  9., -1., -2.,  2.,  6.],
          [ 0.,  4.,  2., -3.,  4.,  3.],
          [ 2., -1.,  4., -1., -2.,  3.],
          [ 2., -4.,  5.,  9., -7.,  8.]]]])

# Build a pooling operator with size `2`.
max_pooling = torch.nn.MaxPool2d(2)
# Apply the pooling operator
output_feature = max_pooling(im)
# Use pooling operator in the image
output_feature_F = F.max_pool2d(im, 2)
# print the results of both cases
print(output_feature)
print(output_feature_F)


#4)Average-pooling operator
# After coding the max-pooling operator, you are now going to code the average-pooling operator. 
# You just need to replace max-pooling with average pooling.

# Instructions
# Build an average-pooling operator with size 2.
# Apply the average-pooling operator in the image.
# Use an average-pooling operator in functional way in the image, called im.
# Print the results of both cases.

# Build a pooling operator with size `2`.
avg_pooling = torch.nn.AvgPool2d(2)
# Apply the pooling operator
output_feature = avg_pooling(im)
# Use pooling operator in the image
output_feature_F = F.avg_pool2d(im, 2)
# print the results of both cases
print(output_feature)
print(output_feature_F)


#5) Your first CNN - __init__ method
# You are going to build your first convolutional neural network. You're going to use the MNIST 
# dataset as the dataset, which is made of handwritten digits from 0 to 9. The convolutional 
# neural network is going to have 2 convolutional layers, each followed by a ReLU nonlinearity, 
# and a fully connected layer. We have already imported torch and torch.nn as nn. 
# Remember that each pooling layer halves both the height and the width of the image, so by using 2 pooling layers, the height and width are 1/4 of the original sizes. MNIST images have shape (1, 28, 28)

# For the moment, you are going to implement the __init__ method of the net. In the next exercise, 
# you will implement the .forward() method.

# NB: We need 2 pooling layers, but we only need to instantiate a pooling layer once, because each 
# pooling layer will have the same configuration. Instead, we will use self.pool twice in the next exercise.

# Instructions
# Instantiate two convolutional filters: the first one should have 5 channels, while the second one 
# should have 10 channels. The kernel_size for both of them should be 3, and both should use padding=1. 
# Use the names of the arguments (instead of using 1, use padding=1).
# Instantiate a ReLU() nonlinearity.
# Instantiate a max pooling layer which halves the size of the image in both directions.
# Instantiate a fully connected layer which connects the units with the number of classes 
# (we are using MNIST, so there are 10 classes).

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Instantiate two convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=5, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=5, out_channels=10, kernel_size=3, padding=1)
        # Instantiate the ReLU nonlinearity
        self.relu = nn.ReLU()
        # Instantiate a max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        # Instantiate a fully connected layer
        self.fc = nn.Linear(7 * 7 * 10, 10)

# Great job! Now all that remains is implementing the forward() method and you have your first CNN.



#6)Your first CNN - forward() method

# Now that you have declared all the parameters of your CNN, all you need to do is to implement the 
# net's forward() method, and voila, you have your very first PyTorch CNN.

# Note: for evaluation purposes, the entire code of the class needs to be in the script. 
# We are using the __init__ method as you have coded it on the previous exercise, while you are 
# going to code the .forward() method here.

# Instructions
# Apply the first convolutional layer, followed by the relu nonlinearity, then in the next line apply 
# max-pooling layer.
# Apply the second convolutional layer, followed by the relu nonlinearity, then in the next line apply 
# max-pooling layer.
# Transform the feature map from 4 dimensional to 2 dimensional space. The first dimension contains 
# the batch size (-1), deduct the second dimension, by multiplying the values for height, width and depth.
# Apply the fully-connected layer and return the result.

class Net(nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()
		
        # Instantiate the ReLU nonlinearity
        self.relu = nn.ReLU()
        
        # Instantiate two convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=5, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=5, out_channels=10, kernel_size=3, padding=1)
        
        # Instantiate a max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Instantiate a fully connected layer
        self.fc = nn.Linear(7 * 7 * 10, 10)

    def forward(self, x):

        # Apply conv followed by relu, then in next line pool
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        # Apply conv followed by relu, then in next line pool
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        # Prepare the image for the fully connected layer
        x = x.view(-1, 7 * 7 * 10)
        # Apply the fully connected layer and return the result
        return self.fc(x)


#7)Training CNNs
# Similarly to what you did in Chapter 2, you are going to train a neural network. 
# This time however, you will train the CNN you built in the previous lesson, instead 
# of a fully connected network. The packages you need have been imported for you and 
# the network (called net) instantiated. The cross-entropy loss function (called criterion) 
# and the Adam optimizer (called optimizer) are also available. We have subsampled the 
# training set so that the training goes faster, and you are going to use a single epoch.

# Instructions
# Compute the predictions from the net.
# Using the predictions and the labels, compute the loss function.
# Compute the gradients for each weight.
# Update the weights using the optimizer.
'''
# not working properly, net was given in datacamp exercise.
# probable nn.Net initianization needed.

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
#import torch_optimizer as optim
import torch.optim as optim
train_loader = trainloader # not the real data of datacamp
net = Net()
optimizer = optim.AdamP(net.parameters(), lr=3e-4)
criterion = nn.CrossEntropyLoss()

for i, data in enumerate(train_loader, 0):
    inputs, labels = data
    optimizer.zero_grad()

    # Compute the forward pass
    outputs = net(inputs)
    # Compute the loss function, the predictions are the predicted outputs
    loss = criterion(outputs, labels)
    # Compute the gradients
    loss.backward()
    # Update the weights
    optimizer.step()
'''


#8)Using CNNs to make predictions (checking the net predictions)
# Building and training neural networks is a very exciting job (trust me, I do it every day)! 
# However, the main utility of neural networks is to make predictions. This is the entire reason 
# why the field of deep learning has bloomed in the last few years, as neural networks predictions 
# are extremely accurate. 
# On this exercise, we are going to use the convolutional neural network you already trained in order 
# to make predictions on the MNIST dataset.

# Remember that torch.max() takes two arguments: -output.data - the tensor which contains the data.
# Either 1 to do argmax or 0 to do max.

#Instructions
# Iterate over the given test_loader, saving the results of each iteration in data.
# Get the image and label from the data tuple, storing the results in image and label.
# Make a forward pass in the net using your image.
# Get the net prediction using torch.max() function.

''' # comes from previous part, whis has missing data.
#import torch_optimizer as optim
import torch.optim as optim
test_loader = testloader # not the real data of datacamp
# Iterate over the data in the test_loader
for i, data in enumerate(test_loader):
    # Get the image and label from data
    image, label = data
    # Make a forward pass in the net with your image
    output = net(image)
    # Argmax the results of the net
    _, predicted = torch.max(output.data, 1)
    if predicted == label:
        print("Yipes, your net made the right prediction " + str(predicted))
    else:
        print("Your net prediction was " + str(predicted) + ", but the correct label is: " + str(label))
'''

#___________________________________________________________________________________
##  Part IV Using Convolutional Neural Networks
# _______________________________________________________

 
#1)Sequential module - init method
# Having learned about the sequential module, now is the time to see how you can convert a 
# neural network that doesn't use sequential modules to one that uses them. We are giving the
# code to build the network in the usual way, and you are going to write the code for the same
# network using sequential modules.
'''
class Net(nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=5, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=5, out_channels=10, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=20, out_channels=40, kernel_size=3, padding=1)

        self.relu = nn.ReLU()

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(7 * 7 * 40, 1024)
        self.fc2 = nn.Linear(1024, 2048)
        self.fc3 = nn.Linear(2048, 10) 
'''
# We want the pooling layer to be used after the second and fourth convolutional layers, while the
#  relu nonlinearity needs to be used after each layer except the last (fully-connected) layer. 
# For the number of filters (kernels), stride, passing, number of channels and number of units, 
# use the same numbers as above.

# Instructions 1/2
# Declare all the layers needed for feature extraction in the self.features
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # Declare all the layers for feature extraction
        self.features = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=5, kernel_size=3, padding=1), 
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(in_channels=5, out_channels=10, kernel_size=3, padding=1), 
                                      nn.MaxPool2d(2, 2), nn.ReLU(inplace=True),
                                      nn.Conv2d(in_channels=10, out_channels=20, kernel_size=3, padding=1), 
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(in_channels=20, out_channels=40, kernel_size=3, padding=1), 
                                      nn.MaxPool2d(2, 2), nn.ReLU(inplace=True))

# Instructions 2/2
# Declare the three linear layers in self.classifier.

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # Declare all the layers for feature extraction
        self.features = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=5, kernel_size=3, padding=1), 
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(in_channels=5, out_channels=10, kernel_size=3, padding=1), 
                                      nn.MaxPool2d(2, 2), nn.ReLU(inplace=True),
                                      nn.Conv2d(in_channels=10, out_channels=20, kernel_size=3, padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(in_channels=20, out_channels=40, kernel_size=3, padding=1),
                                      nn.MaxPool2d(2, 2), nn.ReLU(inplace=True))
        
        # Declare all the layers for classification
        self.classifier = nn.Sequential(nn.Linear(7 * 7 * 40, 1024), nn.ReLU(inplace=True),
                                       	nn.Linear(1024, 2048), nn.ReLU(inplace=True),
                                        nn.Linear(2048, 10))



#2)Sequential module - forward() method
# Now, that you have defined all the modules that the network needs, it is time to apply 
# them in the forward() method. For context, we are giving the code for the forward() method, 
# if the net was written in the usual way.
'''
class Net(nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=5, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=5, out_channels=10, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=20, out_channels=40, kernel_size=3, padding=1)

        self.relu = nn.ReLU()

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(7 * 7 * 40, 1024)
        self.fc2 = nn.Linear(1024, 2048)
        self.fc3 = nn.Linear(2048, 10) 

    def forward():
        x = self.relu(self.conv1(x))
        x = self.relu(self.pool(self.conv2(x)))
        x = self.relu(self.conv3(x))
        x = self.relu(self.pool(self.conv4(x)))
        x = x.view(-1, 7 * 7 * 40)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
'''
# Note: for evaluation purposes, the entire code of the class needs to be in the script. 
# We are using the __init__ method as you have coded it on the previous exercise, while 
# you are going to code the forward() method here.

# Instructions
# Extract the features from the images.
# Squeeze the three spatial dimensions of the feature maps into one using the view() method.
# Classify images based on the extracted features.

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # Declare all the layers for feature extraction
        self.features = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=5, kernel_size=3, padding=1), 
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(in_channels=5, out_channels=10, kernel_size=3, padding=1), 
                                      nn.MaxPool2d(2, 2), nn.ReLU(inplace=True),
                                      nn.Conv2d(in_channels=10, out_channels=20, kernel_size=3, padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(in_channels=20, out_channels=40, kernel_size=3, padding=1),
                                      nn.MaxPool2d(2, 2), nn.ReLU(inplace=True))
        
        # Declare all the layers for classification
        self.classifier = nn.Sequential(nn.Linear(7 * 7 * 40, 1024), nn.ReLU(inplace=True),
                                       	nn.Linear(1024, 2048), nn.ReLU(inplace=True),
                                        nn.Linear(2048, 10))
        
    def forward(self, x):
        # Apply the feature extractor in the input
        x = self.features(x)
        # Squeeze the three spatial dimensions in one
        x = x.view(-1, 7 * 7 * 40)
        # Classify the images
        x = self.classifier(x)
        return x


#3)Validation set
# You saw the need for validation set in the previous video. Problem is that the datasets 
# typically are not separated into training, validation and testing. 
# It is your job as a data scientist to split the dataset into training, testing and validation. 
# The easiest (and most used) way of doing so is to do a random splitting of the dataset. 
# In PyTorch, that can be done using SubsetRandomSampler object. You are going to split the 
# training part of MNIST dataset into training and validation. After randomly shuffling the dataset, 
# use the first 55000 points for training, and the remaining 5000 points for validation.

# Instructions
# Use numpy.arange() to create an array containing numbers [0, 59999] and then randomly shuffle the array.
# In the train_loader using SubsetRandomSampler() use the first 55k points for training.
# In the val_loader use the remaining 5k points for validation.

# Import needed stuffs
import torch
import torchvision
import torch.utils.data
import numpy as np
from torchvision import datasets, transforms
#import torchvision.transforms as transforms

# Shuffle the indices
indices = np.arange(60000)
np.random.shuffle(indices)
# Build the train loader
train_loader = torch.utils.data.DataLoader(datasets.MNIST('mnist', download=True, train=True,
                     transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])),
                     batch_size=64, shuffle=False, sampler=torch.utils.data.SubsetRandomSampler(indices[:55000]))
# Build the validation loader
val_loader = torch.utils.data.DataLoader(datasets.MNIST('mnist', download=True, train=True,
                   transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])),
                   batch_size=64, shuffle=False, sampler=torch.utils.data.SubsetRandomSampler(indices[55000:]))



#4)Detecting overfitting
# Overfitting is arguably the biggest problem in machine learning and data science, and being able to detect 
# it will make you a much better data scientist. While reaching a high (or even perfect) accuracy on training 
# sets is quite easy when you use neural networks, reaching a high accuracy on validation and testing sets is 
# a very different thing.

# Let's see if you can now detect overfitting. Amongst the accuracy scores below, which network presents the 
# biggest overfitting problem. ?

# Answer the question
# Possible Answers

# The accuracy in the training set is 90%, the accuracy in the validation set is 88%.

# The accuracy in the training set is 90%, the accuracy in the testing set is 70%.

# The accuracy in the training set is 90%, the accuracy in the validation set is 70%. (correct one)

# The accuracy in the validation set is 85%, the accuracy in the testing set is 82%.



#5)L2-regularization
# You are going to implement each of the regularization techniques explained in the previous video. 
# Doing so, you will also remember important concepts studied throughout the course. You will start 
# with l2-regularization, the most important regularization technique in machine learning. 
# As you saw in the video, l2-regularization simply penalizes large weights, and thus enforces the 
# network to use only small weights.

# Instructions
# Instantiate an object called model from class Net(), which is available in your workspace 
# (consider it as a blackbox).
# Instantiate the cross-entropy loss.
# Instantiate Adam optimizer with learning_rate equals to 3e-4, and l2 regularization parameter equals to 0.001.

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
#import torch_optimizer as optim
import torch.optim as optim
                                ## Initialize NN ##
# Instantiate the network
model = Net()
# Instantiate the cross-entropy loss
criterion = nn.CrossEntropyLoss()
# Instantiate the Adam optimizer
optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=0.001)

# Cool! See how easy it is to use l2 regularization in PyTorch? When you will start using bigger networks, 
# this might well make the difference between your network overfitting or not.



#6)Dropout
# You saw that dropout is an effective technique to avoid overfitting. 
# Typically, dropout is applied in fully-connected neural networks, or in the fully-connected layers 
# of a convolutional neural network. You are now going to implement dropout and use it on a small 
# fully-connected neural network.

# For the first hidden layer use 200 units, for the second hidden layer use 500 units, and for the output 
# layer use 10 units (one for each class). For the activation function, use ReLU. 
# Use .Dropout() with strength 0.5, between the first and second hidden layer. 
# Use the sequential module, with the order being: fully-connected, activation, dropout, fully-connected, 
# activation, fully-connected.

# Instructions 1/2
# Implement the __init__ method, based on the description of the network in the context.

class Net(nn.Module):
    def __init__(self):
        
        # Define all the parameters of the net
        self.classifier = nn.Sequential(
            nn.Linear(28*28, 200), # fully-connected, 200 units for first hidden layer.
            nn.ReLU(inplace=True), # activation, initiate ReLU activation function.
            nn.Dropout(p=0.5),     # dropout, use 50% of strength between layers 1 and 2 (% units that will be drop out). 
            nn.Linear(200, 500),   # fully-connected, 500 units for second layer.
            nn.ReLU(inplace=True), # activation, initiate ReLU AF.
            nn.Linear(500, 10),    # fully-connected, 10 units for the output layer.
            )    

# Instructions 2/2
# Apply the forward pass in the forward() method.
    
    def forward(self, x):
        # Apply the feature extractor in the input
        x = self.features(x)
        # Squeeze the three spatial dimensions in one
        x = x.view(-1, 14 * 14 * 5) # numbers selected as half input/output layers
        # Classify the images
        x = self.classifier(x)
    	# Do the forward pass
        return x



#7)Batch-normalization
# Dropout is used to regularize fully-connected layers. Batch-normalization is used to make the 
# training of convolutional neural networks more efficient, while at the same time having 
# regularization effects. 
# ou are going to implement the __init__ method of a small convolutional neural network, with batch-normalization. 
# The feature extraction part of the CNN will contain the following modules (in order): 
# convolution, max-pool, activation, batch-norm, convolution, max-pool, relu, batch-norm.

# The first convolutional layer will contain 10 output channels, while the second will contain 20 output channels. 
# As always, we are going to use MNIST dataset, with images having shape (28, 28) in grayscale format (1 channel). 
# In all cases, the size of the filter should be 3, the stride should be 1 and the padding should be 1.

# Instructions
# Implement the feature extraction part of the network, using the description in the context.
# Implement the fully-connected (classifier) part of the network.

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # Implement the sequential module for feature extraction
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2, 2), # division while pooling, divided by 2 (50% left)
            nn.ReLU(inplace=True), 
            nn.BatchNorm2d(10), # batch is the number of channels
            nn.Conv2d(in_channels=10, out_channels=20, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2, 2), # division while pooling, dividing by 2 again (25% left)
            nn.ReLU(inplace=True), 
            nn.BatchNorm2d(20)  # batch is the number of channels
            )
        
        # Implement the fully connected layer for classification
        # because of 28*28 and 20 units of second layer, input features is 7*7*20...
        # output is alf of out channels
        self.fc = nn.Linear(in_features=7*7*20, out_features=10)



#8)Finetuning a CNN
# Previously, you trained a model to classify handwritten digits and saved the model parameters
#  to my_net.pth. Now you're going to classify handwritten letters, but you have a smaller training set.

# In the first step, you'll create a new model using this training set, but the accuracy will be poor. 
# Next, you'll perform the same training, but you'll start with the parameters from your digit classifying model. 
# Even though digits and letters are two different classification problems, you'll see that using information 
# from your previous model will dramatically improve this one.

# Instructions 1/2
# Create a new model using the Net() module.
# Change the number of output units, to the number of classifications for letters.
'''
# Create a new model 
model = Net()
# Change the number of out channels
model.fc = nn.Linear(7 * 7 * 512, 26) # 26 because of the letters of the alphabet
# Train and evaluate the model
model.train()
train_net(model, optimizer, criterion)
print("Accuracy of the net is: " + str(model.eval())) # gives accuracy 57%
''' # train_net needs some call to execute them

# Instructions 2/2
# Repeat the training process, but first load the digit classifier parameters from my_net.pth.
'''
# Create a model using
model = Net()
# Load the parameters from the old model
model.load_state_dict(torch.load('my_net.pth'))
# Change the number of out channels
model.fc = nn.Linear(7 * 7 * 512, 26)
# Train and evaluate the model
model.train()
train_net(model, optimizer, criterion)
print("Accuracy of the net is: " + str(model.eval()))  gives accuracy 84%
'''# train_net needs some call to execute them


#9)Torchvision module
# You already finetuned a net you had pretrained. In practice though, it is very common to finetune 
# CNNs that someone else (typically the library's developers) have pretrained in ImageNet. 
# Big networks still take a lot of time to be trained on large datasets, and maybe you cannot afford 
# to train a large network on a dataset of 1.2 million images on your laptop.

# Instead, you can simply download the network and finetune it on your dataset. 
# That's what you will do right now. You are going to assume that you have a personal dataset, containing 
# the images from all your last 7 holidays. You want to build a neural network that can classify each image 
# depending on the holiday it comes from. However, since the dataset is so small, you need to use the finetuning 
# technique.

# Instructions
# Import the module that lets you download state-of-the-art CNNs.
# Download and load a pretrained ResNet18 network.
# Freeze all the layers bar the final one.
# Change the last layer to correspond to the number of classes (7) in your dataset.

# Import the module
import torchvision
# Download resnet18
model = torchvision.models.resnet18(pretrained=True)
# Freeze all the layers bar the last one
for param in model.parameters():
    param.requires_grad = False
# Change the number of output units
model.fc = nn.Linear(512, 7) # last number is number of classes (num_classes), 7 here.


# end of course
