from Object_Modeling import Model
from LayersofNeurons_part2 import Dense_layer
from ReLu_condition_Part2 import ReLu_Activation
from Activation_Linear import Linear_Activation
from Adam_Optimizer import Adam_optimizer
from Mean_Squared_Error import Loss_MeanSquaredError
from nnfs.datasets import sine_data

# Create dataset
X, y = sine_data()
# Instantiate the model
model = Model()
# Add layers
model.add(Dense_layer(1, 64))
model.add(ReLu_Activation())
model.add(Dense_layer(64, 64))
model.add(ReLu_Activation())
model.add(Dense_layer(64, 1))
model.add(Linear_Activation())


# Set loss and optimizer objects
model.set(
Loss=Loss_MeanSquaredError(),
optimizer=Adam_optimizer(learning_rate=0.01, decay=1e-7),
)
# Finalize the model
model.finalize()
# Train the model
model.train(X, y, epochs=10000, print_every=100)