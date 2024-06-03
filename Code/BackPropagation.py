
#@ Forward Pass:
x=[1.0, -2.0, 3.0] #input values
w=[-3.0, -1.0, 2.0] #weights
b=1.0 #bias

#@Multiplying inputs with weight:
xw0=x[0]*w[0]
xw1=x[1]*w[1]
xw2=x[2]*w[2]

#@ Adding all with bias:
z=xw0+xw1+xw2+b

#@ ReLU activation function:
y=max(z, 0)
#print(y)

#@ Backward Pass:

# the derivative from next layer:
dvalue=1.0 # For demonstration 

#Derivative of ReLU and chain rule:
drelu_dz=dvalue*(1. if z > 0 else 0.)
#print(drelu_dz)

#@ Partial derivatives of multiplication, the chain rule:
dsum_dxw0=1 #this is because The partial derivative of the sum 
dsum_dxw1=1 #operation is always 1, no matter the inputs.
dsum_dxw2=1
dsum_db=1

drelu_dxw0=drelu_dz*dsum_dxw0
drelu_dxw1=drelu_dz*dsum_dxw1
drelu_dxw2=drelu_dz*dsum_dxw2
drelu_db=drelu_dz*dsum_db

print(drelu_dxw0, drelu_dxw1, drelu_dxw2, drelu_db)