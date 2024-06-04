
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
print(y)

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
#print(drelu_dxw0, drelu_dxw1, drelu_dxw2, drelu_db)

#@ Partial derivative of the multiplication, the chain rule:
dmul_dx0=w[0]
dmul_dx1=w[1]
dmul_dx2=w[2]
dmul_dw0=x[0]
dmul_dw1=x[1]
dmul_dw2=x[2]
drelu_dx0=drelu_dxw0*dmul_dx0
drelu_dw0=drelu_dxw0*dmul_dw0
drelu_dx1=drelu_dxw0*dmul_dx1
drelu_dw1=drelu_dxw0*dmul_dw1
drelu_dx2=drelu_dxw0*dmul_dx2
drelu_dw2=drelu_dxw0*dmul_dw2
#print(drelu_dx0, drelu_dw0, drelu_dx1, drelu_dw1, drelu_dx2, drelu_dw2)

dx=[drelu_dx0, drelu_dx1, drelu_dx2] #gradients on inputs
dw=[drelu_dw0, drelu_dxw1, drelu_dxw2] #gradient on weights
db=drelu_db #gradient on bias

w[0]+=-0.001*dw[0]
w[1]+=-0.001*dw[1]
w[2]+=-0.001*dw[2]
b+=-0.001* db
#print(w, b)

# Multiplying inputs by weights
xw0 = x[0] * w[0]
xw1 = x[1] * w[1]
xw2 = x[2] * w[2]

# Adding
z = xw0 + xw1 + xw2 + b
# ReLU activation function
y = max(z, 0)
print(y)