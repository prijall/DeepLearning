import torch

x1=torch.Tensor([2.0]).double();  x1.requires_grad=True
x2=torch.Tensor([4.0]).double();  x2.requires_grad=True
w1=torch.Tensor([-3.0]).double(); w1.requires_grad=True
w2=torch.Tensor([5.0]).double();  w2.requires_grad=True
b=torch.Tensor([3.5]).double();   b.requires_grad=True

n=x1*w1+x2*w2+b
o=torch.relu(n)

print(o.data.item())
o.backward()

print('x2', x2.grad.item())
print('w2', w2.grad.item())
print('x1', x1.grad.item())
print('w1', w1.grad.item())
