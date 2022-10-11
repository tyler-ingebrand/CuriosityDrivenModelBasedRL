import torch

# calculate deriv of 1D function
print("Part 1")
x = torch.tensor(10.0, requires_grad = True)
print("x:", x)
y = 3*x**2 + 1
y.backward()
dx = x.grad
print("x.grad :", dx)


# calculate deriv of multiple variables
print("\nPart 2")
x = torch.tensor(3) # create tensor without requires_grad = true
w = torch.tensor(2.0, requires_grad = True)
b = torch.tensor(5.0, requires_grad = True)
y = w*x + b
print("y:", y)
y.backward()
print("x.grad :", x.grad)
print("w.grad :", w.grad)
print("b.grad :", b.grad)

# calculate derivs for nd function
print("\nPart 3")
x = torch.tensor([3.0, 2.0]) # create tensor without requires_grad = true
w = torch.tensor([[2.0, 5.0]], requires_grad = True)
b = torch.tensor([5.0], requires_grad = True)
y = torch.matmul(w, x) + b
print("y:", y)
y.backward()
print("x.grad :", x.grad)
print("w.grad :", w.grad)
print("b.grad :", b.grad)

# Do gradient through a function
print("\nPart 4")
def something(x,y,z):
    return x*y + z
x = torch.tensor(1.0, requires_grad=True)
y = torch.tensor(2.0, requires_grad=True)
z = torch.tensor(3.0, requires_grad=True)
out = something(x,y,z)
out.backward()
print("x.grad :", x.grad)
print("y.grad :", y.grad)
print("z.grad :", z.grad)
