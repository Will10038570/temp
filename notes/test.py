

import torch
from torchviz import make_dot

x = torch.tensor(2.0, requires_grad=True)

a = x ** 2
b = 3 * x
loss = a + b

graph = make_dot(loss, params={"x": x})
graph.render("computation_graph", format="svg", cleanup=True)

