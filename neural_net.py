import random
import numpy as np
from value import Value

class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []

class Neuron(Module):
    def __init__(self, nin, nonlin=True):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(0)
        self.nonlin = nonlin

    def __call__(self, x):
        act = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)
        return act.relu() if self.nonlin else act

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"

class Layer(Module):
    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

class MLP(Module):
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], nonlin=i!=len(nouts)-1) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
    
    def fit(self, X_train, y_train, epochs=100, learning_rate=0.01):
        for epoch in range(epochs):
            total_loss = 0
            for x, y in zip(X_train, y_train):
                x_formatted = [Value(xi) for xi in x]

                pred_y = self(x_formatted)

                if not isinstance(pred_y, list):
                    pred_y = [pred_y]
                if not isinstance(y, list):
                    y = [y]


                loss = sum([(py - Value(yy))**2 for py, yy in zip(pred_y, y)])
                total_loss += loss.data

                loss.backward()

                for param in self.parameters():
                    param.data -= learning_rate * param.grad

                self.zero_grad()
            print(f"Epoch {epoch}, Loss {total_loss}")

    def predict(self, X):
        predictions = []
        for x in X:
            formatted_x = [Value(xi) for xi in x]
            output = self(formatted_x)

            pred = 1 if output.data > 0.5 else 0
            predictions.append(pred)

        return np.array(predictions)
