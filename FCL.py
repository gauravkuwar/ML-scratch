import numpy as np # only using numpy for computation speed up

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def der_sigmoid(x): # derivative
    return sigmoid(x) * (1 - sigmoid(x))

class FCL:
    def __init__(self, input_layer=784, hidden_layer_count=2, hidden_layer=16, output_layer=10): 
        self.output_layer = output_layer
        
        # initialize weights
        self.weights = [0] * (hidden_layer_count + 1)
        self.weights[0] = np.random.uniform(-1, 1, size=(hidden_layer, input_layer))
        self.weights[-1] = np.random.uniform(-1, 1, size=(output_layer, hidden_layer))

        for i in range(1, len(self.weights)-1):
            self.weights[i] = np.random.uniform(-1, 1, size=(hidden_layer, hidden_layer))
        
        # initialize biases
        self.biases = [0] * (hidden_layer_count + 1)
        self.biases[0] = np.random.uniform(-1, 1, size=(hidden_layer, 1))
        self.biases[-1] = np.random.uniform(-1, 1, size=(output_layer, 1))

        for i in range(1, len(self.weights)-1):
            self.biases[i] = np.random.uniform(-1, 1, size=(hidden_layer, 1))
            
        
    def train(self, data, labels, lr=0.01, epochs=5):
        tot = len(data)
            
        for epoch in range(epochs):
            count = 1

            for j in range(tot):
                a0, label = data[j], labels[j]
                delta_w, delta_b = self.backward_prop(a0, label)
                
                for i in range(len(self.weights)):
                    self.weights[i] -= lr * delta_w[i]
                    self.biases[i] -= lr * delta_b[i]

                print(f"Epoch {epoch}: {count}/{tot}", end="\r")
                count += 1
        
        
    def backward_prop(self, a0, label):
        a, z = self.forward_prop(a0)    
        y = np.zeros((self.output_layer, 1))
        y[label] = 1
        
        delta_w = [0] * len(self.weights)
        delta_b = [0] * len(self.biases)
        cur_delta_a = 2 * (a[-1] - y)
        
        for i in reversed(range(len(self.weights))):
            delta_b[i] = der_sigmoid(z[i]) * cur_delta_a        # change in biases
            delta_w[i] = a[i].flatten() * delta_b[i]            # change in weights
            cur_delta_a = np.dot(self.weights[i].T, delta_b[i]) # change in activations
        
        return delta_w, delta_b
        
    def forward_prop(self, a):
        a = [a.reshape(-1, 1)]
        z = []
        
        for i in range(len(self.weights)):
            z.append(np.dot(self.weights[i], a[i]) + self.biases[i])
            a.append(sigmoid(z[i]))
        
        return a, z
    
    def predict(self, img):
        a, z = self.forward_prop(img)
        return np.argmax(a[-1]), a[-1]
    
    def accuracy(self, data, labels):
        correct = 0
        count = 1
        tot = len(data)
        
        for i in range(tot):
            a0, label = data[i], labels[i]
            predicted_label, res  = self.predict(a0)
            
            cur_acc = round(correct / count, 5)
            print(f"Current Acc: {cur_acc} Count: {count}/{tot}", end="\r")
            
            if predicted_label == label: correct += 1
            count += 1
        
        acc = correct / tot
        err = 1 - acc
        print(f"Acc: {acc}\nError Rate: {err}")
        
        return correct / tot