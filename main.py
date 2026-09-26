import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.metrics import confusion_matrix

class Layer_Dense:
    def __init__(self,n_inputs,n_neurons):
        self.weights = 0.1*np.random.randn(n_inputs,n_neurons)
        self.biases = np.zeros((1,n_neurons))
    def forward(self,inputs):
        self.inputs = inputs
        self.output = np.dot(inputs,self.weights) + self.biases

    def backward(self,dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dinputs = np.dot(dvalues, self.weights.T)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)


class Activation_ReLU:
    def forward(self,inputs):
        self.inputs = inputs
        self.output =np.maximum(0,self.inputs)
    def backward(self,dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <=0] =0
    

class SoftMax_Activation:
    def forward(self,inputs):
        self.inputs = inputs
        exp_values = np.exp(inputs - np.max(inputs , axis=1 , keepdims=True))
        norm_values = exp_values/np.sum(exp_values, axis=1 ,keepdims=True)
        self.output = norm_values

class Loss:
    def calculate(self,output,y):
        sample_losses = self.forward(output,y)
        data_loss = np.mean(sample_losses)
        return data_loss

class Loss_Categorical_Cross_Entropy(Loss):
    def forward(self,y_pred,y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred,1e-7,1-1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples),y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped*y_true,axis=1)

        negetive_log_likelyhoods = -np.log(correct_confidences)
        return negetive_log_likelyhoods

class Softmax_Catecorical_Crossentropy:
    def __init__(self):
        self.activation = SoftMax_Activation()
        self.loss = Loss_Categorical_Cross_Entropy()

    def forward(self, inputs , y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.calculate(self.output,y_true)
    def backward(self,dvalues,y_true):
        samples = len(dvalues)
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true,axis=1)
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples),y_true] -=1
        self.dinputs = self.dinputs/samples
              
class Optimizer_SGD:
    def __init__(self,learning_rate = 0.5):
        self.learning_rate = learning_rate
    def update_params(self,layer):
        layer.weights -= self.learning_rate * layer.dweights
        layer.biases -= self.learning_rate * layer.dbiases

mnist = fetch_openml('mnist_784' , version=1 , as_frame=False)

X = mnist['data'].astype(np.float32)/255.0
y = mnist['target'].astype(int)

X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]


dense1 = Layer_Dense(784,128)
activation1 = Activation_ReLU()

dense2 = Layer_Dense(128,10)
loss_activation = Softmax_Catecorical_Crossentropy()

optimizer = Optimizer_SGD(learning_rate=0.5)

#Training Loop

loss_history = []
acc_history = []

for epoch in range(201):
    #forward
    dense1.forward(X_train)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    loss = loss_activation.forward(dense2.output,y_train)

    predictions = np.argmax(loss_activation.output,axis=1)
    accuracy = np.mean(predictions == y_train)
    loss_history.append(loss)          # <- must be inside the loop, unconditional
    acc_history.append(accuracy) 

    if epoch % 20 ==0:
        print(f"epoch: {epoch}, acc: {accuracy:.3f}, loss: {loss:.3f}")

    #backward pass
    loss_activation.backward(loss_activation.output , y_train)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    optimizer.update_params(dense1)
    optimizer.update_params(dense2)

#Testing:
dense1.forward(X_test)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
loss = loss_activation.forward(dense2.output,y_test)

predictions = np.argmax(loss_activation.output,axis=1)
accuracy = np.mean(predictions == y_test)
print(f"test accuracy: {accuracy:.3f}, test loss: {loss:.3f}")
 
 

fig1, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(loss_history, color="crimson")
axes[0].set_title("Training Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
 
axes[1].plot(acc_history, color="seagreen")
axes[1].set_title("Training Accuracy")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
fig1.tight_layout()
fig1.savefig("training_curves.png")
 

sample_idx = np.random.choice(len(X_test), size=16, replace=False)
 
fig2, grid_axes = plt.subplots(4, 4, figsize=(8, 8))
for ax, idx in zip(grid_axes.flat, sample_idx):
    img = X_test[idx].reshape(28, 28)
    pred = predictions[idx]
    true = y_test[idx]
    color = "green" if pred == true else "red"
    ax.imshow(img, cmap="gray")
    ax.set_title(f"pred: {pred}  true: {true}", color=color, fontsize=10)
    ax.axis("off")
fig2.suptitle("Sample Test Predictions", fontsize=14)
fig2.tight_layout()
fig2.savefig("sample_predictions.png")
 

cm = confusion_matrix(y_test, predictions)
 
fig3, cm_ax = plt.subplots(figsize=(7, 6))
im = cm_ax.imshow(cm, cmap="Blues")
cm_ax.set_xticks(range(10))
cm_ax.set_yticks(range(10))
cm_ax.set_xlabel("Predicted")
cm_ax.set_ylabel("Actual")
cm_ax.set_title("Confusion Matrix")
for i in range(10):
    for j in range(10):
        cm_ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
fig3.colorbar(im, ax=cm_ax)
fig3.tight_layout()
fig3.savefig("confusion_matrix.png")
 
plt.show()
