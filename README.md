# MNIST Neural Network from Scratch

![Python](https://img.shields.io/badge/python-3.x-blue) ![License](https://img.shields.io/badge/license-MIT-green)

A fully-connected neural network built from scratch in NumPy — no PyTorch, TensorFlow, or Keras — that classifies handwritten digits from the MNIST dataset. Forward propagation, backpropagation, and the optimizer are all implemented manually.

## Architecture

- **Input:** 784 (28×28 images, flattened and normalized to [0, 1])
- **Hidden layer:** 256 neurons, ReLU activation, He initialization
- **Output layer:** 10 neurons (digits 0–9), Softmax activation
- **Loss:** Categorical Cross-Entropy
- **Optimizer:** Mini-batch SGD with momentum (0.9) and learning rate decay

## Results

- **Test accuracy: ~97–98%**
- Trained with mini-batch gradient descent (batch size 128) over 30 epochs

## Visualizations

The script generates and saves three plots after training:

| Training Curves | Sample Predictions | Confusion Matrix |
|---|---|---|
| ![Training curves](<img width="1200" height="400" alt="loss and accuracy" src="https://github.com/user-attachments/assets/5094b77b-15a5-4e66-a5e3-0635ef0d9cbc" />
) | ![Sample predictions](<img width="1536" height="800" alt="predictions" src="https://github.com/user-attachments/assets/253855f4-7833-4105-8ecd-ae6097011245" />
) | ![Confusion matrix](<img width="700" height="600" alt="confusion matrix" src="https://github.com/user-attachments/assets/a8c2202d-1173-43f3-b934-4aebed7a8e6a" />
) |

## Getting Started

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run
```bash
python main.py
```

MNIST is downloaded automatically on first run (via `sklearn.datasets.fetch_openml`) and cached locally afterward.

## What's implemented from scratch

- Dense (fully-connected) layer — forward and backward pass
- ReLU activation — forward and backward pass
- Softmax activation
- Categorical cross-entropy loss
- Combined softmax + cross-entropy backward pass (simplified gradient)
- SGD optimizer with momentum and learning rate decay
- Mini-batch training loop

## Key Learnings

- Implementing backpropagation by hand makes it clear *why* the softmax + cross-entropy gradient simplifies to `predictions - true_labels` — an elegant shortcut that's easy to take for granted when a framework's autograd handles it for you.
- Mini-batching had a much bigger effect on convergence than expected: going from 1 weight update per epoch (full-batch) to ~469 updates per epoch (batch size 128) was the single biggest jump in accuracy, bigger than any architecture change.
- Debugging shape mismatches (a transposed weight matrix, forwarding the wrong split of the data) forced a much deeper understanding of how data actually flows through each layer than just reading the theory does.

## Future Improvements

- [ ] Add a second hidden layer / experiment with different widths
- [ ] Implement the Adam optimizer for comparison against SGD + momentum
- [ ] Add dropout or L2 regularization
- [ ] Try a convolutional architecture to push past ~98% accuracy
- [ ] Add unit tests for each layer's forward/backward pass

## Acknowledgments

- Architecture and training patterns based on Harrison Kinsley's [*Neural Networks from Scratch*](https://nnfs.io) (NNFS) book/series
- Dataset: [MNIST](http://yann.lecun.com/exdb/mnist/), loaded via `sklearn.datasets.fetch_openml`

## Author

[Your name] — [GitHub](#) · [LinkedIn](#)

## License

MIT
