<h1 align="center" id="top">Mixture of Idiots Experts 🤪🧠💥</h1>

<p align="center">
  <strong>Sometimes, a team of highly specialized "idiot" experts is exactly what your model needs!</strong>
</p>

Welcome to **Mixture of Idiots Experts (MoIE)**! This project provides a PyTorch implementation of a Sparsely-Gated Mixture of Experts (MoE) layer. We're all about making neural networks bigger and smarter, but in a more efficient, "sparsely-gated" kind of way. Even if the experts are a bit... focused.

## 📜 Table of Contents

1.  [🤔 The Core Idea: Strength in Specialization (and Sparsity!)](#core-idea)
2.  [📂 Project Structure](#project-structure)
3.  [🤓 Inside the "Mixture of Idiots Experts" Layer (`idiot_expert.py`)](#expert-layer-deep-dive)
    *   [🚪 The Gating Mechanism: Who Gets the Mic?](#gating-mechanism)
    *   [💡 The "Idiot" Experts: Specialized but Focused](#the-experts)
    *   [🚀 The Sparse Dispatcher: Efficient Routing](#sparse-dispatcher)
    *   [⚖️ Auxiliary Loss: Keeping Experts Honest](#auxiliary-loss)
    *   [🎬 Workflow Diagram](#workflow-diagram)
4.  [🛠️ Tech Stack](#tech-stack)
5.  [🚀 Getting Started](#getting-started)
    *   [1. Prerequisites](#prerequisites)
    *   [2. Clone the Repository](#clone-repository)
    *   [3. Set Up a Virtual Environment (Recommended)](#set-up-virtual-environment)
    *   [4. Install Dependencies](#install-dependencies)
6.  [🏃 How to Run the Examples](#how-to-run)
    *   [🧪 Basic Test Drive: `blueprint_run.py`](#run-blueprint)
    *   [🖼️ CIFAR-10 Vision Quest: `deepview.py`](#run-deepview)
    *   [🧩 Using `idiot_expert.py` in Your Own Models](#using-expert-module)
7.  [✨ Example Output](#example-output)
8.  [📜 Acknowledgements & Citation](#acknowledgements)
9.  [🤝 Contributing](#contributing)

---

<h2 id="core-idea">🤔 The Core Idea: Strength in Specialization (and Sparsity!)</h2>

The traditional way to make neural networks more powerful is often to make them bigger—more layers, more parameters. However, this can lead to massive computational costs for every single input.

The **Sparsely-Gated Mixture of Experts (MoE)** approach offers a clever alternative:
*   **Many Experts:** Instead of one giant network, you have numerous smaller, specialized "expert" sub-networks. Each expert can be thought of as an "idiot savant," good at a particular type of task or feature.
*   **Selective Activation (Sparsity):** For any given input, a "gating mechanism" intelligently selects only a few of these experts to process the input. The other experts remain dormant, saving computation.
*   **Conditional Computation:** The model learns to route different types of inputs to the experts best suited for them. This allows the model to have a huge total number of parameters (the sum of all experts) but use only a fraction of them for any single prediction.

This project provides a PyTorch module (`idiot_expert.py`) that implements such a layer, letting you build your own "Mixture of Idiots Experts" into your neural network architectures!

<h2 id="project-structure">📂 Project Structure</h2>

Here's what you'll find in this repository:

```
Mixture-of-Idiots-Experts/
├── .git/                     # Git version control files
├── .gitignore                # Specifies intentionally untracked files
├── .venv/                    # Your local Python virtual environment (GITIGNORED!)
├── idiot_expert.py           # The core MoE layer implementation 🤓
├── blueprint_run.py          # A minimal example with dummy data 🧪
├── deepview.py               # A CIFAR-10 image classification example 🖼️
├── requirements.txt          # Python package dependencies
└── README.md                 # This fabulous file!
```

<h2 id="expert-layer-deep-dive">🤓 Inside the "Mixture of Idiots Experts" Layer (`idiot_expert.py`)</h2>

The `idiot_expert.py` file is where the magic happens. It defines the `MoE` class, which you can integrate into your PyTorch models. Here's a simplified breakdown of its components:

<h3 id="gating-mechanism">🚪 The Gating Mechanism: Who Gets the Mic?</h3>

*   **Concept:** A small neural network (the "gate") takes the input and decides which experts are most relevant for it.
*   **How it Works:**
    1.  The input `x` is passed through a linear layer (`self.w_gate`).
    2.  If `noisy_gating` is enabled during training, some noise is added to the gate's logits. This encourages exploration and better load balancing.
    3.  The logits are passed through a `softmax` to get probabilities for each expert.
    4.  It then selects the top `k` experts based on these probabilities.
    5.  The output is a "gates" tensor that weights the contribution of the selected experts.

<h3 id="the-experts">💡 The "Idiot" Experts: Specialized but Focused</h3>

*   **Concept:** These are individual neural networks. In this implementation, they are simple Multi-Layer Perceptrons (MLPs), but they could be more complex.
*   **How they Work:**
    1.  The `MoE` module initializes a list of `num_experts` MLP instances (`self.experts`).
    2.  Each expert MLP has an input layer, a hidden ReLU layer, and an output layer with a softmax.
    3.  Only the inputs routed to a specific expert (by the dispatcher) are processed by that expert.

<h3 id="sparse-dispatcher">🚀 The Sparse Dispatcher: Efficient Routing</h3>

*   **Concept:** This helper class (`SparseDispatcher`) efficiently sends the correct input slices to the active experts and then combines their outputs.
*   **How it Works:**
    1.  **Dispatch:** Takes the full input batch and the "gates" tensor. It figures out which input examples go to which of the `k` chosen experts and creates smaller, expert-specific mini-batches.
    2.  **Combine:** After the experts process their respective inputs, the dispatcher takes their outputs and combines them, weighted by the gate values, to produce the final output for the MoE layer.

<h3 id="auxiliary-loss">⚖️ Auxiliary Loss: Keeping Experts Honest (and Busy!)</h3>

*   **Concept:** To prevent the gating network from always picking the same few "favorite" experts and ignoring others, an auxiliary loss is calculated.
*   **How it Works:**
    1.  This loss encourages the load to be balanced across all experts. It typically uses the coefficient of variation (`cv_squared`) of both the importance (sum of gate values per expert) and the actual load (number of examples per expert).
    2.  This `aux_loss` should be added to your main task loss during training to ensure all "idiot" experts get a chance to learn and contribute.

<h3 id="workflow-diagram">🎬 Workflow Diagram</h3>

The following diagram illustrates the flow of data through the MoE layer:

```text
+-------------------------+
|    Input Data (Batch X) |
+-------------------------+
            |
            |
  +---------+---------+
  |                   |
  v                   v
+-----------------------+         +------------------------------------------+
|  Gating Mechanism     |         |                                          |
| - Input: X            |         |                                          |
| - Output:             |         |                                          |
|   - gate_scores (g)   |---------+--> SparseDispatcher                      |
|   - top_k_indices     |--+      |   - Input: X, gate_scores, top_k_indices |
+-----------------------+  |      |                                          |
                           |      |   1. DISPATCH:                           |
                           |      |      For each item in X, route it to     |
                           |      |      the 'top_k_indices' experts:        |
                           |      |                                          |
                           |      |      X_item --> Expert_k1, Expert_k2 ... |
                           |      |                    |        |            |
                           |      |                    v        v            |
                           |      |             +----------+ +----------+    |
                           |      |             | Expert_k1| | Expert_k2|... |
                           |      |             |  (MLP)   | |  (MLP)   |    |
                           |      |             +----------+ +----------+    |
                           |      |                    |        |            |
                           |      |                    v        v            |
                           |      |              (output_k1)(output_k2)      |
                           |      |                                          |
                           |      |   2. COMBINE:                            |
                           +------|--> Collects expert outputs & combines    |
                                  |   them weighted by 'gate_scores' (g):    |
                                  |   Final_Y_item = sum(g_i * output_ki)    |
                                  |                                          |
                                  +------------------------------------------+
                                                        |
                                                        v
                                          +---------------------------+
                                          | Final Output of MoE Layer |
                                          +---------------------------+
``` 

---

<h2 id="tech-stack">🛠️ Tech Stack</h2>

*   **Python 3.x**
*   **PyTorch:** For all the neural network goodness.
*   **NumPy:** For numerical operations (though primarily handled by PyTorch tensors).

<h2 id="getting-started">🚀 Getting Started</h2>

Follow these steps to get the "Mixture of Idiots Experts" running on your local machine.

<h3 id="prerequisites">1. Prerequisites</h3>

*   Python 3.8 or higher.
*   `pip` for installing packages.

<h3 id="clone-repository">2. Clone the Repository</h3>

```bash
git clone https://github.com/VinsmokeSomya/Mixture-of-Idiots-Experts.git
```
```bash
cd Mixture-of-Idiots-Experts
```
(Assuming you're already in the project directory named `mixture-of-experts` which you might want to rename to `Mixture-of-Idiots-Experts` locally if you prefer.)

<h3 id="set-up-virtual-environment">3. Set Up a Virtual Environment (Recommended)</h3>

It's highly recommended to use a virtual environment to keep dependencies clean.

```bash
python -m venv .venv
```
Activate the environment:
*   **On Windows (PowerShell/CMD):**
    ```bash
    .venv\Scripts\activate
    ```
*   **On macOS/Linux (bash/zsh):**
    ```bash
    source .venv/bin/activate
    ```
You should see `(.venv)` at the beginning of your terminal prompt.

<h3 id="install-dependencies">4. Install Dependencies</h3>

With your virtual environment activated, install the required packages:

```bash
pip install -r requirements.txt
```
This will install PyTorch and Torchvision.

<h2 id="how-to-run">🏃 How to Run the Examples</h2>

This project comes with two example scripts to demonstrate the MoE layer.

<h3 id="run-blueprint">🧪 Basic Test Drive: `blueprint_run.py`</h3>

This script runs a very simple test of the `MoE` layer with dummy data. It's great for a quick sanity check.

```bash
python blueprint_run.py
```
You'll see output showing the training loss and evaluation loss for the dummy task.

<h3 id="run-deepview">🖼️ CIFAR-10 Vision Quest: `deepview.py`</h3>

This script applies the `MoE` layer to the CIFAR-10 image classification task. It includes data loading, training, and evaluation.

```bash
python deepview.py
```
This will:
1.  Download the CIFAR-10 dataset (if not already present in a `./data` directory).
2.  Train a model using the `MoE` layer.
3.  Print training progress and final accuracy on the test set.
Remember the `if __name__ == '__main__': freeze_support()` is included for Windows compatibility with `DataLoader`.

<h3 id="using-expert-module">🧩 Using `idiot_expert.py` in Your Own Models</h3>

The main purpose of `idiot_expert.py` is to be used as a layer in your custom PyTorch models.

```python
from idiot_expert import MoE
import torch

# Example instantiation
input_dim = 784  # e.g., flattened MNIST image
num_classes = 10
num_total_experts = 8
expert_hidden_size = 64
k_selected_experts = 2

model = MoE(
    input_size=input_dim,
    output_size=num_classes,
    num_experts=num_total_experts,
    hidden_size=expert_hidden_size,
    k=k_selected_experts,
    noisy_gating=True
)

# Dummy input
batch_s = 32
dummy_input = torch.randn(batch_s, input_dim)

# Forward pass (training mode)
model.train()
output, aux_loss = model(dummy_input)
print("Output shape:", output.shape) # Should be [batch_s, num_classes]
print("Auxiliary loss:", aux_loss.item())

# Your main loss calculation would go here, e.g.:
# main_loss = criterion(output, targets)
# total_training_loss = main_loss + aux_loss
# total_training_loss.backward()
# ...
```

<h2 id="example-output">✨ Example Output</h2>

*   **`blueprint_run.py`**:
    ```
    Training Results - loss: X.XX, aux_loss: Y.YYY
    Evaluation Results - loss: X.XX, aux_loss: Y.YYY
    ```
*   **`deepview.py`**:
    ```
    [1,   100] loss: A.AAA
    [1,   200] loss: B.BBB
    ...
    Finished Training
    Accuracy of the network on the 10000 test images: ZZ %
    ```
(Where X, Y, A, B, ZZ are numerical values you'll see upon execution.)

<h2 id="acknowledgements">📜 Acknowledgements & Citation</h2>

*   This code is inspired by and based on the TensorFlow implementation of Mixture of Experts found in the [tensor2tensor library](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/utils/expert_utils.py).
*   The `deepview.py` (CIFAR-10 example) structure is partly based on the official PyTorch CIFAR-10 tutorial.

And this PyTorch implementation can be informally credited if desired:
```
@misc{rau2019moe_pytorch,
    title={Sparsely-gated Mixture-of-Experts PyTorch implementation (Mixture of Idiots Experts adaptation)},
    author={David Rau (original), Adapted by VinsmokeSomya},
    journal={https://github.com/davidmrau/mixture-of-experts (original), https://github.com/VinsmokeSomya/Mixture-of-Experts-Idiots.git (adaptation)},
    year={2019-Present}
}
```

<h2 id="contributing">�� Contributing</h2>

Contributions, issues, and feature requests are welcome! While this started as a fun take on a serious concept, improvements are always appreciated. Feel free to fork the repo, make your changes, and submit a pull request.

---

Let the most efficient (and well-gated) "idiot" experts win! 🎉

