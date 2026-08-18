#!/usr/bin/env python3
"""
quantum_classifier_factory.py

Variational Quantum Classifier (VQC) — Binary and Multiclass.

Design decisions
----------------
Binary:
    - Measures ⟨Z₀⟩ → probability via (exp+1)/2
    - Binary cross-entropy loss

Multiclass:
    - Measures ⟨Z_q⟩ for q in range(n_classes)
    - Softmax over expectation values → class probabilities
    - Cross-entropy loss
    - WHY: n_classes <= n_qubits is required so each class
      maps to one dedicated measurement qubit.

parameter-shift gradient:
    WHY: exact quantum-compatible gradient. Hardware-ready.

lightning.qubit:
    WHY: faster C++ CPU simulator than default.qubit.
    NOT a GPU device. To use RTX 3080 install
    pennylane-lightning-gpu and set device_name="lightning.gpu".

torch.optim.Adam:
    WHY: cleaner batched training than PennyLane AdamOptimizer.
    Weights stay on CPU (simulator runs on CPU).
"""

from typing import Dict, Any

import numpy as np
import torch
import pennylane as qml
import time

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler


class VQCClassifier(BaseEstimator, ClassifierMixin):

    def __init__(
        self,
        n_qubits: int = 6,
        n_layers: int = 2,
        epochs: int = 40,
        learning_rate: float = 0.01,
        batch_size: int = 32,
        early_stopping: bool = True,
        patience: int = 8,
        min_delta: float = 1e-4,
        random_state: int = 42,
        device_name: str = "lightning.gpu",
    ):
        self.n_qubits      = n_qubits
        self.n_layers      = n_layers
        self.epochs        = epochs
        self.learning_rate = learning_rate
        self.batch_size    = batch_size
        self.early_stopping = early_stopping
        self.patience      = patience
        self.min_delta     = min_delta
        self.random_state  = random_state
        self.device_name   = device_name

    # ========================================================
    # CIRCUIT
    # ========================================================

    def _circuit(self, x, weights):
        """
        Angle encoding → RY+RZ variational layers → CNOT ring.
        Same architecture for both binary and multiclass —
        only the measurement differs.
        """
        for q in range(self.n_qubits):
            qml.RY(x[q], wires=q)

        for layer in range(self.n_layers):
            for q in range(self.n_qubits):
                qml.RY(weights[layer, q, 0], wires=q)
                qml.RZ(weights[layer, q, 1], wires=q)

            for q in range(self.n_qubits - 1):
                qml.CNOT(wires=[q, q + 1])
            qml.CNOT(wires=[self.n_qubits - 1, 0])

    def _get_device_and_diff(self):
        """
        WHY: lightning.gpu requires adjoint diff_method —
        parameter-shift is not supported on GPU device.
        lightning.qubit (CPU) supports both but adjoint is faster.
        default.qubit supports parameter-shift only.

        Priority:
            1. lightning.gpu  → adjoint  (RTX 3080, fastest)
            2. lightning.qubit → adjoint (CPU C++, fast)
            3. default.qubit  → parameter-shift (fallback)
        """
        if self.device_name == "lightning.gpu":
            diff_method = "adjoint"
        elif self.device_name == "lightning.qubit":
            diff_method = "adjoint"
        else:
            diff_method = "parameter-shift"

        try:
            dev = qml.device(self.device_name, wires=self.n_qubits)
            print(f"          Device  : {self.device_name} "
                f"(diff={diff_method})")
            return dev, diff_method
        except Exception as e:
            print(f"          WARNING: {self.device_name} failed: {e}")
            print(f"          Falling back to lightning.gpu")
            dev = qml.device("lightning.gpu", wires=self.n_qubits)
            return dev, "adjoint"


    def _build_binary_qnode(self):
        dev, diff_method = self._get_device_and_diff()

        @qml.qnode(dev, interface="torch", diff_method=diff_method)
        def circuit(x, weights):
            self._circuit(x, weights)
            return qml.expval(qml.PauliZ(0))

        return circuit


    def _build_multiclass_qnode(self):
        if self.n_classes_ > self.n_qubits:
            raise ValueError(
                f"Multiclass VQC requires n_classes <= n_qubits. "
                f"Got n_classes={self.n_classes_}, "
                f"n_qubits={self.n_qubits}.")

        n_outputs = self.n_classes_
        dev, diff_method = self._get_device_and_diff()

        @qml.qnode(dev, interface="torch", diff_method=diff_method)
        def circuit(x, weights):
            self._circuit(x, weights)
            return tuple(
                qml.expval(qml.PauliZ(q))
                for q in range(n_outputs)
            )

        return circuit

    # ========================================================
    # SOFTMAX (kept from your original)
    # ========================================================

    @staticmethod
    def _softmax(values):
        values = np.asarray(values, dtype=float)
        values = values - np.max(values, axis=-1, keepdims=True)
        exp_v  = np.exp(values)
        return exp_v / np.sum(exp_v, axis=-1, keepdims=True)

    # ========================================================
    # FIT
    # ========================================================

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)

        self.classes_   = np.unique(y)
        self.n_classes_ = len(self.classes_)

        if X.shape[1] != self.n_qubits:
            raise ValueError(
                f"VQC expects {self.n_qubits} features "
                f"(one per qubit), got {X.shape[1]}.\n"
                f"For feature-level fusion set n_qubits=18.")

        # Scale → angles
        self.scaler_ = StandardScaler()
        X_sc  = self.scaler_.fit_transform(X)
        X_sc  = np.clip(X_sc, -3.0, 3.0)
        X_ang = (X_sc / 3.0 * np.pi).astype(np.float32)

        # Initialize weights
        rng = np.random.default_rng(self.random_state)
        w0  = rng.normal(0.0, 0.05,
                         size=(self.n_layers,
                               self.n_qubits, 2)).astype(np.float32)
        self.weights_ = torch.tensor(w0, requires_grad=True,
                                      device="cuda")  # GPU weights


        # Build correct qnode
        if self.n_classes_ == 2:
            self.qnode_ = self._build_binary_qnode()
        else:
            self.qnode_ = self._build_multiclass_qnode()

        # Optimizer
        optimizer = torch.optim.Adam(
            [self.weights_], lr=self.learning_rate)

        X_t = torch.tensor(X_ang, device="cuda")
        y_t = torch.tensor(y, dtype=torch.float32, device="cuda")


        n_samples    = len(X_t)
        rng2         = np.random.default_rng(self.random_state)
        best_loss    = np.inf
        best_weights = None
        patience_ctr = 0

        print(f"          Device  : {self.device_name}")
        print(f"          Qubits  : {self.n_qubits}")
        print(f"          Layers  : {self.n_layers}")
        print(f"          Classes : {self.n_classes_}")
        print(f"          Samples : {n_samples}")
        

        train_start = time.time()

        for epoch in range(self.epochs):

            idx  = rng2.permutation(n_samples)
            X_ep = X_t[idx]
            y_ep = y_t[idx]
            ep_losses = []

            for start in range(0, n_samples, self.batch_size):
                end     = min(start + self.batch_size, n_samples)
                X_batch = X_ep[start:end]
                y_batch = y_ep[start:end]

                optimizer.zero_grad()
                batch_losses = []

                for xi, yi in zip(X_batch, y_batch):

                    if self.n_classes_ == 2:
                        # ── Binary loss ───────────────────────
                        exp = self.qnode_(xi, self.weights_)
                        p1  = torch.clamp(
                            (exp + 1.0) / 2.0, 1e-6, 1.0 - 1e-6)
                        loss = -(yi * torch.log(p1)
                                 + (1.0 - yi) * torch.log(1.0 - p1))

                    else:
                        # ── Multiclass loss (softmax) ──────────
                        # WHY stack: qnode returns tuple of
                        # tensors, stack converts to single tensor
                        outputs = torch.stack(
                            list(self.qnode_(xi, self.weights_)))
                        probs = torch.softmax(outputs, dim=0)
                        # Cross-entropy: -log(p_true_class)
                        loss  = -torch.log(
                            probs[int(yi.item())] + 1e-8)

                    batch_losses.append(loss)

                batch_loss = torch.mean(torch.stack(batch_losses))
                batch_loss.backward()
                optimizer.step()
                ep_losses.append(batch_loss.item())

            ep_loss = float(np.mean(ep_losses))
            print(f"          Epoch {epoch+1:03d}/{self.epochs}"
                  f"  loss: {ep_loss:.5f}")

            if self.early_stopping:
                if ep_loss < best_loss - self.min_delta:
                    best_loss    = ep_loss
                    best_weights = self.weights_.detach().clone()
                    patience_ctr = 0
                else:
                    patience_ctr += 1
                    if patience_ctr >= self.patience:
                        print("          Early stopping triggered.")
                        break

        self.train_time_ = time.time() - train_start
        print(f"    Train time: {self.train_time_:.2f}s")


        if self.early_stopping and best_weights is not None:
            self.weights_ = best_weights.requires_grad_(True)

        return self

    # ========================================================
    # PREDICT PROBA
    # ========================================================

    def predict_proba(self, X):
        X     = np.asarray(X, dtype=np.float32)
        X_sc  = self.scaler_.transform(X)
        X_sc  = np.clip(X_sc, -3.0, 3.0)
        X_ang = (X_sc / 3.0 * np.pi).astype(np.float32)

        weights = self.weights_.detach().cpu()
        probs   = []

        for xi in X_ang:
            xi_t = torch.tensor(xi)

            if self.n_classes_ == 2:
                exp = float(self.qnode_(xi_t, weights))
                p1  = float(np.clip((exp + 1.0) / 2.0, 0.0, 1.0))
                probs.append([1.0 - p1, p1])

            else:
                # WHY softmax on expectation values:
                # PauliZ expectations ∈ [-1,1] are not
                # probabilities. Softmax converts them to a
                # valid probability distribution over classes.
                outputs = np.array(
                    [o.item() if hasattr(o, 'item')
                     else float(o)
                     for o in self.qnode_(xi_t, weights)],
                    dtype=float)
                probs.append(self._softmax(outputs))

        return np.array(probs, dtype=np.float32)

    # ========================================================
    # PREDICT
    # ========================================================

    def predict(self, X):
        probs   = self.predict_proba(X)
        indices = np.argmax(probs, axis=1)
        return self.classes_[indices]


# ============================================================
# FACTORY
# ============================================================

def make_vqc(params: Dict[str, Any]) -> VQCClassifier:
    return VQCClassifier(
        n_qubits      = params.get("n_qubits",        6),
        n_layers      = params.get("n_layers",         2),
        epochs        = params.get("epochs",           40),
        learning_rate = params.get("learning_rate", 0.01),
        batch_size    = params.get("batch_size",      32),
        early_stopping = params.get("early_stopping", True),
        patience      = params.get("patience",         8),
        min_delta     = params.get("min_delta",     1e-4),
        random_state  = params.get("random_state",    42),
        device_name   = params.get("device", "lightning.gpu"),
    )


CLASSIFIER_FACTORIES = {"vqc": make_vqc}


def create_quantum_classifier(
    classifier_name: str,
    classifier_config: Dict[str, Any],
) -> VQCClassifier:

    name = classifier_name.lower()

    if name not in CLASSIFIER_FACTORIES:
        raise ValueError(
            f"Unknown quantum classifier '{name}'. "
            f"Available: {list(CLASSIFIER_FACTORIES)}")

    return CLASSIFIER_FACTORIES[name](classifier_config)