"""Simulated BCI intent channel (DAY 2).

The channel is NOT real EEG. It is an abstract stand-in for a decoded-intent bottleneck:
 - a limited discrete vocabulary of K high-level commands (target object x goal zone),
 - a confusion matrix parameterised by decoder accuracy,
 - a rate limit on how often a command can be issued.
Optionally, the confusion matrix can later be estimated from a real MI decoder
(e.g. BCI Competition IV 2a) and loaded via `from_confusion_matrix`.
"""
from __future__ import annotations

import numpy as np


class IntentChannel:
    def __init__(self, vocab_size: int, accuracy: float, neighbor_bias: float = 0.7,
                 min_steps_between_commands: int = 1, seed: int = 0):
        self.K = vocab_size
        self.accuracy = accuracy
        self.neighbor_bias = neighbor_bias
        self.min_gap = min_steps_between_commands
        self.rng = np.random.default_rng(seed)
        self.confusion = self._build_confusion()

    def _build_confusion(self) -> np.ndarray:
        """Row i = P(decoded | intended=i). Errors biased toward adjacent indices."""
        K, acc, nb = self.K, self.accuracy, self.neighbor_bias
        C = np.zeros((K, K))
        for i in range(K):
            C[i, i] = acc
            err = 1.0 - acc
            if K == 1:
                continue
            neigh = sorted({(i - 1) % K, (i + 1) % K} - {i})
            others = [j for j in range(K) if j != i and j not in neigh]
            if others:
                for j in neigh:
                    C[i, j] += err * nb / len(neigh)
                for j in others:
                    C[i, j] += err * (1 - nb) / len(others)
            else:
                for j in range(K):
                    if j != i:
                        C[i, j] = err / (K - 1)
        return C / C.sum(axis=1, keepdims=True)

    @classmethod
    def from_confusion_matrix(cls, C: np.ndarray, seed: int = 0) -> "IntentChannel":
        ch = cls(vocab_size=C.shape[0], accuracy=float(np.mean(np.diag(C))), seed=seed)
        ch.confusion = C / C.sum(axis=1, keepdims=True)
        return ch

    def decode(self, intended: int) -> int:
        """Sample a decoded command given the user's true intended command."""
        return int(self.rng.choice(self.K, p=self.confusion[intended]))

    def belief(self, decoded: int, prior=None) -> np.ndarray:
        """Posterior over the true intent given one decoded command (Bayes rule)."""
        prior = np.ones(self.K) / self.K if prior is None else prior
        post = self.confusion[:, decoded] * prior
        return post / post.sum()


def command_to_goal(command: int, n_objects: int, n_zones: int):
    """Map high-level command index -> (target_obj, goal_zone).

    Ordering cycles through objects first, then zones, so every object is reachable even for small
    vocabularies: K=3 -> zone 0 for all objects, K=6 -> zones 0-1, K=9 -> all 3x3 pairs (n_objects=3, n_zones=3).
    The vocabulary size K therefore equals the number of distinct tasks the simulated user can request.
    """
    return command % n_objects, (command // n_objects) % n_zones


class IntentSession:
    """Interaction protocol between a (simulated) BCI user and the planner.

    The user has ONE fixed intended command and re-issues it; the decoder returns a noisy command each time
    (IntentChannel). The system keeps a Bayesian belief over the K possible intents and COMMITS once the
    belief of the most likely intent reaches `threshold` (or after `max_commands`). After committing the
    user stops issuing commands and the planner executes autonomously. Commands issued is the bandwidth
    metric. Setting threshold <= 1/K... i.e. 0.0 commits on the first decoded command (no confirmation).
    """

    def __init__(self, channel: IntentChannel, intended: int, threshold: float = 0.9, max_commands: int = 6):
        self.channel, self.intended = channel, intended
        self.threshold, self.max_commands = threshold, max_commands
        self.b = np.ones(channel.K) / channel.K
        self.n_commands = 0
        self.committed = None

    def issue(self) -> int:
        d = self.channel.decode(self.intended)
        self.n_commands += 1
        self.b = self.channel.belief(d, prior=self.b)
        if self.b.max() >= self.threshold or self.n_commands >= self.max_commands:
            self.committed = int(np.argmax(self.b))
        return d

    def map_command(self) -> int:
        return int(np.argmax(self.b))
