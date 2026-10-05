"""Visualisation (DAY 6): rollout videos and plots."""
import imageio.v2 as imageio


def save_video(frames, path: str, fps: int = 20) -> None:
    imageio.mimsave(path, frames, fps=fps)

# TODO: success-vs-accuracy plot, commands-per-task plot, latent diagnostics (effect-code clusters).
