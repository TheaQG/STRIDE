

# STRIDE Model Architecture

STRIDE uses a diffusion‑based generative model for climate downscaling.

---

# Core model

The architecture follows the **Elucidated Diffusion Model (EDM)** framework.

It uses a U‑Net backbone to predict diffusion scores.

---

# U‑Net structure

Components:

- encoder blocks
- decoder blocks
- skip connections
- residual blocks

Optional attention layers allow long‑range spatial interactions.

---

# Time embedding

Noise levels are embedded using sinusoidal time embeddings.

This allows the network to condition on the diffusion timestep.
