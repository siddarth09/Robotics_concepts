# Bayes Filter

A **Bayes filter** is a probabilistic framework used to estimate the state of a dynamic system over time, based on noisy sensor measurements and control inputs. It operates by recursively updating a belief (a probability distribution) of the system's state, considering both the uncertainty in motion (from control inputs) and the uncertainty in measurements (from sensors).

At the core of a Bayes filter is the concept of belief `bel(x_t)`, which represents the probability distribution over the state `x_t` at time `t`, given all prior control inputs and observations. The Bayes filter alternates between two phases:

### 1. Prediction Step:

The system's state is predicted based on a motion model and the control input `u_t`. This step updates the prior belief to account for the expected motion of the system but increases uncertainty due to potential errors in control.

\[
bel^-(x_t) = \int p(x_t \mid u_t, x_{t-1}) \cdot bel(x_{t-1}) \, dx_{t-1}
\]

Here, `p(x_t \mid u_t, x_{t-1})` is the probability of transitioning from `x_{t-1}` to `x_t` given the control input `u_t`.

### 2. Update Step:

In this step, the predicted belief is updated using the actual sensor measurement `z_t` to correct the estimate. This reduces uncertainty by incorporating real-world observations.

\[
bel(x_t) = \eta \cdot p(z_t \mid x_t) \cdot bel^-(x_t)
\]

Here, `p(z_t \mid x_t)` is the likelihood of observing `z_t` given the state `x_t`, and `\eta` is a normalization factor ensuring the belief sums to 1.

### Summary

By repeating these steps over time, the Bayes filter provides a way to continuously estimate the system's state in the presence of uncertainty. This method is fundamental for many applications in robotics and autonomous systems.
