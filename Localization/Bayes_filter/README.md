# Bayes Filter

A **Bayes filter** is a probabilistic framework used to estimate the state of a dynamic system over time, based on noisy sensor measurements and control inputs. It operates by recursively updating a belief (a probability distribution) of the system's state, considering both the uncertainty in motion (from control inputs) and the uncertainty in measurements (from sensors).

At the core of a Bayes filter is the concept of belief `bel(x_t)`, which represents the probability distribution over the state `x_t` at time `t`, given all prior control inputs and observations. The Bayes filter alternates between two phases:

### 1. Prediction Step:

The system's state is predicted based on a motion model and the control input `u_t`. This step updates the prior belief to account for the expected motion of the system but increases uncertainty due to potential errors in control.

![prediction step](https://github.com/user-attachments/assets/3e29fd3d-46a2-44b3-8857-b02438ee9dd4)
Here, `p(x_t \mid u_t, x_{t-1})` is the probability of transitioning from `x_{t-1}` to `x_t` given the control input `u_t`.


![bayes_fil1](https://github.com/user-attachments/assets/80aca5c6-e55d-4f6c-a3f8-f475763a2ff9)

### 2. Update Step:

In this step, the predicted belief is updated using the actual sensor measurement `z_t` to correct the estimate. This reduces uncertainty by incorporating real-world observations.


![update step](https://github.com/user-attachments/assets/1552fd16-0d52-450d-8b58-7f4bebc857a5)

Here, `p(z_t \mid x_t)` is the likelihood of observing `z_t` given the state `x_t`, and `\eta` is a normalization factor ensuring the belief sums to 1.


![bayes_fil2](https://github.com/user-attachments/assets/035915f6-ce4c-4a8e-b099-b2f19780884b)

### Summary

By repeating these steps over time, the Bayes filter provides a way to continuously estimate the system's state in the presence of uncertainty. This method is fundamental for many applications in robotics and autonomous systems.





