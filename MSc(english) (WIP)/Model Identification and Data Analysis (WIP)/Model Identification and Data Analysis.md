
# Model Identification and data analysis

[Introduction to Model Identification and Data Analysis](src/01.Introduction%20to%20MIDA1.md)
[MA, AR and ARMA processes](src/02.MA,%20AR%20and%20ARMA%20processes.md)
[Frequency Domain](src/03.Frequency%20Domain.md)
[Prediction](src/04.Model%20Prediction.md)
[Model Identification](src/05.Model%20Identification.md)
[Model validation](src/06.Model%20validation.md)




The usual AR(1) process:
$$v(t) = av(t - 1) + \eta(t)$$

The optimal 1-step ahead predictor is
$$\widehat{W}_1(z) = \frac{az}{z - a}$$
Hence
$$\widehat{W}(z) = \frac{z}{z - a} \Rightarrow W_1(z) = \frac{az}{z} = a$$

Coming from a process
$$\hat{v}(t + 1|t) = av(t)$$
$$v(t + 1) = av(t) + \eta(t + 1) \text{ unpredictable}$$

The optimal 2-steps ahead predictor is
$$\widehat{W}_2(z) = \frac{a^2z}{z - a}$$
Coming from a process
$$v(t + 2) = av(t + 1) + \eta(t + 2) = a(av(t) + \eta(t + 1)) + \eta(t + 2)$$
$$= a^2v(t) + a\eta(t + 1) + \eta(t + 2) \text{ unpredictable}$$

By generalization
$$\hat{v}(t + r|t) = a^rv(t)$$



The **blocking property** of a system's transfer function refers to the phenomenon where a zero on the unit circle in the z-domain causes the system to completely attenuate a specific frequency component of the input signal. Formally, if a discrete-time system has a transfer function $H(z)$ with a zero at $z = e^{j\omega_0}$, then the frequency response of the system at that particular frequency $\omega_0$ is zero. 

Thus, for a given input signal \( x(t) \) with a frequency component at \( \omega_0 \), the output \( y(t) \) will have no component at this frequency, as it is "blocked" by the system.

In the context of the frequency response of a linear time-invariant (LTI) system, the effect of the poles and zeros of the transfer function \( H(z) \) can be summarized as follows:

- **Near the zeros**: The frequency response \( H(e^{j\omega}) \) will exhibit attenuation. For a zero on the unit circle at \( z = e^{j\omega_0} \), the frequency response is zero at \( \omega_0 \). As the frequency moves away from this zero, the attenuation decreases.

- **Near the poles**: The frequency response is amplified. If a pole is near the unit circle, it signifies a strong frequency response at the angle corresponding to the pole's position. The closer the pole is to the unit circle, the greater the amplification, leading to a resonant peak in the spectrum.

Hence, the frequency response of a system can be characterized as:

- Attenuated near zeros: \( |H(e^{j\omega})| \) is small near the angles of the zeros.
- Amplified near poles: \( |H(e^{j\omega})| \) is large near the angles of the poles.
