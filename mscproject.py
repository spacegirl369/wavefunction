import streamlit as st

st.title("What is Wave Function?")

st.write("Quantum mechanics is a branch of science that deals with the study and behavior of matter as well as light. The wave function in quantum mechanics can be used to illustrate the wave properties of a particle. Therefore, a particle’s quantum state can be described using its wave function. ")

st.write("This interpretation of wave function helps define the probability of the quantum state of an element as a function of position, momentum, time, and spin. It is represented by a Greek alphabet Psi, 𝚿.")

st.write("However, it is important to note that there is no physical significance of the wave function itself. Nevertheless, its proportionate value of " + "Ψ" + "² at a given time and point of space does have physical importance.")
st.title("Schrödinger Equation")

st.write("In 1925, Erwin Schrödinger introduced this partial differential equation for wave function definition as a reward to the Quantum mechanics branch. According to him, the wave function can be satisfied and solved. Here is a time-independent and time-dependent equation of Schrödinger shown in the image below.")

st.write("The time-independent Schrödinger equation is given by:")
st.latex('- \\frac{\\hbar^2}{2m}\\nabla^2 \\psi(\\vec{r}) + V(\\vec{r}) \\psi(\\vec{r}) = E \\psi(\\vec{r})')
st.write("The time-dependent Schrödinger equation is given by:")
st.latex(r'''i \hbar \frac{\partial}{\partial t} \Psi(\vec{r},t) = -\frac{\hbar^2}{2m} \nabla^2 + V(\vec{r},t)''')

st.write("In the above equations, ")

bullet_points = """
- m refers to the particle’s mass.
- ∇ is laplacian.
- h equals to h/2π, which is also known as the reduced Planck’s constant.
- i is the imaginary unit.
- E is a constant matching energy level of a system
"""

st.write(bullet_points)

st.subheader("**Properties of Wave Function**")

bullet_points = """
- ψ must be normalizable, i.e. integral of |ψ|² over all space must be finite.
- ψ must be continuous and single-valued every where.
- partial derivatives of ψ, i.e. ∂ψ/∂x, ∂ψ/∂y, ∂ψ/∂z must be continuous and single-valued everywhere.
- ψ must be normalizable, which means that ψ must go to 0 as x→±∞, y→±∞, z→±∞.
"""

st.write(bullet_points)
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
st.write("**These are some examples of non-well-behaved wavefunctions:**")

psi = lambda x: x**2
x = np.linspace(-10, 10, 100)
y = psi(x)

fig, ax = plt.subplots(figsize=(6, 3))
ax.plot(x, y)
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 200)
ax.text(0, 220, " (1) ψ = x²", fontsize=14, fontweight="bold")
st.pyplot(fig)
st.write("**ψ = x²** is not a well-behaved wavefunction.")
st.write("Reasons:")
st.write("**(1)** When x → ±∞, the value of ψ = x² or y-value of the above graph also goes to infinity, which violates the primary condition for a well-behaved wave functions.")
st.write("**(2)** It is also non-square integral, i.e. ∫₋|ψ(x)|² dx is also infinite. So, it is not a physically acceptable or well-behaved wave function.")


x = np.linspace(-np.pi, np.pi, 1000)
y = 2 * np.tan(x)

fig, ax = plt.subplots(figsize=(6, 3))
ax.plot(x, y)
ax.set_xlim(-np.pi, np.pi)
ax.set_ylim(-10, 10)
ax.text(0, 15, " (2) ψ = 2 tanx", fontsize=14, fontweight="bold")
st.pyplot(fig)

st.write("**ψ = 2 tanx** is not finite as the value of tanx goes infinity at x of an odd multiple of π/2. This violates the primary condition for the well-behaved wave function.")

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# Define the wave function
def wave_function(x):
    return np.exp(-x**2)

# Generate an array of x values
x = np.linspace(-5, 5, 100)

# Calculate the y values of the wave function
y = wave_function(x)

# Plot the wave function
fig, ax = plt.subplots()
ax.plot(x, y, label=r'$\psi = e^{-x^2}$')
ax.set_xlabel('x')
ax.set_ylabel('$\psi(x)$')
ax.set_title('Well-Defined Wave Function: ψ = e^−x^2')
ax.legend()
ax.grid()
st.pyplot(fig)

# Explain why this wave function is well-defined
st.write("The wave function ψ = e^−x^2 is an allowed wavefunction since its derivative is continuous. This means that the wave function does not exhibit any oscillatory or divergent behavior, which is often a sign of a poorly-defined wave function.")
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps
import sympy as sp

def parse_and_eval(user_input, x_symbol, x_values):
    try:
        expr = sp.sympify(user_input)
        y = np.vectorize(sp.lambdify(x_symbol, expr, 'numpy'))(x_values)
        return y
    except Exception as e:
        st.error(f"Error in evaluating the wavefunction: {e}")
        return None

def normalize_wavefunction(x_values, y):
    norm = simps(np.abs(y)**2, x_values)
    return x_values, y / np.sqrt(norm)

def plot_wavefunction_2d(x, y, title):
    fig, ax = plt.subplots()
    ax.plot(x, np.real(y), label='Real part')
    ax.plot(x, np.imag(y), label='Imaginary part')
    ax.set_title(title)
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Function value')
    ax.legend()
    return fig

def main():
    st.title(" Check Wavefunction Normalization")

    # Take user input for the wavefunction
    user_input = st.text_area("Enter the wavefunction (e.g., sin(x), tan(x), abs(x), sin(x)*cos(x)):", "exp(-x**2)")

    # Define the range of x values
    x_values = np.linspace(-5, 5, 400)

    # Symbolic variable for x
    x_symbol = sp.symbols('x')

    # Parse and evaluate the user input
    y = parse_and_eval(user_input, x_symbol, x_values)

    if y is not None:
        # Check if the wavefunction is normalized
        norm = simps(np.abs(y)**2, x_values)
        is_normalized = np.isclose(norm, 1.0, atol=1e-4)

        # Normalize the wavefunction if not already normalized
        if not is_normalized:
            x_values, y = normalize_wavefunction(x_values, y)

        # Plot the wavefunction in 2D
        fig = plot_wavefunction_2d(x_values, y, "Wavefunction")
        st.pyplot(fig)

        # Display normalization status
        st.write("Normalization Status:", "Normalized" if is_normalized else "Not Normalized")

if __name__ == "__main__":
    main()

import streamlit as st
st.title("Quantum Bound states")
st.title("(1) 1 dimensional infinite potential well")

# Write the introduction to the 1D infinite potential well
st.write("A one-dimensional (1D) infinite potential well, also known as the infinite square well, is a common problem in quantum mechanics used to illustrate the differences between classical and quantum systems. It describes a particle free to move in a small space surrounded by impenetrable barriers.")

st.write("In classical mechanics, a particle trapped inside a large box can move at any speed within the box and it is no more likely to be found at one position than another. However, when the well becomes very narrow (on the scale of a few nanometers), quantum effects become important. The particle may only occupy certain positive energy levels and can never have zero energy, meaning that the particle can never 'sit still'.")

st.write("Additionally, it is more likely to be found at certain positions than at others, depending on its energy level, and may never be detected at certain positions, known as spatial nodes.")

# Display the wave function of the particle in the 1D infinite potential well
st.write("The wave function of the particle in a 1D infinite potential well is given by:")
st.latex(r'''
	\psi(x,t) = A\sin(kx)e^{-i\omega t} + B\cos(kx)e^{-i\omega t}
''')
st.write("where $A$ and $B$ are arbitrary complex numbers, $k$ is the wavenumber, $ω$ is the angular frequency, and $x$ and $t$ are the position and time, respectively.")

st.write("The frequency of the oscillations through space and time is given by the expression:")
st.latex(r'''
	\omega = \sqrt{\frac{2E}{m}}
''')
st.write("where $E$ is the total energy of the particle and $m$ is its mass.")

st.write("The amplitude of the wavefunction at a given position is related to the probability of finding a particle there by:")
st.latex(r'''
	P(x,t) = |\psi(x,t)|^2
''')

# Display the boundary conditions for the wave function
st.write("The wavefunction must vanish everywhere beyond the edges of the box, and the amplitude of the wavefunction may not 'jump' abruptly from one point to the next. These conditions are satisfied by wavefunctions with the form:")
st.latex(r'''
	\psi(x,t) = \sqrt{\frac{2}{L}}\sin(kx)e^{-i\omega t}
''')
st.write("where $L$ is the width of the well.")

# Display the energy of the particle in the well
st.write("The energy of the particle in the well is given by:")
st.latex(r'''
	E = \frac{\hbar^2 k^2}{2m}
''')
st.write("where $\hbar$ is the reduced Planck constant. The allowed values of $k$ are given by:")
st.latex(r'''
	k = \frac{n\pi}{L}, \quad n = 1, 2, 3, \ldots
''')
st.write("So the allowed energy levels are:")
st.latex(r'''
	E_n = \frac{n^2\hbar^2\pi^2}{2mL^2}, \quad n = 1, 2, 3, \ldots
''')

# Display the relation between energy and momentum
st.write("It is expected that the eigenvalues, i.e., the energy $E_n$ of the box should be the same regardless of its position in space, but $\\psi_n(x,t)$ changes. Notice that $x_c-L/2$ represents a phase shift in the wave function. This phase shift has no effect when solving the Schrödinger equation, and therefore does not affect the eigenvalue.")

# Parameters
L = 1  # length of the box
m = 1  # mass of the particle
n = 5  # quantum number

# Wave function
x = np.linspace(0, L, 100)
psi = np.sqrt(2/L) * np.sin(n * np.pi * x / L)

# Energy level
hbar = 1.0545718e-34  # in Joule seconds
E = n**2 * (np.pi**2 * hbar**2) / (2 * m * L**2)

# Probability density
prob_density = psi**2

# Plot
fig, ax = plt.subplots()
ax.plot(x, prob_density, label=f"n = {n}, E = {E:.2f} eV")
ax.set_xlabel("x")
ax.set_ylabel("Probability density")
ax.set_title("1D Infinite Potential Well")
ax.legend()
plt.show()
st.pyplot(fig)
st.write(" Below are the plot of wavefunction (figure 3) and corresponding probablity density for n=1 to n=4 ")
st.image('infinite.png')

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
st.title("(2)Harmonic Oscillator")
import streamlit as st

st.write(
    "A first step toward a quantum formulation is to use the classical expression ",
    "k=mω^2",
    " to limit mention of a ",
    "‘spring’",
    " constant between the atoms. In this way the potential energy function can be written in a more general form,",
    "U(x)=12mω^2x^2.",
)
st.latex(r"U(x) = \frac{1}{2} m \omega^2 x^2.")
st.subheader("Time-Independent Schrödinger Equation")
st.latex(
    r"""
    -\frac{\hbar^2}{2m} \frac{d^2\psi(x)}{dx^2} + \frac{1}{2} m \omega^2 x^2 \psi(x) = E\psi(x)
    """
)
st.subheader("Solving the Time-Independent Schrödinger Equation")
st.write(
    "To solve this equation, we require the wavefunctions to be symmetric about ",
    "x=0",
    " (the bottom of the potential well) and to be normalizable. These conditions ensure that the probability density ",
    "|ψ(x)|^2",
    " must be finite when integrated over the entire range of x from ",
    "-∞",
    " to ",
    "+∞",
    "."
)
st.latex(r"|\psi(x)|^2 \text{ must be finite when integrated over the entire range of } x \text{ from } -\infty \text{ to } +\infty.")
st.subheader("Allowed Energies and Wavefunctions")
st.write(
    "The allowed energies are ",
    "E_n = (n + 1/2)ℏω",
    " with ",
    "n=0,1,2,3,...",
    ".",
    "The wavefunctions that correspond to these energies (the stationary states or states of definite energy) are ",
    "ψ_n(x) = N_n e^{-β^2x^2/2} H_n(βx)",
    ".",
    "where ",
    "β=mω/ℏ",
    " and ",
    "N_n",
    " is the normalization constant, and ",
    "H_n(y)",
    " is a polynomial of degree ",
    "n",
    " called a Hermite polynomial."
)
st.latex(r"E_n = \left(n + \frac{1}{2}\right)\hbar\omega \quad \text{with} \quad n=0,1,2,3,\ldots")
st.latex(r"\psi_n(x) = N_n e^{-\beta^2x^2/2} H_n(\beta x) \quad \text{where} \quad \beta=\frac{m\omega}{\hbar} \quad \text{and} \quad N_n \text{ is the normalization constant}.")
st.subheader("First Four Hermite Polynomials")
st.write(
    "The first four Hermite polynomials are ",
    "H_0(y)=1",
    ", ",
    "H_1(y)=2y",
    ", ",
    "H_2(y)=4y^2-2",
    ", ",
    "H_3(y)=8y^3-12y."
)
st.latex(r"H_0(y) = 1, \quad H_1(y) = 2y, \quad H_2(y) = 4y^2-2, \quad H_3(y) = 8y^3-12y.")
st.image('hoscillator.png')


st.title("(3) Solving the Schrodinger equation for the hydrogen atom using numerical method")
st.markdown("The Schrödinger equation is a fundamental equation in quantum mechanics that describes the behavior of particles at the quantum level. Solving the Schrödinger equation for the hydrogen atom provides insight into the energy levels and wavefunctions of the electron orbiting the nucleus. This information is important for understanding the behavior of hydrogen atoms in various physical and chemical systems.For example, the energy levels of the hydrogen atom can be used to explain the emission and absorption spectra of hydrogen gas, which are important in astrophysics and spectroscopy. The wavefunctions can be used to calculate the probability density of the electron, which can provide insight into the spatial distribution of the electron around the nucleus. This information is important for understanding chemical reactions and bonding in molecules containing hydrogen atoms.Additionally, solving the Schrödinger equation numerically using the finite difference method as implemented in this simulation can be a useful exercise for learning about numerical methods in physics and engineering. It can also be used as a starting point for solving more complex quantum mechanical problems")
st.write("This simulation calculates the energy eigenvalues and wavefunctions of the hydrogen atom by solving the Schrödinger equation numerically using the finite difference method.The user inputs the number of points in the discretized space, the orbital quantum number l, and the minimum and maximum radii of the discretized space in nanometers.Then it calculates the Hamiltonian matrix for the Schrödinger equation using the finite difference method and the user-specified parameters.The Hamiltonian matrix is then diagonalized numerically using the eigs function from the SciPy library to obtain the energy eigenvalues and wavefunctions. The energy eigenvalues are converted from joules to electron volts and displayed to the user. The wavefunctions are squared to obtain the probability densities and displayed as a plot.Note that it assumes that the user will input valid quantum numbers (n > 0, 0 <= l < n, -l <= m <= l). If invalid inputs are given, the code may produce unexpected results or errors. This should be noted in the user interface.")

N = st.number_input("Number of points in the discretized space", value=2000)
l = st.number_input("Orbital quantum number", value=0)
r_min = st.number_input("Minimum radius of the discretized space (in nm)", value=2)
r_max = st.number_input("Maximum radius of the discretized space (in nm)", value=20)

if st.button("Calculate"):
    r = np.linspace(r_min * 1e-9, r_max * 1e-9, N, endpoint=False)
    hamiltonian = build_hamiltonian(r, l)

    number_of_eigenvalues = 30
    eigenvalues, eigenvectors = eigs(hamiltonian, k=number_of_eigenvalues, which='SM')

    eigenvectors = np.array([x for _, x in sorted(zip(eigenvalues, eigenvectors.T), key=lambda pair: pair[0])])
    eigenvalues = np.sort(eigenvalues)

    st.write(eigenvalues.real / e)

    densities = [np.absolute(eigenvectors[i, :]) ** 2 for i in range(len(eigenvalues))]