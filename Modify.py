# Implementation of a General Krotov Algorithm

import matplotlib
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt
import math


# Set Up the Krotov Class


class SetUpKrotov:
    def __init__(self, x_i, x_f, dx, t_i, t_f, dt, alpha, delta, eta):
        # Discretise Lattice and Time
        self.x_i = x_i
        self.x_f = x_f
        self.dx = dx
        self.t_i = t_i
        self.t_f = t_f
        self.dt = dt

        self.delta = delta
        self.eta = eta
        self.alpha = alpha

        self.num_x = int((x_f - x_i) / dx)
        self.num_t = int((t_f - t_i) / dt)

        # Array of Space and Time
        self.x = np.linspace(self.x_i, self.x_f, self.num_x)
        self.t = np.linspace(self.t_i, self.t_f, self.num_t)

        # Fourier Space
        self.dk = 2 * np.pi / (self.num_x * self.dx)
        self.k_i = -0.5 * self.num_x * self.dk
        self.k = self.k_i + np.arange(self.num_x) * self.dk

        # Current Time Index
        self.t_index = 0

        self.Psi = np.zeros((self.num_t, self.num_x), dtype='complex')
        self.Chi = np.zeros((self.num_t, self.num_x), dtype='complex')

        # Identity Ket (1,1,1,1......,1)
        self.Iket = np.ones((self.num_x,), dtype='float')

        # Specify the initial wave function as |Psi> = Sigma ( Ci |xi> )
        self.Psi[0] = self.Psi_init_Morse()

        # Specify the final wave function as |Psi> = Sigma ( Ci |xi> )
        # self.Psi[self.num_t-1] = self.Psi_final()

        # |Chi> = O|Psi>
        self.Chi[self.num_t - 1] = self.O(self.Psi[self.num_t - 1])

        # Definition of mu(x)
        self._mu = self.mu(self.Iket)

        # Definition of E(t) (Epsilon and Epsilon`)
        self.E = self.E_init()
        self.Etilda = self.E_init()
        self.Update_Etilda()

        # Definition of V
        self._V = self.V()

        # Plotting Overlap
        self.overlap = np.array([], dtype='complex')

    def Ho(self, ket):
        ket = self.Fourier(ket)

        operator = self.k ** 2 / (2 * self.m)

        ket = operator * ket

        ket = self.InvFourier(ket)

        return (ket)

    def O(self, ket):
        gamma = 25.0
        x_dash = 2.5

        func = np.zeros((0,), dtype='complex')

        for i in range(0, len(ket)):
            val = float(gamma * np.exp(- (gamma * (self.x[i] - x_dash)) ** 2) / np.sqrt(np.pi))

            func = np.append(func, val * ket[i])

        return (func)

    # Ground State of Morse Potential
    def Psi_init_Morse(self):
        x_e = 1.821
        beta = 1.189
        D = 0.1994
        #        self.m = 12000
        #        lamda = np.sqrt(2*self.m*D)/beta
        lamda = 27.4
        self.m = ((beta ** 2) * (lamda ** 2)) / (2 * D)

        ar_x_e = x_e * np.ones(self.num_x)
        z = 2 * lamda * np.exp(-beta * (self.x - ar_x_e))
        N = ((2 * lamda - 1) / (math.gamma(2 * lamda))) ** 0.5

        psi = N * np.power(z, lamda - 0.5) * np.exp(-0.5 * z)

        return (psi)

    def Psi_init_Gauss(self):
        mu = 1.9
        sigma = 0.04
        scale = 1. / 10
        p0 = 0.

        coeff = np.zeros((0,), dtype='complex')

        for i in self.x:
            val = scale * scipy.stats.norm(mu, sigma).pdf(i) * np.exp(1.j * p0 * i)

            coeff = np.append(coeff, val)

        return (coeff)

    def Psi_final(self):
        mu = 2.5
        sigma = 0.1

        coeff = np.zeros((0,), dtype='complex')

        for i in self.x:
            val = scipy.stats.norm(mu, sigma).pdf(i)

            coeff = np.append(coeff, val)

        return (coeff)

    def mu(self, ket):
        mu0 = 3.088
        xStar = 0.6

        func = np.zeros((0,), dtype='complex')

        for i in range(0, len(ket)):
            val = mu0 * self.x[i] * np.exp(-self.x[i] / xStar)

            func = np.append(func, val * ket[i])

        return (func)

    def E_init(self):
        E = np.zeros((0,), dtype='complex')

        for i in self.t:
            E = np.append(E, 0)

        return (E)

    def V(self):
        D = 0.1994
        beta = 1.189
        x0 = 1.821

        func = np.zeros((0,), dtype='complex')

        for i in self.x:
            val = D * ((np.exp(-beta * (i - x0)) - 1) ** 2) - D
            func = np.append(func, val)

        return (func)

    # Fourier Transform of State
    def Fourier(self, ket):
        N = self.num_x
        psiX_mod = ket * self.dx * np.exp(-1.j * self.k[0] * self.x) / np.sqrt(2 * np.pi)
        psiK_mod = np.fft.fft(psiX_mod)
        psiK = psiK_mod * np.exp(1.j * self.x[0] * self.dk * np.arange(N))

        return (psiK)

        # Inverse Fourier Transform of State

    def InvFourier(self, ket):
        N = self.num_x
        psiK_mod = ket * np.exp(-1.j * self.x[0] * self.dk * np.arange(N))
        psiX_mod = np.fft.ifft(psiK_mod)
        psiX = psiX_mod * np.exp(1.j * self.k[0] * self.x) * np.sqrt(2 * np.pi) / self.dx

        return (psiX)

    # Function for diagonalizing all parts of the Hamiltonian together (In the morse-potential example it is already diagonalized !)

    # This function changes the values of self.H and also the state ket into to their eigen vector representations !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    def diagonalize(self):

        return (0)

    # Compute the Commutator [mu,Ho]|ket>
    def commutator(self, ket):
        newKet = self.mu(self.Ho(ket)) - self.Ho(self.mu(ket))
        return (newKet)

        # Updates Etilda at t0 to t0 - dt/2

    def Update_Etilda(self):
        ket = self.Psi[self.t_index]
        bra = self.Chi[self.t_index]

        ket_new = self.mu(ket) + (self.dt * (1.j / 2) * self.commutator(ket))

        part2 = - np.imag(np.vdot(bra, ket_new)) * (self.eta / self.alpha)

        part1 = (1 - self.eta) * self.E[self.t_index]

        self.Etilda[self.t_index] = part1 + part2

        return (0)

    # Updates E at t0 to t0 + dt/2
    def Update_E(self):
        ket = self.Psi[self.t_index]
        bra = self.Chi[self.t_index]

        ket_new = self.mu(ket) - (self.dt * (1.j / 2) * self.commutator(ket))

        part2 = - np.imag(np.vdot(bra, ket_new)) * (self.delta / self.alpha)

        part1 = (1 - self.delta) * self.Etilda[self.t_index]

        self.E[self.t_index] = part1 + part2

        return (0)

    #### Updating Both the States and Co-States (Assuming Diagonalization is done before)####


    # Updates Chi at t0 to t0-dt or Psi to t0+dt (string for 'Chi' or 'Psi'
    def Update_State(self, string):
        if string == 'Chi':
            ket = self.Chi[self.t_index]

            # Call diaglonalize here ! if not diagonalized yet !

            ket = self.Fourier(ket)
            ket = np.exp(1.j * self.k ** 2 / (2 * self.m) * self.dt / 2) * ket
            ket = self.InvFourier(ket)

            ket = np.exp(1.j * (self._V - self._mu * self.Etilda[self.t_index]) * self.dt) * ket

            ket = self.Fourier(ket)
            ket = np.exp(1.j * self.k ** 2 / (2 * self.m) * self.dt / 2) * ket
            ket = self.InvFourier(ket)

            self.Chi[self.t_index - 1] = ket

        elif string == 'Psi':
            ket = self.Psi[self.t_index]

            # Call diaglonalize here ! if not diagonalized yet !

            ket = self.Fourier(ket)
            ket = np.exp(- 1.j * self.k ** 2 / (2 * self.m) * self.dt / 2) * ket
            ket = self.InvFourier(ket)

            ket = np.exp(- 1.j * (self._V - self._mu * self.E[self.t_index]) * self.dt) * ket

            ket = self.Fourier(ket)
            ket = np.exp(- 1.j * self.k ** 2 / (2 * self.m) * self.dt / 2) * ket
            ket = self.InvFourier(ket)

            self.Psi[self.t_index + 1] = ket

    # IMPLEMENTING KROTOV ALGORITHM

    def Start_Krotov(self, num_iter):
        T_index = self.num_t - 1  # Final time T

        for t in range(0, self.num_t - 1):
            self.t_index = t
            self.Update_State('Psi')

        for i in range(0, num_iter):

            self.Chi[T_index] = self.O(self.Psi[T_index])

            for t in range(self.num_t - 1, 0, -1):  # Decreasing Time for Chi
                self.t_index = t
                self.Update_Etilda()
                self.Update_State('Chi')

            self.t_index = 0
            self.Update_Etilda()

            for t in range(0, self.num_t - 1):  # Increasing Time for Psi
                self.t_index = t
                self.Update_E()
                self.Update_State('Psi')

            self.t_index = self.num_t - 1
            self.Update_E()

            self.overlap = np.append(self.overlap, self.Overlap_Integral())
            print(i, "th iteration....\t  <Psi|O|Psi>  = ", self.Overlap_Integral())

    def Overlap_Integral(self):
        T_index = self.num_t - 1

        ket = self.Psi[T_index]
        bra = ket
        newKet = self.O(ket)

        val = np.vdot(bra, newKet)

        return (val)

    # Cost Function J
    def J(self):
        integration = scipy.integrate.simps(self.E ** 2, x=self.t, dx=self.dt)

        J = self.Overlap_Integral() - integration

        return (J)


if __name__ == "__main__":
    k = SetUpKrotov(1.25, 5, 0.01, 0, 100 * 1000, 1, 1, 1, 1)
    k.Start_Krotov(3)
    plt.plot(k.overlap)
    plt.show()

# for i in range(0,k.num_t,200) : plt.plot(k.x,np.absolute(k.Psi[i]))
