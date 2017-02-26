'''Krotov algorithm for Open Quantum Systems! - Version 2.0'''

''' 

d |PSI >> = (-i lambda_1 H_0 - lambda_2^2 L(C)) |PSI>>
-
dt

lambda_1 = lambda_2 ^2
'''

import matplotlib
import scipy.stats
import scipy.linalg
import scipy.sparse.linalg
import numpy as np
import matplotlib.pyplot as plt
import math
import numpy.random as rnd
import scipy
from scipy.integrate import ode


class Krotov:
	def __init__(self,k,n_bar,omega,lambda_1,t_i,t_f,num_t):
		# Hamiltonian 
		self.k = float(k)
		self.n_bar = float(n_bar)
		self.omega = float(omega)
		self.lambda_1 = float(lambda_1)
		self.lambda_2 = np.abs(np.sqrt(lambda_1))
		self.t_index = 0
		self.num_basis = 2

		#Sigma Matrices
		self.sigmaX = np.array([[0,1],[1,0]],dtype='complex')
		self.sigmaY = np.array([[0,-1.j],[1.j,0]],dtype='complex')
		self.sigmaZ = np.array([[1,0],[0,-1]],dtype='complex')
		self.I = np.array([[1,0],[0,1]],dtype='complex')
		self.sigmaMinus = self.sigmaX - 1.j * self.sigmaY
		self.sigmaPlus = self.sigmaX + 1.j * self.sigmaY

		# Time
		self.t_i = t_i
		self.t_f = t_f
		self.num_t = num_t
		self.t = np.linspace(self.t_i, self.t_f, self.num_t)
		self.dt = self.t[1] - self.t[0]

		# Parameters for Krotov
		self.delta = 1
		self.eta = 1
		self.alpha = .01

		# Epsilon
		self.Ex = rnd.rand(self.num_t)
		self.Ex_tilde = np.zeros((self.num_t,))


		# Hamiltonian of the system 
		self._H_s = 0.5 * self.sigmaZ * self.omega 

		# Density Matrix
		self.col_rho = np.zeros((self.num_t,self.num_basis**2),dtype='complex')
		self.col_chi = np.zeros((self.num_t,self.num_basis**2),dtype='complex')
		
		self._rho_init = self.rho_Init()
		self.col_rho[0,:] = self.matrix2col(self._rho_init)

		#Thermal or Target State
		self.thermal = np.array([self.n_bar/(2*self.n_bar + 1),0,0,(self.n_bar+1)/(2*self.n_bar + 1)],dtype='complex')
		self.target = self.thermal
		self.matrix_target_interaction =  np.dot(scipy.linalg.expm(1.j * 0.5*self.omega*self.sigmaZ * self.t[-1] ), np.dot(self.col2matrix(self.target),scipy.linalg.expm(-1.j * 0.5*self.omega*self.sigmaZ * self.t[-1] ) ) )
		self.target_interaction = self.matrix2col(self.matrix_target_interaction)

		#Time Independent Lindbladian
		self._H_1 = self.H_1()

	# Stacks the columns of a nXn matrix on top of other and makes a column
	def matrix2col(self,matrix):
		col = np.zeros((0,),dtype='complex')
		for i in range(2):
			col = np.append(col,matrix[:,i])

		return(col)

	def col2matrix(self,col):
		matrix = np.array([[col[0],col[2]],[col[1],col[3]]],dtype='complex')
		return (matrix)


	def rho_Init(self):
		#Z = rnd.rand(2,2) + 1.j * rnd.rand(2,2)
		#Z_dagger = np.transpose(np.conjugate(Z))

		#trace = np.trace(np.dot(Z,Z_dagger))

		#rho = np.dot(Z,Z_dagger) / trace

		rho = np.array([[1,0],[0,0]],dtype='complex')
		return(rho)

	def mu_prime(self,t_index):
		mu_prime =  - ( np.cos(self.omega * t_index * self.dt) * self.sigmaX - np.sin(self.omega * t_index * self.dt) * self.sigmaY )
		return (mu_prime)

	def mu(self,t_index):
		mu = np.kron(self.I,self.mu_prime(t_index)) - np.kron( np.conjugate(self.mu_prime(t_index)), self.I )
		return (mu)

	def H_0(self,t_index):
		return (self.mu(t_index) * self.Ex[t_index] )

	def H_0_tilde(self,t_index):
		return (self.mu(t_index) * self.Ex_tilde[t_index] )

	# L is Lindblad and C is a matrix    
	def L(self,C):
		return (np.kron( np.conjugate(C), C ) - 0.5 * ( np.kron(self.I,  np.dot(np.conjugate(np.transpose(C)), C) ) )   - 0.5 * ( np.kron(  np.dot(np.transpose(C), np.conjugate(C) ) , self.I) ) )

	# H_1 : Thermal Lindbladian in Liouville Form
	def H_1(self):
		return( self.L(np.sqrt(2*self.k*(1+self.n_bar)) * self.sigmaMinus) + self.L(np.sqrt(2*self.k*self.n_bar)* self.sigmaPlus) )

	# Final Liouville Space Hamiltonian

	def H(self,t_index):
		return(-1.j * self.lambda_1 * self.H_0(t_index) + self.lambda_2**2 * self._H_1)

	def H_tilde(self,t_index):
		return(-1.j * self.lambda_1 * self.H_0_tilde(t_index) - self.lambda_2**2 * self._H_1)

	'''
	The equation becomes 
	d |PSI >> = H(t) |PSI>>
	-
	dt

	We solve it by exponentiation!

	|PSI (t+dt)>> = exp(H(t) dt) |Psi(t)>>
	'''

	def update_Psi(self,t_index):
		t= t_index
		column = np.dot( scipy.linalg.expm(self.H(t) * self.dt), self.col_rho[t] )
		self.col_rho[t+1] = column
		


	def update_Chi(self,t_index):
		t=t_index
		column = np.dot( scipy.linalg.expm( np.conjugate(np.transpose(-self.H_tilde(t))) * -self.dt), self.col_chi[t] )
		self.col_chi[t-1] = column

	# Overlap operator with target state O|PSI>>

	def O(self,psi):
		return( np.vdot(self.target_interaction,psi) * self.target_interaction)

	# Returns <<PSI|O|PSI>>>
	def Overlap(self,psi):
		return( np.vdot(psi,self.O(psi)) )

	def update_Epsilon(self,t_index):
		t=t_index
		part1 = (1-self.delta) * self.Ex_tilde[t-1]
		part2 = -self.delta * self.lambda_1 * np.imag( np.vdot(self.col_chi[t-1], np.dot( self.mu(t), self.col_rho[t] ) ) )/ self.alpha
		self.Ex[t] = -(part1 + part2)

	def update_Epsilon_tilde(self,t_index):
		t=t_index
		part1 = (1-self.eta) * self.Ex[t]
		part2 = -self.eta * self.lambda_1 * np.imag( np.vdot(self.col_chi[t], np.dot( self.mu(t), self.col_rho[t] ) ) )/ self.alpha
		self.Ex_tilde[t] = -(part1 + part2)

	def evolution_Psi(self,string='not initial'):
		if string == 'initial':
			for t in range(0,self.num_t-1):
				self.update_Psi(t)
		else:
			for t in range(0,self.num_t-1):
				self.update_Epsilon(t)
				self.update_Psi(t)
			t = self.num_t-1
			self.update_Epsilon(t)

	def evolution_Chi(self):
		for t in range(self.num_t-1,0,-1):
			self.update_Epsilon_tilde(t)
			self.update_Chi(t)
		t = 0
		self.update_Epsilon_tilde(t)

	def Run_Krotov(self,num_iter):
		T = self.num_t-1
		self.evolution_Psi('initial')

		self.overlap = []

		for i in range(0,num_iter):
			print ("Iteration : ", i)
			self.col_chi[T] = self.O(self.col_rho[T])
			self.evolution_Chi()
			self.evolution_Psi()
			self.overlap.append(self.Overlap(self.col_rho[T]))

	def distance(self,t_index):
		t= t_index
		matrix = self.col2matrix(self.col_rho[t])
		diff = self.col2matrix(self.target_interaction) - matrix
		trace = np.trace( scipy.linalg.sqrtm(np.dot(diff,np.conjugate(np.transpose(diff)))) )
		return(0.5*trace)


if __name__ == '__main__':
	# INTITIALIZE : k,n_bar,omega,lambda_1,t_i,t_f,t_num

	k = Krotov(.01,1,10,1,0.1,10,5000)
	k.Run_Krotov(10)
	s = Krotov(.01,1,10,1,0.1,10,5000)
	s.Run_Krotov(0)

	plt.figure(1)
	plt.title('Overlap Vs Iteration')
	plt.ylabel('Overlap')
	plt.xlabel('Iteration')
	plt.plot(np.abs(k.overlap))

	dist1 = []
	dist2 = []

	for i in range(0,len(s.t)):
		dist1.append(np.real(s.distance(i)))
	
	for i in range(0,len(k.t)):
		dist2.append(np.real(k.distance(i)))

	plt.figure(2)
	plt.title('Distance')
	plt.ylabel('Distance')
	plt.xlabel('Time')
	plt.plot(s.t,dist1)
	plt.plot(k.t,dist2,'r')
	

	plt.show()


