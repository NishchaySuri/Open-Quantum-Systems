'''
Implementation of an Open Quantum System Spin Chain : Solved with Krotov.

Author : Nishchay Suri
Email : surinishchay@gmail.com
Course : M.Sc. Physics II
University : Indian Institute of Technology Bombay

'''
import matplotlib
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt
import math


# Set up a Class to compute Hamiltonian for a N atom spin chain with 2^N basis vectors.

class SetUpEquation:
    def __init__(self,N,k,n_bar,omega,Jx,Jy):
        self.N = N
        self.k = k
        self.n_bar = n_bar
        self.omega = omega
        self.Jx = Jx
        self.Jy = Jy
        self.num_basis = pow(2,N)

    # Takes input as an array and generates 2^N dim. sigma matrix by taking the tensor product.

    def Generate_Matrix(self,array,string):
        # array contains 0's and 1's to indicate I and M respectively in the tensor product.

        I = np.array(([1,0],[0,1])) # Identity Matrix

        if string == 'sigmaX':              # Sigma X
            X = np.array(([0,1],[1,0]))
            if array[0] == 0 :
                M = I
            elif array[0] == 1 :
                M=X

            for i in range(1,len(array)):    
                if(array[i] == 0):
                    M = np.kron(M,I)
                elif(array[i] == 1):
                    M = np.kron(M,X)
                    
            return(M)

        if string == 'sigmaY':              # Sigma Y
            Y = np.array(([0,-1.j],[1.j,0]))
            if array[0] == 0 :
                M = I
            elif array[0] == 1 :
                M=Y

            for i in range(1,len(array)):    
                if(array[i] == 0):
                    M = np.kron(M,I)
                elif(array[i] == 1):
                    M = np.kron(M,Y)
                    
            return(M)

        if string == 'sigmaZ':              # Sigma Z
            Z = np.array(([1,0],[0,-1]))
            if array[0] == 0 :
                M = I
            elif array[0] == 1 :
                M=Z

            for i in range(1,len(array)):    
                if(array[i] == 0):
                    M = np.kron(M,I)
                elif(array[i] == 1):
                    M = np.kron(M,Z)
                    
            return(M)


    # H1 = Summation 0.5*omega*(sigmaZ (i)).
    def H1(self):
        array = np.zeros((self.N,))
        array[0] = 1
        _H1 = np.zeros((self.num_basis,self.num_basis),dtype='complex')

        for i in range(0,self.N):
            _H1 += self.Generate_Matrix(array,'sigmaZ')
            array = np.roll(array,1)

        _H1 = 0.5 * self.omega * _H1

        return(_H1)
            
    # H2 = Summation Jx*sigmaX(i)*sigmaX(i+1) + Summation Jy*sigmaY(i)*sigmaY(i+1).
    def H2(self):
        array = np.zeros((self.N,))
        array[0] = 1
        array[1] = 1
        
        _Hx = np.zeros((self.num_basis,self.num_basis),dtype='complex')
        _Hy = np.zeros((self.num_basis,self.num_basis),dtype='complex')
        
        for i in range(0,self.N-1):
            print array
            _Hx +=self.Generate_Matrix(array,'sigmaX') 
            _Hy +=self.Generate_Matrix(array,'sigmaY')
            array = np.roll(array,1)

        _H2 = (self.Jx * _Hx) + (self.Jy * _Hy)
        
        return(_H2)

    # Implementing muX and muY of H(t) = muX Ex(t) + muY Ey(t) 
    def muX(self):
        array = np.zeros((self.N,))
        array[0] = 1
        _muX = np.zeros((self.num_basis,self.num_basis),dtype='complex')

        for i in range(0,self.N):
            _muX += self.Generate_Matrix(array,'sigmaX')
            array = np.roll(array,1)

        return(_muX)

    def muY(self):
        array = np.zeros((self.N,))
        array[0] = 1
        _muY = np.zeros((self.num_basis,self.num_basis),dtype='complex')

        for i in range(0,self.N):
            _muY += self.Generate_Matrix(array,'sigmaY')
            array = np.roll(array,1)

        return(_muY)



    '''
    Columnizing the Density Matrix

        d/dt (rho) = [rho,H] + L(rho)

    here rho = [r11,r12,r13........r1N
                r21,r22,r23........r2N
                .
                .
                .
                .
                .
                rN1,rN2,rN3.........rNN]

    is converted to rho' = [r11,r12,r13.......,r21,r22,r23,......,rN1,rN2,rN3......rNN].
    other operators like [rho,H] and L(rho) are modified accordingly.
    '''

    # Converts [rho,M] to the column matrix form ----> L rho
    def commutator_2_Matrix(self,M):

        commutator = self.rho_action_Op(M) - self.Op_action_rho(M)
        
        return(commutator)
    

    def rho_action_Op(self,M): # Converts rho M in column matrix form
        L1 = np.zeros((self.num_basis**2,self.num_basis**2),dtype='complex')
        array = np.arange(0,self.num_basis)

        for r in range(0,self.num_basis**2):
            for c in range(0,self.num_basis):
                L1[r][(r/self.num_basis)*self.num_basis+c] = M[c][array[0]]
            array = np.roll(array,-1)

        return(L1)
    

    def Op_action_rho(self,M):   # Converts M rho in column matrix form
        L2 = np.zeros((self.num_basis**2,self.num_basis**2),dtype='complex') 
        array = np.arange(0,self.num_basis)

        for r in range(0,self.num_basis**2):
            for c in range(0,self.num_basis):
               L2[r][array[0]+c*self.num_basis] = M[r/self.num_basis][c]
            array = np.roll(array,-1)

        return(L2)
    

    def Op1_rho_Op2(self,M1,M2): # Converts M1 rho M2 in column matrix form
        L = np.zeros((self.num_basis**2,self.num_basis**2),dtype='complex')
        array1 = np.arange(0,self.num_basis)
        array2 = np.arange(0,self.num_basis)

        for r in range(0,self.num_basis**2):
            for c in range(0,self.num_basis**2):
                L[r][c] = M1[r/self.num_basis][c/self.num_basis] * M2[array1[0]][array2[0]]
                array1 = np.roll(array1,-1)
            array2 = np.roll(array2,-1)

        return(L)
    
        
