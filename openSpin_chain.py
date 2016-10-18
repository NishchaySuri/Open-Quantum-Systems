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
import numpy.random as rnd

# Set up a Class to compute Hamiltonian for a N atom spin chain with 2^N basis vectors.

class SetUpEquation:
    def __init__(self,N,k,n_bar,omega,Jx,Jy,t_i,t_f,num_t):

    # Defining the Hamiltonian and the Problem.
        self.N = N
        self.k = k
        self.n_bar = n_bar
        self.omega = omega
        self.Jx = Jx
        self.Jy = Jy
        self.num_basis = pow(2,N)

    # Defining Time.
        self.t_i = t_i
        self.t_f = t_f
        self.num_t = num_t
        
        self.t = np.linspace(self.t_i, self.t_f, self.num_t)
        self.dt = self.t[1] - self.t[0]
        
        #self.delta = delta
        #self.eta = eta
        #self.alpha = alpha

     # Current Time Index.
        self.t_index = 0

     # Define Standard Basis Matrix
        self.std_basis = self.Standard()
        
        self.Transform_Matrix = np.zeros((self.num_basis**2,self.num_basis**2),dtype='complex')


     # Columnized Density Matrices.
        self.Psi = np.zeros((self.num_t, self.num_basis**2), dtype='complex')
        self.Psi_Init()
        self.Chi = np.zeros((self.num_t, self.num_basis**2), dtype='complex')

     # Defining Ex(t) and Ez(t).
        self.Ex = np.zeros((self.num_t,))
        self.Ez = np.zeros((self.num_t,))

     # Time Independent Part    
        self._H0 = self.H0()


        '''
    INITIALIZATION

    '''
    # Initializes the density Matrix 
    def rho_Init(self):
        Z = rnd.rand(self.num_basis,self.num_basis) + 1.j * rnd.rand(self.num_basis,self.num_basis)
        Z_dagger = np.transpose(np.conjugate(Z))

        trace = np.trace(np.dot(Z,Z_dagger))

        rho = np.dot(Z,Z_dagger) / trace

        return(rho)
        
    
    # Initializes the density column matrix
    def Psi_Init(self):
        self.Psi[0][:] = self.Matrix_2_Column(self.rho_Init())
        return(0)


     # Standard Basis Matrix
    def Standard(self):
        matrix = np.zeros((self.num_basis**2,self.num_basis**2),dtype='complex')
        array = np.zeros((self.num_basis**2,))
        array[0] = 1

        for c in range(0,self.num_basis**2):
            matrix[:,c] = array
            array = np.roll(array,1)
            
        return(matrix)


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

    # Input String as + or - to produce Sigma+ = sigmaX + i sigmaY and Sigma = sigmaX - i sigmaY
    def sigma_PlusMinus(self,string):
        array = np.zeros((self.N,))
        array[0] = 1
        
        sigmaX = np.zeros((self.num_basis,self.num_basis),dtype='complex')
        sigmaY = np.zeros((self.num_basis,self.num_basis),dtype='complex')

        for i in range(0,self.N):
            sigmaX += self.Generate_Matrix(array,'sigmaX')
            sigmaY += self.Generate_Matrix(array,'sigmaY')
            array = np.roll(array,1)

        if string == '+':
            sigma = 0.5*(sigmaX + 1.j * sigmaY)
        elif string == '-':
            sigma = 0.5*(sigmaX - 1.j * sigmaY)

        return(sigma)


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
            _Hx +=self.Generate_Matrix(array,'sigmaX') 
            _Hy +=self.Generate_Matrix(array,'sigmaY')
            array = np.roll(array,1)

        _H2 = (self.Jx * _Hx) + (self.Jy * _Hy)
        
        return(_H2)


    # H = H_timeInd + H_timeDep (H is split into a time independent and a time dependant part)
    
    def H_timeInd(self):
        return(self.H1() + self.H2())
    

    # Implementing muX and muY of H(t) = muX Ex(t) + muY Ey(t) 
    def muX(self):
        array = np.zeros((self.N,))
        array[0] = 1
        _muX = np.zeros((self.num_basis,self.num_basis),dtype='complex')

        for i in range(0,self.N):
            _muX += self.Generate_Matrix(array,'sigmaX')
            array = np.roll(array,1)

        return(_muX)

    def muZ(self):
        array = np.zeros((self.N,))
        array[0] = 1
        _muZ = np.zeros((self.num_basis,self.num_basis),dtype='complex')

        for i in range(0,self.N):
            _muZ += self.Generate_Matrix(array,'sigmaZ')
            array = np.roll(array,1)

        return(_muZ)



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

   # Takes a column Density matrix and returns square Density Matrix
    def Column_2_Matrix(self,column):
        matrix = np.zeros((self.num_basis,self.num_basis),dtype='complex')
        r=0
        col=0
        for c in range(0,self.num_basis**2):

            if(c!=0 and c%self.num_basis == 0):
                r += 1
                col = 0
                
            matrix[r][col] = column[c]
            col += 1
            
        return(matrix)
    
    # Takes a square Density matrix and returns Column Density Matrix
    def Matrix_2_Column(self,Matrix):
        column = np.zeros((self.num_basis**2,),dtype='complex')
        count = 0
        for r in range(0,self.num_basis):
            for c in range(0,self.num_basis):
                column[count] = Matrix[r][c]
                count += 1
        return(column)


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

    def Lindblad(self,M1): # Lindblad operator (rho) = M1 rho M1* - 0.5(M1* M1 rho + rho M1* M1)

        M2 = np.conjugate(np.transpose(M1))
        _Lindblad = self.Op1_rho_Op2(M1,M2) - 0.5 * (self.Op_action_rho(np.dot(M2,M1)) + self.rho_action_Op(np.dot(M2,M1)))

        return(_Lindblad)

    def Lindblad_Thermal(self): # Lindblad_Thermal = L(sqrt(2 k (1+n) sigma-)) + L(sqrt(2 k n sigma+))
        sigmaMinus = self.sigma_PlusMinus('-')
        sigmaPlus = self.sigma_PlusMinus('+')
            
        _L_thermal = self.Lindblad(np.sqrt(2*self.k*(1+self.n_bar)) * sigmaMinus) +  self.Lindblad(np.sqrt(2*self.k*self.n_bar) * sigmaPlus)

        return(_L_thermal)


    '''
    For Implementing Krotov Algorithm, we separate our rho equation in time dependant and time- independent parts.
    H0 = Time independent Part which has [rho,H] + Lindblad  (H does not include epsilons)
    '''

    def H0(self):
        return(1.j*self.commutator_2_Matrix(self.H_timeInd()) + self.Lindblad_Thermal())

    def Ht(self):
        t = self.t_index
        
        Matrix = self.muX() * self.Ex[t] + self.muZ() * self.Ez[t]

        return(1.j*self.commutator_2_Matrix(Matrix))
    
    def H(self):
        return(self._H0 + self.Ht())
        
    
    def Update_Ex_Ez(self):
        return(0)




    def Update_Transform_Matrix(self,eigVec):

        for r in range(self.num_basis**2):
            for c in range(self.num_basis**2):
                self.Transform_Matrix[r][c] = np.vdot(self.std_basis[:,r],eigVec[:,c])

        return(0)

        

    def Update_State(self):
        t = self.t_index
        
        # Diagonalize the Matrix.
        self.eigVal,self.eigVec = np.linalg.eig(self.H())

        # Update the Transformation Matrix.
        self.Update_Transform_Matrix(self.eigVec)

        # Calculate the coefficients of Psi in EigenBasis.
        ket = np.linalg.inv(self.Transform_Matrix).dot(self.Psi[t])

        # Calculate |Psi(t+dt)> by taking the action of the operator
        ket_new = np.exp(self.eigVal * self.dt) * ket

        # Convert Psi[t+1] back to the standard basis.
        self.Psi[t+1] = self.Transform_Matrix.dot(ket_new)
        
        return(0)

    def Evolution(self):
        for t in range(0,self.num_t-1):
            self.t_index = t
            self.Update_State()


    
if __name__ == '__main__':
    s = SetUpEquation(2,1,.99,1,0,0,0,5,1000)
    s.Evolution()





        
