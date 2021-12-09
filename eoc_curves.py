#%%
import numpy as np
import math

# Defining activation function and their derivatives
def relu(x):
    return x*(x>0)

def relu_dash(x):
    return (x > 0)

def sigmoid(x):
    """Implement signmoid activation function."""
    return 1 / (1 + np.exp(-x))
def sigmoid_dash(x):
    return sigmoid(x)*(1-sigmoid(x))

def swish(x):
    return x * sigmoid(x)

def swish_dash(x):
    return (sigmoid(x) + x * sigmoid_dash(x))


def elu(x):
    return x *(x>0) + (np.exp(x)-1) * (x<=0)

def elu_dash(x):
    return (x>0) + np.exp(x) * (x<=0)

def mish(x):
    return x*np.tanh(np.log(1+np.exp(x)))

def mish_dash(x):
    return (np.exp(x)*(4*(x+1.)+4*np.exp(2*x)+np.exp(3*x)+np.exp(x)*(4*x+6.)))/(2*np.exp(x)+np.exp(2*x)+2.)**2


def arctanish(x):
    return x*((np.arctan(x)/math.pi)+0.5)

def arctanish_dash(x):
    return (1/math.pi)*(x/(x**2+1))+((np.arctan(x)/math.pi)+0.5)

def softsignish(x):
    return x*((x/2*(1+np.abs(x)))+0.5)


def softsignish_dash(x):
    return ((2*x**2+4*x+1)/2*(1-x)**2)*(x>0) + ((1)/2*(1-x)**2) * (x<=0)

def loglogish(x):
    return x*(1-np.exp(-np.exp(x)))

def loglogish_dash(x):
    return np.exp(-np.exp(x)*(x*np.exp(x)-1))+1

def arctanexp(x):
    return x*(2/math.pi)*np.arctan(np.exp(x))

def arctanexp_dash(x):
    return (2/math.pi)*(np.arctan(np.exp(x))+(x*np.exp(x))/(np.exp(2*x)+1))
######################################### Getting the EOC curve ########################################

# We define a function get_eoc that returns triplets (\sigma_b, \sigma_w, q) on the EOC
def get_eoc(act, act_dash, sigma_bs):
    eoc = []
    for sigma in sigma_bs:
        q = 0
        for i in range(200):
            q = sigma**2 + np.mean(act(np.sqrt(q)*z1)**2)/np.mean(act_dash(np.sqrt(q)*z1)**2)
        eoc.append([sigma, 1/np.sqrt(np.mean(act_dash(np.sqrt(q)*z1)**2)), q])
    return np.array(eoc)

# simulate gaussian variables for mean calculations
N = 500000
z1 = np.random.randn(N)
z2 = np.random.randn(N)


activation = mish
activation_dash = mish_dash
sigma_b = np.arange(0,5.1,0.1)
eoc = get_eoc(activation, activation_dash, sigma_b)
# print(eoc)




######################################## Getting the best EOC point ########################################

def beta_q(sigma_bs, act, act_dash, act_sec):
    q_s = get_eoc(act, act_dash, sigma_bs)
    return [(sigma_b, q * np.mean(act_sec(np.sqrt(q)*z1)**2)/ np.mean(act_dash(np.sqrt(q)*z1)**2) / 2) for (sigma_n, q) in zip(sigma_bs, q_s)]

#%%


#%%
import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(8,8))
x = eoc[:,0]
y = eoc[:,1]
x=np.arange(0,5,0.1)
y=np.full_like(x,1.4)
# # plt.ylim(1,2)

plt.fill_between(x, y, 3, color='midnightblue')
plt.fill_between(x,0, y, color='royalblue')
plt.text(1, 1.7, 'Chaotic phase',color='w',size=20)
plt.text(1, 0.7, 'Ordered phase',color='w',size=20)
plt.xlim((0,3))
plt.ylim((0,3))
plt.xticks(np.arange(0,3.1,0.1),size=18)
plt.yticks(np.arange(0,3.1,0.5),size=18)
plt.xlabel(xlabel=r'$\sigma_b$',fontsize=25)
#设置x轴标签及其字号
plt.ylabel(ylabel=r'$\sigma_{\omega}$',fontsize=25)




plt.plot(x,y, linestyle = '--',color='r',linewidth=2)
plt.show()


.1
# %%
