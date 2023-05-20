import numpy as np
import matplotlib.pyplot as plt

from gradient import gradient
from approximations import euler_method

def x2(x):
    return (x[0]**2+x[1]**2)

def expeq(x0, t, param=0.3):
    return x0*np.exp(t*param)

def d_expeq(y, param=0.3):
    return param*y

gr = gradient(x2, np.array([0,400000]))

print(f'gradient: {gr}')

time_t = np.arange(0, 10, 0.5)
e_t = expeq(3, time_t)
et_euler = euler_method(d_expeq, 3, time_t)


fig, ax = plt.subplots(figsize=(4,4))
ax.plot(time_t, e_t, label='Exact')
ax.plot(time_t, et_euler, label='Euler Approx')
ax.legend()
plt.show()
