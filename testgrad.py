import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from gradient import gradient
from approximations import euler_method, runge_kutter2_method, runge_kutter4_method

def rmse(y, yhat):
    return np.sqrt(np.mean(np.power(y-yhat, 2)))

def x2(x):
    return (x[0]**2+x[1]**2)

def expeq(x0, t, param=0.3):
    return x0*np.exp(t*param)

def d_expeq(y, param=0.3):
    return param*y

gr = gradient(x2, np.array([0,400000]))
print(f'gradient: {gr}')

time_t = np.arange(0, 30, 1.0)
e_t = expeq(3, time_t)

et_euler = euler_method(d_expeq, 3, time_t)
rmse_euler = rmse(e_t, et_euler)

et_rk2 = runge_kutter2_method(d_expeq, 3, time_t)
rmse_rk2 = rmse(e_t, et_rk2)

et_rk4 = runge_kutter4_method(d_expeq, 3, time_t)
rmse_rk4 = rmse(e_t, et_rk4)

print(f'euler shape: {et_euler.shape}')


fig, ax = plt.subplots(figsize=(4,4))
line1 = ax.plot(time_t[0], e_t[0], label='Exact')[0]
line2 = ax.plot(time_t[0], et_euler[0], 
                label=f'Euler Approx, RMSE: {rmse_euler:.2f}')[0]
line3 = ax.plot(time_t[0], et_rk2[0],
                 label=f'RK2 Approx, RMSE: {rmse_rk2:.2f}')[0]
line4 = ax.plot(time_t[0], et_rk4[0],
                 label=f'RK4 Approx, RMSE: {rmse_rk4:.2f}')[0]
ax.set(xlim=[time_t.min(), time_t.max()],
        ylim=[0, e_t.max()], xlabel='Time [s]', ylabel='Z [m]')
ax.legend(loc='upper left')

def update(frame):
    # for each frame, update the data stored on each artist.
    # x = time_t[:frame]
    # et_exact= e_t[:frame]
    # et_eu = et_euler[:frame]
    # update the scatter plot:

    # update the line plot:
    line1.set_xdata(time_t[:frame])
    line1.set_ydata(e_t[:frame])
    line2.set_xdata(time_t[:frame])
    line2.set_ydata(et_euler[:frame])
    line3.set_xdata(time_t[:frame])
    line3.set_ydata(et_rk2[:frame])
    line4.set_xdata(time_t[:frame])
    line4.set_ydata(et_rk4[:frame])
    return (line1, line2,line3, line4)

ani = animation.FuncAnimation(fig=fig, func=update, frames=len(time_t),
                              interval=100, repeat=False)
plt.show()
