import scipy.integrate as integrate
import matplotlib.pyplot as plt
import numpy as np
import json
import opt

Lightspeed = 3e11

def f(z, H0, Omega):
    d = np.array([Lightspeed/H0 * (1+z0) * integrate.quad(
        lambda x: 1/((1-Omega) * (1+x)**3 + Omega)**0.5, 0, z0)[0]
        for z0 in z])
    mu = 5 * np.log10(d) - 5
    return mu

def j(z, H0, Omega):
    d = np.array([Lightspeed / H0 * (1 + z0) * integrate.quad(lambda x: 1.0 / ((1 - Omega) * (1 + x)**3 + Omega)**0.5, 0, z0)[0] for z0 in z])
    J = np.empty((z.size, 2), dtype=float)
    J[:, 0] = -5.0 / H0 / np.log(10)
    J[:, 1] = 5.0 / d / np.log(10) * [Lightspeed / H0 * (1 + z0) * integrate.quad(lambda x: ((x + 1)**3 - 1) / (2 * (Omega - (Omega - 1) * (x + 1)**3)**(3 / 2)), 0, z0)[0] for z0 in z] 
    return J


with open('jla_mub.txt') as file:
    d = np.loadtxt(file)
    
    z = d[:,0]
    y = d[:,1]
    
gauss_out = opt.gauss_newton(y, lambda *args: f(z, *args), 
                          lambda *args: j(z, *args), (50.0, 0.5))
lm_out = opt.lm(y, lambda *args: f(z, *args), 
                lambda *args: j(z, *args), (50.0, 0.5))

plt.figure(figsize=(20, 10))
plt.scatter(z, y, label='file_input')
plt.plot(z, f(z, gauss_out[3][0], gauss_out[3][1]), label='Gauss-Newton', color='black')
plt.plot(z, f(z, lm_out[3][0], lm_out[3][1]), label='Levenberg-Marquardt', color='red')
plt.xlabel("z")
plt.ylabel("mu")
plt.grid()
plt.legend()
plt.savefig('mu-z.png')

plt.figure(figsize=(20, 10))
plt.plot(gauss_out.cost, label='Gauss-Newton')
plt.plot(lm_out.cost, label='Levenberg-Marquardt')
plt.xlabel("z")
plt.ylabel("cost")
plt.legend()
plt.savefig('cost.png')

out = {
  "Gauss-Newton":{
    "H0": float(gauss_out[3][0]),
    "Omega": float(gauss_out[3][1]),
    "nfev": int(gauss_out[0])
  },
  "Levenberg-Marquardt": {
    "H0": float(lm_out[3][0]),
    "Omega": float(lm_out[3][1]),
    "nfev": int(lm_out[0])
  }
}

with open('parameters.json', 'w') as f:
    json.dump(out, f)
