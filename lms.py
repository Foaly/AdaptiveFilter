import numpy as np
import scipy.io
import scipy.signal
import matplotlib.pyplot as plt


# x = Eingangsvektor
# d = Rauschen + Ausgangssignal vom unbekanten system
# M = Anzahl der Filterkoeffizienten
# μ = Schrittweite

# w = Filterkoeffizienten
# e = Fehlersignal
# y = gefilteres Signal

def lms(x, d, M, μ):
    #x = np.insert(x, 0, np.zeros(M - 1, x.dtype))  # insert leading zeros

    N = x.size - M
    w = np.zeros(M)
    e = np.zeros(N)
    y = np.zeros(N)

    for n in range(M-1, N):
        x_n = [ x[n], x[n-1], x[n-2], x[n-3], x[n-4] ]
        x_n = np.array(x_n)
        y[n] = np.dot(w, x_n)
        e[n] = d[n] - y[n]
        w += μ * e[n] * x_n

    return w, e, y




μ = 0.2
M = 5

mat_FIR = scipy.io.loadmat('System_FIR25')

x = mat_FIR["X"][0]

#d_ = scipy.signal.lfilter([0.7, 0.1, -0.03, 0.18, -0.24], [1], x)
d_ = mat_FIR["D_"][0]
noise = np.random.normal(0.0, 1.0, d_.size)
d = d_ + noise

out, e, y = lms(x, d_, M, μ)
out = np.around(out, 3)
plt.plot(np.abs(e[0:200]))
plt.show()
print(out)
