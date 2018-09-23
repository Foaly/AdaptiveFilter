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

# nach Moschytz S.85
def lms(x, d, M, μ):

    N = x.size
    w = np.zeros(M)
    e = np.zeros(N)
    y = np.zeros(N)

    for n in range(M, N):
        x_n = x[n:n-M:-1]  # Works but the first element still gets ignored (due to exclusive end index)
        y[n] = np.dot(w, x_n)
        e[n] = d[n] - y[n]
        w += μ * e[n] * x_n

    return w, e, y


# x   = Eingangsvektor
# d   = Rauschen + Ausgangssignal vom unbekanten system
# M   = Anzahl der Filterkoeffizienten
# rho = Vergessensfaktor

# w = Filterkoeffizienten
# e = Fehlersignal
# y = gefilteres Signal

# nach Moschytz S.145
def rls(x, d, M, rho):
    p0 = 1000000
    inv_R = p0 * np.identity(M)

    N = x.size
    w = np.zeros(M)
    e = np.zeros(N)
    y = np.zeros(N)

    for n in range(M, N):
        x_n = x[n:n-M:-1]  # Works but the first element still gets ignored (due to exclusive end index)

        a = np.dot(w, x_n)
        y[n] = a
        e[n] = d[n] - y[n]

        z_denom = rho + np.dot(np.dot(x_n, inv_R), x_n.transpose())
        z = np.dot(inv_R, x_n) / z_denom
        inv_R = 1 / rho * (inv_R - np.dot(np.dot(z, x_n.transpose()), inv_R))

        w += np.dot(inv_R, x_n) * e[n]

    return w, e, y



μ = 0.2
M = 5

mat_FIR = scipy.io.loadmat('System_FIR25')

x = mat_FIR["X"][0]

#d_ = scipy.signal.lfilter([0.7, 0.1, -0.03, 0.18, -0.24], [1], x)
d_ = mat_FIR["D_"][0]
noise = np.random.normal(0.0, 1.0, d_.size)
d = d_ + noise

out, e, y = rls(x, d_, M, μ)
out = np.around(out, 3)
plt.plot(np.abs(e[0:200]))
plt.show()
print(out)
