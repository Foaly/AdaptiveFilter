import numpy as np
import scipy.io
import scipy.signal
import matplotlib.pyplot as plt


def smoother(x, N):
    csum = np.cumsum(np.insert(x, 0, 0))
    return (csum[N:] - csum[:-N]) / float(N)

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
        # Filterausgang
        y[n] = np.dot(w, x_n)
        # Fehlerwert
        e[n] = d[n] - y[n]
        # Aufdatierung des Koeffizientenvektors
        w += μ * e[n] * x_n

    # Quadratischer Fehler
    e = np.square(e)
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

        # A priori Ausgangswert
        y[n] = np.dot(w, x_n)

        # A priori Fehler
        e[n] = d[n] - y[n]

        # Gefilterter normierter Datenvektor
        z_denom = rho + np.dot(np.dot(x_n, inv_R), x_n.transpose())
        z = np.dot(inv_R, x_n) / z_denom

        # Aufdatierung des optimalen Gewichtsvektors
        w += e[n] * z

        # Aufdatieren der Inversen der deterministischen Autokorrelationsmatrix
        inv_R = 1 / rho * (inv_R - np.dot(np.dot(z, x_n.transpose()), inv_R))

    # Quadratischer Fehler
    e = np.square(e)
    return w, e, y

def addNoise(x, variance):
    sigma = np.sqrt(variance)
    return x + np.random.normal(0.0, sigma, x.size)


μ = 0.01
rho = 0.7

mat_FIR = scipy.io.loadmat('System_FIR25')
mat_FIR_Systemwechsel= scipy.io.loadmat('Systemwechsel_FIR25')

# x = mat_FIR["X"][0]
x = mat_FIR_Systemwechsel["X"][0]

# d_ = scipy.signal.lfilter([0.7, 0.1, -0.03, 0.18, -0.24], [1], x)
# d_ = mat_FIR["D_"][0]
d_ = mat_FIR_Systemwechsel["D_"][0]


for variance in [0.001]:  # , 0.1, 1.0, 10.0]:
    for N in [5]:
        d = addNoise(d_, variance)

        w1, e1, y1 = lms(x, d, N, μ)
        w2, e2, y2 = rls(x, d, N, rho)

        f, axarr = plt.subplots(2)
        plt.subplots_adjust(hspace=0.5)
        #f.tight_layout()
        e1 = smoother(e1, 30)
        axarr[0].plot(e1, linewidth=1.2)
        axarr[0].set_title('LMS')
        axarr[0].set_xlabel("Samples")
        axarr[0].set_ylabel("Error")
        axarr[0].grid(True)
        axarr[0].set_xlim(-1)

        e2 = smoother(e2, 30)
        axarr[1].plot(e2, linewidth=1.2)
        axarr[1].set_title('RLS')
        axarr[1].set_xlabel("Samples")
        axarr[1].set_ylabel("Error")
        axarr[1].grid(True)
        axarr[1].set_xlim(-1)

        # plt.savefig("N_" + str(N) + "_sig_" + str(sigma) + ".pdf", bbox_inches='tight')
        plt.show()

