import numpy as np
import scipy.io
import scipy.signal


def lms(x, d, M, μ):
    #x = np.insert(x, 0, np.zeros(M - 1, x.dtype))  # insert leading zeros

    N = x.size - M
    w = np.zeros(M)
    #μ_max = 2 / N * np.abs(np.average(x))

    for n in range(N):
        x_n = x[n:n + M]
        y = np.dot(w, x_n)
        e = d[n] - y
        w += μ * e * x_n

    return w


μ = 0.2
M = 5

mat_FIR = scipy.io.loadmat('System_FIR25')

x = mat_FIR["X"][0]

#d_ = scipy.signal.lfilter([0.7, 0.1, -0.03, 0.18, -0.24], [1], x)
d_ = mat_FIR["D_"][0]
noise = np.random.normal(0.0, 1.0, d_.size)
d = d_ + noise

out = lms(x, d_, M, μ)
out = np.around(out, 3)
print(out)
