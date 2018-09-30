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
    W = np.zeros((M, N))
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

        W[:, n] = w

    # Quadratischer Fehler
    e = np.square(e)
    return W, e, y


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
    W = np.zeros((M, N))
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

        W[:, n] = w

    # Quadratischer Fehler
    e = np.square(e)
    return W, e, y




# KLMS


# nach Haykin, Liu, Principe, S.34
def klmsPredict(kernel, x, M):
    N = x.shape[0]
    kernel.prediction = [0]
    kernel.errors = [0]

    for i in range(M, N):
        # get data slice
        x_n = x[i - M:i]
        # reverse
        x_n = x_n[::-1]

        y = kernel.predict(x_n)
        kernel.prediction.append(y)

        # calculate error
        kernel.error = x[i] - y
        kernel.errors.append(np.square(kernel.error))


def klmsTrain(kernel, x, M):
    N = x.shape[0]

    for i in range(M, N):
        # get data slice
        x_n = x[i - M:i]
        # reverse
        x_n = x_n[::-1]

        # update with slice
        kernel.update(x_n, x[i])



class klmsHelper():
    """
    KLMS class
    Nach Haykin, Liu, Principe, p.34 / Algorithm 2
    """

    def __init__(
        self,
        mu=0.5,
        sigma=1.0
    ):
        self.data = [0]
        self.weights = [0]
        self.mu = mu
        self.sigma = sigma
        self.error = None
        self.prediction = [0]
        self.errors = [0]

        # kernel function
        self.gauss = lambda a, b: np.exp(-1 * np.square(np.linalg.norm(a - b)) / (2 * np.square(self.sigma)))
        self.kFun = self.gauss


    def predict(self, x_n):
        # initialize estimate
        predict = 0
        for i in range(len(self.weights)):
            # predict for every datapoint i with according weight (past)
            predict_i = self.weights[i] * self.kFun(self.data[i], x_n)
            # sum all datapoints running i to estimate prediction
            predict += predict_i
        return predict


    def update(self, x_n, desired):
        # calculate error for current prediction towards desired output
        self.error = desired - self.predict(x_n)
        # calculate weight based on error & learning rate
        new_weight = self.mu * self.error
        # add weights to stack (past)
        self.weights.append(new_weight)
        # add datapoint to stack (past)
        self.data.append(x_n)
        # add prediction to stack
        self.prediction.append(self.predict(x_n))

        self.errors.append(self.error ** 2)





# x        = Eingangsvektor
# variance = Varianz des Rauschen

def addNoise(x, variance):
    sigma = np.sqrt(variance)
    return x + np.random.normal(0.0, sigma, x.size)



# Script

μ = 0.03
rho = 0.9
np.random.seed(42)

mat_FIR = scipy.io.loadmat('System_FIR25')
mat_IIR = scipy.io.loadmat('System_IIR25')
mat_FIR_Systemwechsel= scipy.io.loadmat('Systemwechsel_FIR25')
mat_IIR_Systemwechsel= scipy.io.loadmat('Systemwechsel_IIR25')

# x = mat_FIR["X"][0]
# x = mat_IIR["X"][0]
# x = mat_FIR_Systemwechsel["X"][0]
x = mat_IIR_Systemwechsel["X"][0]

# d_ = scipy.signal.lfilter([0.7, 0.1, -0.03, 0.18, -0.24], [1], x)
# d_ = mat_FIR["D_"][0]
# d_ = mat_IIR["D_"][0]
# d_ = mat_FIR_Systemwechsel["D_"][0]
d_ = mat_IIR_Systemwechsel["D_"][0]


mat_training = scipy.io.loadmat('Training')
train_data = mat_training['x_training'].flatten()

mat_test = scipy.io.loadmat('Test')
test_data = mat_test['x_test'].flatten()


# for variance in [0.001]: #, 0.1, 1.0, 10.0]:
#     for N in [5]:
#         d = addNoise(d_, variance)
#
#         end = 10000
#
#         w1, e1, y1 = lms(x, d, N, μ)
#         w2, e2, y2 = rls(x, d, N, rho)
#
#         avg_e1 = np.average(e1)
#         avg_e2 = np.average(e2)
#
#         f, axarr = plt.subplots(2, 2, figsize=(12, 5))
#         f.tight_layout()
#         plt.subplots_adjust(hspace=0.5, wspace=0.2)
#
#         e1 = smoother(e1, 30)
#         axarr[0][0].plot(e1[:end], 'b', linewidth=1)
#         axarr[0][0].plot([0, end], [avg_e1, avg_e1], 'r--', linewidth=1.2)
#         axarr[0][0].set_title('LMS (Squared Error)')
#         axarr[0][0].set_xlabel("Samples")
#         axarr[0][0].set_ylabel("Squared Error")
#         axarr[0][0].grid(True)
#         axarr[0][0].set_xlim([-1, end])
#
#         coeffCount = w1.shape[0]
#         for coeff in range(0, coeffCount):
#             axarr[0][1].plot(w1[coeff][:end], linewidth=1)
#         legend = ['w' + str(i+1) + ' = ' + str(np.around(w1[i, -1], 3)) for i in range(0, coeffCount)]
#         axarr[0][1].legend(legend, loc='right', title="Final Weights")
#         axarr[0][1].set_title('LMS (Filterkoeffizienten)')
#         axarr[0][1].set_xlabel("Samples")
#         axarr[0][1].set_ylabel("Koeffizienten")
#         axarr[0][1].set_xlim([-10, end])
#
#         e2 = smoother(e2, 30)
#         axarr[1][0].plot(e2[:end], 'b', linewidth=1)
#         axarr[1][0].plot([0, end], [avg_e2, avg_e2], 'r--', linewidth=1.2)
#         axarr[1][0].set_title('RLS (Squared Error)')
#         axarr[1][0].set_xlabel("Samples")
#         axarr[1][0].set_ylabel("Squared Error")
#         axarr[1][0].grid(True)
#         axarr[1][0].set_xlim([-1, end])
#
#         coeffCount = w2.shape[0]
#         for coeff in range(0, coeffCount):
#             axarr[1][1].plot(w2[coeff][:end], linewidth=1)
#         legend = ['w' + str(i+1) + ' = ' + str(np.around(w2[i, -1], 3)) for i in range(0, coeffCount)]
#         axarr[1][1].legend(legend, loc='right', title="Final Weights")
#         axarr[1][1].set_title('RLS (Filterkoeffizienten)')
#         axarr[1][1].set_xlabel("Samples")
#         axarr[1][1].set_ylabel("Koeffizienten")
#         axarr[1][1].set_xlim([-10, end])
#
#         # plt.savefig("N_" + str(N) + "_var_" + str(variance) + ".pdf", bbox_inches='tight')
#         # plt.savefig("N_" + str(N) + "_var_" + str(variance) + "_mu_" + str(μ) + ".pdf", bbox_inches='tight')
#         plt.savefig("mu_" + str(μ) + "_rho_" + str(rho) + ".pdf", bbox_inches='tight')
#         plt.show()


# Plots
# d = addNoise(d_, 0.001)
# N = 5
# end = 2000
#
# for μ in [0.01, 0.1, 0.2]:
#     w1, e1, y1 = lms(x, d, N, μ)
#
#     avg_e1 = np.average(e1)
#
#     plt.figure(figsize=(10, 2.5))
#     plt.tight_layout()
#
#     e1 = smoother(e1, 30)
#     plt.plot(e1[:end], 'b', linewidth=1)
#     plt.plot([0, end], [avg_e1, avg_e1], 'r--', linewidth=1.2)
#     plt.title('LMS (MSE, μ = ' + str(μ) + ')')
#     plt.xlabel("Samples")
#     plt.ylabel("Error")
#     plt.grid(True)
#     plt.xlim([-1, end])
#     plt.savefig("lms_N_5_var_0.001_mu_" + str(μ) + ".pdf", bbox_inches='tight')
#     plt.show()


for [M, mu, sigma] in [[5, 0.5, 0.5]]:#, [10, 0.5, 0.5]]:
    prefix = 'KLMS_' + str(M) + '_' + str(mu) + '_' + str(sigma)
    trainErrorName = prefix + '_train_error'
    trainPredictionName = prefix + '_train_prediction'
    testErrorName = prefix + '_test_error'
    testPredictionName = prefix + '_test_prediction'
    kernelWeigthName = prefix + '_kernel_weights'

    # # klms
    # kernel = klmsHelper(mu, sigma)
    # klmsTrain(kernel, train_data, M)
    #
    # trainErrors = kernel.errors.copy()
    # trainPrediction = kernel.prediction.copy()
    #
    # # warning: long computation time
    # klmsPredict(kernel, test_data, M)
    #
    # testErrors = kernel.errors
    # testPrediction = kernel.prediction
    # kernelWeights = kernel.weights
    #
    # np.savetxt(trainErrorName + '.txt', trainErrors)
    # np.savetxt(trainPredictionName + '.txt', trainPrediction)
    # np.savetxt(testErrorName + '.txt', testErrors)
    # np.savetxt(testPredictionName + '.txt', testPrediction)
    # np.savetxt(kernelWeigthName + '.txt', kernel.weights)

    trainErrors = np.loadtxt(trainErrorName + '.txt')
    trainPrediction = np.loadtxt(trainPredictionName + '.txt')
    testErrors = np.loadtxt(testErrorName + '.txt')
    testPrediction = np.loadtxt(testPredictionName + '.txt')
    kernelWeights = np.loadtxt(kernelWeigthName + '.txt')

    # # plotting
    # N = len(trainPrediction);
    # M2 = M - 1
    #
    # for i in range(M2):
    #     trainErrors[i] = 0
    #     testErrors[i] = 0
    #
    # plt.figure(figsize=(12, 5))
    # plt.tight_layout()
    # plt.subplot(211)
    # plt.title('KLMS (N: ' + str(M) + ', μ: ' + str(mu) + ', σ²: ' + str(sigma) + ')')
    #
    # plt.plot(train_data[M:], 'k--')
    # plt.plot(trainPrediction, 'r')
    #
    # plt.plot(test_data[M:], 'k-.')
    # plt.plot(testPrediction, 'b')
    # plt.xlim([M2, N])
    # plt.legend(['Train data', 'Training', 'Test data', 'Testing'])
    # #plt.xlabel("Samples")
    # plt.ylabel("Amplitude")
    #
    # plt.subplot(212)
    # plt.plot(trainErrors[:], 'r')
    # plt.plot(testErrors[:N], 'b')
    # plt.xlim([M2, N])
    # plt.legend(['Train', 'Test'])
    # plt.xlabel("Samples")
    # plt.ylabel("MSE")
    #
    # plt.savefig(kernelWeigthName + ".pdf", bbox_inches='tight')
    # plt.show()





    N = 5
    w1, e1, y1 = lms(train_data, train_data, N, 0.01)

    # use final pre-trained Weights
    data_len = test_data.shape[0]
    y = np.zeros(data_len)
    lms_error = np.zeros(data_len)
    w = w1.transpose()[-1]
    for n in range(N, data_len - N):
        # Input vector
        x = test_data[n:n-N:-1]
        # Prediction value
        y[n] = np.dot(x, w)
        # Save error
        lms_error[n] = (y[n] - test_data[n]) ** 2

    avg_e1 = np.average(e1)
    end = 10000

    plt.figure(figsize=(12, 5))
    plt.tight_layout()

    e1 = smoother(e1, 30)
    plt.plot(testErrors[:end], 'r', linewidth=1)
    plt.plot(lms_error[:end], 'b', linewidth=1)
    #plt.plot([0, end], [avg_e1, avg_e1], 'r--', linewidth=1.2)
    legend = ['KLMS', 'LMS']
    plt.legend(legend, loc='upper right', title="Squared Error")
    plt.xlabel("Samples")
    plt.ylabel("Squared Error")
    plt.grid(True)
    plt.xlim([-1, end])
    plt.savefig("LMS_KLMS_Vergleich.pdf", bbox_inches='tight')
    plt.show()
