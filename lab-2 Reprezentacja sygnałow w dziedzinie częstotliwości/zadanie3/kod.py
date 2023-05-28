import numpy as np
import matplotlib.pyplot as plt
#from scipy.fft import fft
from numpy.fft import fft
import time


def dft(x):
    N = len(x)
    real = np.zeros(N)
    image = np.zeros(N)
    for k in range(N):
        for n in range(N):
            cos = np.cos((2 * np.pi * k * n) / N)
            sin = np.sin((2 * np.pi * k * n) / N)
            real[k] = real[k] + x[n] * cos
            image[k] = image[k] - x[n] * sin
    return real, image
# --------------------------------------------------------------------------------------------------------------


def x(t):
    # parametry sygnału
    f = 1000   # [Hz] częstotliwosć sygnału podstawowego
    phi = 0     # [rad] faza sygnału
    # Tabela 1 ----->>>>> Funkcja numer 2 <<<<<-----
    return np.abs(np.sin(2 * np.pi * f * t**2)**13) + np.cos(2 * np.pi * t)
# --------------------------------------------------------------------------------------------------------------


def y(t):
    # Tabela 2 ----->>>>> Funkcja numer 2 <<<<<-----
    return ((x(t) * t**3) / 3)
# --------------------------------------------------------------------------------------------------------------


def z(t):
    # Tabela 2 ----->>>>> Funkcja numer 2 <<<<<-----
    return (1.92 * (np.cos(3 * np.pi * (t/2)) + np.cos((y(t))**2 / (8*x(t) + 3) * t)))
# --------------------------------------------------------------------------------------------------------------


def v(t):
    # Tabela 2 ----->>>>> Funkcja numer 2 <<<<<-----
    return (((y(t) * z(t)) / (x(t) + 2)) * np.cos(7.2 * np.pi * t) + np.sin(np.pi * t**2))
# --------------------------------------------------------------------------------------------------------------
#   Tabela 3 ----->>>>> Funkcja numer 1 <<<<<-----


def u1(t):
    #   0.1 > t >= 0
    return (np.sin(6 * np.pi * t) * np.cos(5 * np.pi * t))


def u2(t):
    #   0.4 > t >= 0.1:
    return (-1.1 * t * np.cos(41 * np.pi * t**2))


def u3(t):
    #   0.72 > t >= 0.4:
    return (t * np.sin(20 * t**4))


def u4(t):
    #   1 > t >= 0.72:
    return (3.3 * (t - 0.72) * np.cos(27 * t + 1.3))
# --------------------------------------------------------------------------------------------------------------


def b1(t, h=2):
    values = np.sin(h * t * np.pi) / (2 + np.cos(h**2 * np.pi * t))
    return values
# --------------------------------------------------------------------------------------------------------------


def b2(t, h=6):
    values = np.sin(h * t * np.pi) / (2 + np.cos(h**2 * np.pi * t))
    return values
# --------------------------------------------------------------------------------------------------------------


def b3(t, h=10):
    values = np.sin(h * t * np.pi) / (2 + np.cos(h**2 * np.pi * t))
    return values
# --------------------------------------------------------------------------------------------------------------
def gen(fun, tc=0.2, fs=40000):
    # tc  - [s] czas trwania sygnału w sekundach
    # fs  - [Hz] częstotliwość próbkowania, liczba próbek sygnału w ciągu sekundy

    # Inicjalizacja zmiennych i list
    X = []          # lista dla przechowywania próbek sygnału
    N = int(tc * fs)  # liczba próbek przypadających na cały sygnal
    fmax = fs/2
    for n in range(N):
        t = n / fs
        if fun == 'x':
            X.append(x(t))
        elif fun == 'y':
            X.append(y(t))
        elif fun == 'z':
            X.append(z(t))
        elif fun == 'v':
            X.append(v(t))
        elif fun == 'u':
            if 0.1 > t >= 0:
                X.append(u1(t))
            elif 0.4 > t >= 0.1:
                X.append(u2(t))
            elif 0.72 > t >= 0.4:
                X.append(u3(t))
            elif 1 > t >= 0.72:
                X.append(u4(t))
        elif fun == 'b1':
            X.append(v(t))
        elif fun == 'b2':
            X.append(v(t))
        elif fun == 'b3':
            X.append(v(t))

    t1 = time.time()
    real, image = dft(X)
    t1 = round(time.time() - t1, 10)
    print(fun + ': ' + 'Czas dla DFT: ', t1)

    t2 = time.time()
    _ = fft(X)
    t2 = round(time.time() - t2, 10)

    print(fun + ': ' + 'Czas dla FFT: ' + str(t2))
    if t2 > 0:
        acceleration = round(t1/t2, 10)
    else:
        acceleration = t1 * 1000000000
        t2 = 0.0000000001

    #przyspieszenie = round(t1/t2, 8)
    print(fun + ': ' + 'Przyśpieszenie: ' + str(acceleration) + '\n')

    fk = []
    M2 = []
    for k in range(N//2):
        M = np.sqrt(real[k]**2 + image[k]**2)
        M2.append(10 * np.log10(M))
        fk.append(k * fs / N)

    plt.stem(fk, M2)
    plt.xlabel('Częstotliwość [Hz]')
    plt.ylabel('Amplituda [dB]')
    plt.xlim(0, fmax)
    plt.title('Wykres widma ampitudowego dla ' + fun + '(k)')
    plt.savefig('./' + fun + '.png')
    plt.show()
    return t1, t2
# --------------------------------------------------------------------------------------------------------------
t_dft = 0
t_fft = 0
l_fun = ['x', 'y', 'z', 'v', 'u', 'b1', 'b2', 'b3']

for i in l_fun:
    (t1, t2) = gen(i)
    t_dft += t1
    t_fft += t2
# --------------------------------------------------------------------------------------------------------------

print('=================================================')
print('\tCzas sumaryczny dla DFT: ', round(t_dft, 10))
print('\tCzas sumaryczny dla FFT: ', round(t_fft, 10))
if t2 > 0:
    acceleration = round(t_dft/t_fft, 10)
else:
    acceleration = 0
print('\tPrzyśpieszenie: ', round(acceleration, 10))
print('=================================================')
# --------------------------------------------------------------------------------------------------------------
