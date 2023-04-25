# =============================================================================================================
#    Zadanie 2:
# 2. Dla uzyskanej reprezentacje w dziedzinie częstotliwośći X(k), k=0, ... , N/2-1:
#       - obliczyć widmo amplitudowe
#           można obliczyć ze wzoru: M(k) = sqrt(Re(X(k))^2 + Im(X(k))^2)
#       - wartośći amplitudy przedstawić w skali decybelowej
#           można obliczyć ze wzoru: M'(k) = 10 * log10(M(k))
#       - wyznaczyć skalę częstotłiwości:
#           Skala częstotliwości to wektor wartości częstotliwości, dla których obliczone zostanie
#           widmo amplitudowe. Wartości te są wyznaczane na podstawie ilości próbek sygnału xt i
#           częstotliwości próbkowania. Skala częstotliwości może być wyznaczana ze wzoru:
#               fk = k * fs / N
#           gdzie:
#               fk - to wartość częstotliwości dla k-tego punktu widma amplitudowego,
#               k  - to wartość indeksu punktu widma amplitudowego, gdzie k = 0, 1, ..., N/2-1,
#               fs - to częstotliwość próbkowania,
#               N  - to liczba próbek sygnału xt.
#       - wyznaczyć wykres widma amplitudowego M'(k) (fk oznaczają częstotłiwości prążków widma)
#           Widmo amplitudowe M(k) można wyznaczyć dla każdego k, a następnie przedstawić wartości
#           amplitudy w skali decybelowej M'(k). W efekcie otrzymamy wykres widma amplitudowego M'(k),
#           gdzie wartości M'(k) będą reprezentowane przez wysokość słupków (prążków) na wykresie,
#           a wartości częstotliwości fk będą oznaczać pozycję tych prążków na osi X.
# =============================================================================================================
import numpy as np
import matplotlib.pyplot as plt
# -------------------------------------------------------------------------------------------------------------
def dft(x):
    # obliczamy długość sygnału x (liczba próbek sygnałow w dziedzinie czasu i częstotliwości)
    N = len(x)
    # tworzymy dwie listy o długości N, aby przechowywać wyniki DFT dla części rzeczywistej i urojonej
    real = np.zeros(N)
    image = np.zeros(N)
    for k in range(N):
        for n in range(N):
            # obliczamy wartości cosinus i sinus dla obliczeń DFT
            cos = np.cos((2 * np.pi * k * n) / N)
            sin = np.sin((2 * np.pi * k * n) / N)
            # obliczamy część rzeczywistą i urojoną wyniku DFT dla wartości k
            real[k] = real[k] + x[n] * cos
            image[k] = image[k] - x[n] * sin
    # zwracamy wyniki DFT jako dwie listy zawierające wartości części rzeczywistej i urojonej
    return real, image
# ==============================================================================================================
# -------------------------------------------------------------------------------------------------------------
def X(t, A=200, f0=100, f1=1000):
    # Funkcja zwraca wartość sygnału X w chwili czasowej t, obliczonego z dwóch składowych sinusoidalnych
    # o amplitudzie A i częstotliwościach f0 oraz f1.
    #   t  - czas w chwili, dla której ma zostać obliczona wartość sygnału X
    #   A  - amplituda sygnału sinusoidalnego, domyślna wartość to 200
    #   f0 - częstotliwość pierwszej sinusoidy, domyślna wartość to 100 Hz
    #   f1 - częstotliwość drugiej sinusoidy, domyślna wartość to 1000 Hz
    result = A * np.sin(2 * np.pi * f0 * t) + A * np.sin(2 * np.pi * f1 * t)
    return result
# ==============================================================================================================
# --------------------------------------------------------------------------------------------------------------
# Inicjalizacja zmiennych i list
xt = []         # lista dla przechowywania próbek sygnału
tc = 0.1        # [s] czas trwania sygnału w sekundach
fs = 8000       # [Hz] częstotliwość próbkowania, liczba próbek sygnału w ciągu sekundy
N = int(tc*fs)  # liczba próbek przypadających na cały sygnal

# Generowanie próbek sygnału X(t)
for n in range(N):
    t = n/fs
    xt.append(X(t))

# dyskretna transformate Fouriera (DFT) na sygnale xt, zwraca dwie listy liczb: real, image
real, image = dft(xt)

# Obliczanie widma amplitudowego
fk = []     # to lista, częstotliwości od 0 do wartości połowy liczby próbek sygnału
M = []      # to lista, wartości amplitud sygnału w dziedzinie częstotliwości
M2 = []     # to lista, wartości amplitud sygnału w dziedzinie częstotliwości, ale w skali decybelowej

for k in range(N//2):
    # Obliczanie widmo amplitudowe
    M.append(np.sqrt(real[k]**2 + image[k]**2))    
    # Obliczanie wartośći amplitudy w skali decybelowej
    M2.append(10*np.log10(M[k]))    
    # Obliczanie skalę częstotłiwości
    fk.append(k * fs / N)
# ==============================================================================================================
# --------------------------------------------------------------------------------------------------------------
# Wykres widma amplitudowego
plt.stem(fk, M2)
plt.title('Wykres widma ampitudowego')
plt.xlabel('Częstotliwość [Hz]')
plt.ylabel('Amplituda [dB]')
plt.savefig('./zadanie2/widmo.png')
plt.show()
# ==============================================================================================================