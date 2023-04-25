# =============================================================================================================
# Dyskretne przekształczenie Fouriera (DFT) jest techniką matematyczną wykorzystywaną do wyznaczenia zawartości
# częstotliwościowej sygnału dyskretnego.
#       X(k) - reprezentacja w dziedzinie  częstotliwości (wektor liczb zespolonych postaci a + ib)
#       x(n) - próbki reprezentujące sygnał w dziedzinie czasu
#       j    - jednostka urojona ( i^2 = -1)
#       N    = liczba próbek sygnałow w dziedzinie czasu i częstotliwości
# =============================================================================================================
#    Zadanie 1:
# 1. Proszę zaimplementować przekształczenie DFT na podstawie wzoru przedstawionego  w punkcie 1.
# =============================================================================================================
import numpy as np
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
#==============================================================================================================
print('\n---------------------------- Wyswietlanie ----------------------------')
# przykładowy sygnał wejściowy
k = np.arange(5)
# wywołujemy funkcję dft() dla sygnału k
real, image = dft(k)
# wyświetlamy wyniki części rzeczywistej i urojonej
print("\tCzęść rzeczywista:\n", real, "\n")
print("\tCzęść urojona:\n", image, "\n")
#==============================================================================================================
#===================================== TEST ===================================================================
print('\n------------------------------- TEST ---------------------------------')
k = np.arange(5)
xk = np.fft.fft(k)
real = np.real(xk)
image = np.imag(xk)
print("\tCzęść rzeczywista:\n", real, "\n")
print("\tCzęść urojona:\n", image, "\n")
#==============================================================================================================
