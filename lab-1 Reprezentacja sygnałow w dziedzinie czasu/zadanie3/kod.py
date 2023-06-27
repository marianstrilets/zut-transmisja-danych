import numpy as np
import matplotlib.pyplot as plt


def u(t):
    # Tabela 3 ----->>>>> Funkcja numer 1 <<<<<-----
    if 0.1 > t >= 0:
        return ( np.sin(6 * np.pi * t) * np.cos(5 * np.pi * t ) )
    elif 0.4 > t >= 0.1:
        return (-1.1 * t * np.cos(41 * np.pi * t**2) )
    elif 0.72 > t >= 0.4:
        return (t * np.sin(20 * t**4))
    elif 1 > t >= 0.72:
        return (3.3 * (t - 0.72) * np.cos(27 * t + 1.3))
    else:
        return 0
#------------------------------------- Zadanie 3 --------------------------------------------------------------------------------------
# Wykreslić wykres dladowolnej funkcji u(t) wybranej z tabeli 3. Czas trwania sygnalu wynika z definicji funkcji,
# natomiast częstotliwość próbkowania fs jest taka sama jak w poprzednim ćwiczeniu.
#--------------------------------------------------------------------------------------------------------------------------------------

def main():
    # parametry próbkowania
    Tc = 1                  # [s] czas trwania sygnału, (czas próbkowania w sekundach)
    fs = 8000               # [Hz] częstotliwosć probkowania
    N = int(Tc * fs)        # liczba próbek przypadających na cały sygnal
    Ts = 1 / fs             # okres próbkowania (powina być co najmniej dwa razy więksa niż górna granica częstotliwości sygnalu próbkowanego fmax)
    t = np.arange(N) * Ts   # czas próbkowania
    
    # obliczenie wartości funkcji u(t) dla kolejnych próbek czasowych
    u_values = [u(ti) for ti in t]
    
    plt.plot(t, u_values)
    plt.xlabel('Czas [s]')
    plt.ylabel('Aplituda')
    plt.title('Wykres sygnalu u(t)')
    plt.savefig('./zadanie3/u.png')
    plt.show()

if __name__ == "__main__":
    main()