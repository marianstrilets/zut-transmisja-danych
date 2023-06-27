import numpy as np
import matplotlib.pyplot as plt

#---------------------------------------------- twierdzenie o probkowaniu ------------------------------------------------------------
# Jest podstawową zasadą pozwalającą przekształcać sygnał ciągły w czasie (często nazywany „sygnałem analogowym”)
# w sygnał dyskretny (często nazywany „sygnałem cyfrowym”). Ustanawia warunek dla częstotliwości próbkowania, 
# która pozwala dyskretnej sekwencji próbek (cyfrowych) na przechwytywanie wszystkich informacji z sygnału ciągłego (analogowego)
# o skończonej szerokości pasma – częstotliwość Nyquista. (x - czas, y - aplituda)
#--------------------------------------------------------------------------------------------------------------------------------------

# generowanie sygnału (Numer funkcje z tabele to funkcja_...)
def x(t):
    # parametry sygnału
    f = 1000   # [Hz] częstotliwosć sygnału podstawowego
    phi = 0     # [rad] faza sygnału
    ## Tabela 1 ----->>>>> Funkcja numer 1 <<<<<-----
    #return np.cos(2 * np.pi * f * t + phi) * np.cos(2.5 * t**0.2 * np.pi)

    # Tabela 1 ----->>>>> Funkcja numer 2 <<<<<-----
    return np.abs(np.sin(2 * np.pi * f * t**2)**13) + np.cos(2 * np.pi * t)
    
    ## Tabela 1 ----->>>>> Funkcja numer 3 <<<<<-----
    #return 0.2 * np.log10(t**4 + 8) * np.sin(2 * np.pi * f * t**2 + phi) + np.cos(t / 8)
# -------------------------------------------------------------------------------------------------------------------------------------

#------------------------------------- Zadanie 1 --------------------------------------------------------------------------------------
# Proszę wybrać z tabeli 1 funkcję x(t), wygenerować próbki do bufora oraz wykreślić uzyskanego sygnalu.
# Należy dokonać samodzielnego wyboru parametrów f, phi oraz fn>=8kHz, Tc>=1s.
#--------------------------------------------------------------------------------------------------------------------------------------

def main():
    # parametry próbkowania
    Tc = 1                  # [s] czas trwania sygnału, (czas próbkowania w sekundach)
    fs = 8000               # [Hz] częstotliwosć probkowania
    N = int(Tc * fs)        # liczba próbek przypadających na cały sygnal
    Ts = 1 / fs             # okres próbkowania (powina być co najmniej dwa razy więksa niż górna granica częstotliwości sygnalu próbkowanego fmax)
    t = np.arange(N) * Ts   # czas próbkowania

    # wykres sygnału
    plt.plot(t, x(t))
    plt.xlabel('Czas [s]')
    plt.ylabel('Amplituda')
    plt.title('Wykres sygnału x(t) o częstotliwości 1 kHz')
    plt.savefig('./zadanie1/x.png')   # zapisuje wykres do pliku 'x.png'
    plt.show()


if __name__ == "__main__":
    main()