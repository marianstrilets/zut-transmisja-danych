import numpy as np
import matplotlib.pyplot as plt


def x(t):
    # parametry sygnału
    f = 1000   # [Hz] częstotliwosć sygnału podstawowego
    phi = 0     # [rad] faza sygnału
    # Tabela 1 ----->>>>> Funkcja numer 2 <<<<<-----
    return np.abs(np.sin(2 * np.pi * f * t**2)**13) + np.cos(2 * np.pi * t)

def y(t):
    # Tabela 2 ----->>>>> Funkcja numer 2 <<<<<-----
    return ( (x(t) * t**3) / 3)

def z(t):
    # Tabela 2 ----->>>>> Funkcja numer 2 <<<<<-----
    return ( 1.92 * ( np.cos( 3 * np.pi * (t/2)) + np.cos((y(t))**2 / (8*x(t) + 3) * t)))

def v(t):
    # Tabela 2 ----->>>>> Funkcja numer 2 <<<<<-----
    return (((y(t) * z(t)) / ( x(t) + 2)) * np.cos(7.2 * np.pi * t) + np.sin(np.pi * t**2))

#------------------------------------- Zadanie 2 --------------------------------------------------------------------------------------
# Dla dowolnego zestawu funkcji z tabeli 2 należy wygenerować trzy sygnały reprezętujące funkcje 
# y(t), z(t), oraz v(t), gdzie x(t) jest funkcją wybraną w ćwieczeniu 1. Wykonac wykresy dla każdego z wygenerowanych
# sygnałów przy takich samych parametrach fn oraz Tc jak w poprzednim ćwieczeniu.
#--------------------------------------------------------------------------------------------------------------------------------------
def main():
    # parametry próbkowania
    Tc = 1                  # [s] czas trwania sygnału, (czas próbkowania w sekundach)
    fs = 8000               # [Hz] częstotliwosć probkowania
    N = int(Tc * fs)        # liczba próbek przypadających na cały sygnal
    Ts = 1 / fs             # okres próbkowania (powina być co najmniej dwa razy więksa niż górna granica częstotliwości sygnalu próbkowanego fmax)
    t = np.arange(N) * Ts   # czas próbkowania
    
    
    plt.plot(t, y(t))
    plt.xlabel('Czas [s]')
    plt.ylabel('Aplituda')
    plt.title('Wykres sygnalu y(t) o częstotliwości 1 kHz')
    plt.savefig('./zadanie2/y.png')
    plt.show()
    
    plt.plot(t, z(t))
    plt.xlabel('Czas [s]')
    plt.ylabel('Aplituda')
    plt.title('Wykres sygnalu z(t) o częstotliwości 1 kHz')
    plt.savefig('./zadanie2/z.png')
    plt.show()
    
    plt.plot(t, v(t))
    plt.xlabel('Czas [s]')
    plt.ylabel('Aplituda')
    plt.title('Wykres sygnalu v(t) o częstotliwości 1 kHz')
    plt.savefig('./zadanie2/v.png')
    plt.show()

if __name__ == "__main__":
    main()