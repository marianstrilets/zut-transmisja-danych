import numpy as np
import matplotlib.pyplot as plt


def b(t):
    Hk = [2, 6, 10]
    values = []
    for h in Hk:
        values.append(np.sin( h * t * np.pi) / (2 + np.cos(h**2 * np.pi * t)))
    return values
        
    
    
#------------------------------------- Zadanie 4 --------------------------------------------------------------------------------------
# Wygenerować i wykreslić sygnały bk(t) (k = 1,2,3) dla  fs = 22.05 kHz oraz Tc = 1s
#--------------------------------------------------------------------------------------------------------------------------------------

def main():
    # parametry próbkowania
    Tc = 1                  # [s] czas trwania sygnału, (czas próbkowania w sekundach)
    fs = 8000               # [Hz] częstotliwosć probkowania
    N = int(Tc * fs)        # liczba próbek przypadających na cały sygnal
    Ts = 1 / fs             # okres próbkowania (powina być co najmniej dwa razy więksa niż górna granica częstotliwości sygnalu próbkowanego fmax)
    t = np.arange(N) * Ts   # czas próbkowania
    
    value = b(t)
    cnt = 1
    for i in value: 
        plt.plot(t, i)
        plt.xlabel('Czas [s]')
        plt.ylabel('Aplituda')
        plt.title('Wykres sygnalu b' + str(cnt)  + '(t)')
        plt.savefig('./zadanie4/b' + str(cnt) + '.png')
        cnt = cnt+1
        plt.show()

        

if __name__ == "__main__":
    main()