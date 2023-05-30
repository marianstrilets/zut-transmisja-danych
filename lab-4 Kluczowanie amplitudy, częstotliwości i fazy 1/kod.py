import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------------------------------
# 1. Napisać funkcję zamieniającą dowolny napis w formacie ASCII (kody od 32 do 127)
#    na strumien bitowy kazdemu znaku odpowiada 7-bitowa reprezentcja binarna.
# ------------------------------------------------------------------------------------------------------


def ascii_to_bit(bn):
    # funkcja zaminia ascii na strumien bitowy
    bits = ''
    for char in bn:
        # Sprawdzenie, czy kod ASCII mieści się w zakresie od 32 do 127
        if 32 <= ord(char) <= 127:
            # Zamiana kodu ASCII na 7-bitową reprezentację binarną
            binary = bin(ord(char))[2:].zfill(7)
            # Dodanie reprezentacji binarnej do strumienia bitowego
            bits += binary
    return bits


# ------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------
# 2. Dla dowolnego strumienia bitowego b[n] przyjac czas trwania
#   pojedenczego bitu Tb [s]. Nastepnie nalezy dobrac parametry
#   A1, A2, (A1 != A2), fn = W*Tb^-1 oraz wygenerowac sygnaly
#   zmodulowane za(t), zp(t) oraz zp(t). Cżestotliwosci w przypadku
#   kluczowania FSK mozna dobrac wedlug nastepujacych zaleznosci
#   fn1 = (W + 1) / Tb fn2 = (W + 2) / Tb
#   gzie W jest liczbą calkowitą okręślującą docelową częstotliwość
#   (po wymnozeniu przez odwrotność czasu trwania pojedęczego bitu)
# ------------------------------------------------------------------------------------------------------
#bn = 'ULz'  # strumień bitowy
bn = 'ab'  # strumień bitowy
#bn = 'qwerty'    # strumień bitowy
Tc = 1      # [s] czas trwania sygnału, (czas próbkowania w sekundach)
bits = ascii_to_bit(bn)
B = len(bits)   # liczba bitów sygnalu informatycznego
Tb = Tc / B         # czas trwania pojedynszego bitu Tc=>czas trwania sygnalu B=>ilośc bitów
W = 2               # przebieg czasowy
print(bits)

fn = W / Tb         # zaleznosci cżestotliwosci w przypadku kluczowania PSK
#fn = 5

fn1 = (W + 1) / Tb  # zaleznosci cżestotliwosci w przypadku kluczowania FSK fn1
#fn1 = 50

fn2 = (W + 2) / Tb  # zaleznosci cżestotliwosci w przypadku kluczowania FSK fn2
#fn2 = 10
print(fn1)
print(fn2)
#fs = 8000               # [Hz] częstotliwosć probkowania
fs =1000
N = int(Tc * fs)        # liczba próbek przypadających na cały sygnal
# okres próbkowania (powina być co najmniej dwa razy więksa niż górna granica częstotliwości sygnalu próbkowanego fmax)
Ts = 1 / fs
t = np.linspace(0, Tc, N)

# Kluczowanie z przesuwem amplitudy (ASK)


def za(t, b, A1=1, A2=2):
    # parametry amplitudy  przedział <A1, A2>
    if b == 0:
        return A1 * np.sin(2 * np.pi * fn * t)
    elif b == 1:
        return A2 * np.sin(2 * np.pi * fn * t)

# Kluczowanie z przesuwem fazy (PSK)


def zp(t, b):
    if b == 0:
        return np.sin(2 * np.pi * fn * t)
    elif b == 1:
        return np.sin(2 * np.pi * fn * t + np.pi)

# Kluczowanie z przesuwem częstotliwości (FSK)


def zf(t, b):
    if b == 0:
        return np.sin(2 * np.pi * fn1 * t)
    elif b == 1:
        return np.sin(2 * np.pi * fn2 * t)


# ------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------
# 3. Wygenerować sygnały za(t), zp(t), oraz zf(t) dla W = 2 oraz ich przebiegi
#   czasowe. Przy generowaniu wykresu ograniczyc liczbę bitów do B=10
# ------------------------------------------------------------------------------------------------------
# Generowanie sygnalu
#B = min(B, 10)  # generowaniu ograniczyc liczbę bitów do B=10
B = min(B, 3) 

signal_za = []
signal_zp = []
signal_zf = []

for i in range(B):
    b = int(bits[i])
    print(b)
    signal_za += list(za(t, b))
    signal_zp += list(zp(t, b))
    signal_zf += list(zf(t, b))
# ------------------------------------------------------------------------------------------------------
plt.plot(signal_za, color='blue')
plt.title('Sygnał ASK')
plt.xlabel('Czas [s]')
plt.ylabel('Amplituda')
plt.savefig('./za.png')
plt.show()

plt.plot(signal_zp, color='red')
plt.title('Sygnał PSK')
plt.xlabel('Czas [s]')
plt.ylabel('Amplituda')
plt.savefig('./zp.png')
plt.show()

plt.plot(signal_zf, color='green')
plt.title('Sygnał FSK')
plt.xlabel('Czas [s]')
plt.ylabel('Amplituda')
plt.savefig('./zf.png')
plt.show()

# ------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------
# 4. Wygenerować widma amplitudowe w skali decybelowej sygnałow zmodulowanych za(t), zp(t) oraz zf(t).
#   W tym przypadku sygnał żrodlowy powinien odwzrciedłac cały strumień bitowy. Nałezy tak dobrać skałe
#   częstotliwościową (liniową lub algorytmiczną) aby jak najwięczej prażków
#   widma było widocznych na wykresie
# ------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------
# Obliczenie widm amplitudowych sygnałów zmodulowanych
widmo_za = np.abs(np.fft.fft(signal_za))
widmo_zp = np.abs(np.fft.fft(signal_zp))
widmo_zf = np.abs(np.fft.fft(signal_zf))
# ----------------------------------------

# Skala osi częstotliwości w skali decybelowej
widmo_za_db = 10 * np.log10(widmo_za)
widmo_zp_db = 10 * np.log10(widmo_zp)
widmo_zf_db = 10 * np.log10(widmo_zf)

# ----------------------------------------
# Wykresy widm amplitudowych w skali decybelowej
freq = [] # skala_czestosci
for k in range(len(widmo_za)):
    freq.append(k * (fs/N) )

half_freq = int(len(freq) / 2)

freq = freq[: half_freq]

# ----------------------------------------
# Widmo amplitudowe: za(t)
plt.figure("Widmo amplitudowe: za(t)")
plt.stem(freq, widmo_za_db[: half_freq])
plt.title("Widmo amplitudowe: za(t)")
plt.xlabel("Częstotliwość [Hz]")
plt.ylabel("Amplituda [dB]")
plt.savefig("./za_widmo.png")
plt.show()

# Widmo amplitudowe: zp(t)
plt.figure("Widmo amplitudowe: zp(t)")
plt.stem(freq, widmo_zp_db[: half_freq])
plt.title("Widmo amplitudowe: zp(t)")
plt.xlabel("Częstotliwość [Hz]")
plt.ylabel("Amplituda [dB]")
plt.savefig("./zp_widmo.png")
plt.show()

# Widmo amplitudowe: zf(t)
plt.figure("Widmo amplitudowe: zf(t)")
plt.stem(freq, widmo_zf_db[: half_freq])
plt.title("Widmo amplitudowe: zf(t)")
plt.xlabel("Częstotliwość [Hz]")
plt.ylabel("Amplituda [dB]")
plt.savefig("./zf_widmo.png")
plt.show()
# ------------------------------------------------------------------------------------------------------
# 5. Oszacować szerokość pasma B3dB, B6dB oraz B12dB sygnału zmodułowanego dla
#   każdego z rodzajów kluczowania (ASK, PSK oraz FSK).
# ------------------------------------------------------------------------------------------------------
# Funkcja oblicz szerokosc pasma
def BxdB(A, f, dB):
    A = A[:int(len(A) / 2)]
    max_amplitude = max(A) - 10 * np.log10(dB)
    f_min = f[0]
    f_max = f[-1]
    
    for i in range(len(A)):
        if A[i] > max_amplitude:
            f_min = f[i]
            break
    for i in range(len(A)-1, -1, -1):
        if A[i] > max_amplitude:
            f_max = f[i]
            break
    return f_max - f_min


print('\nSzerokość pasma za(t):')
print('\tSzerokość pasma B3dB: ', BxdB(widmo_za_db, freq, 3))
print('\tSzerokość pasma B6dB: ', BxdB(widmo_za_db, freq, 6))
print('\tSzerokość pasma B12dB: ', BxdB(widmo_za_db, freq, 12))

print('\nSzerokość pasma zp(t):')
print('\tSzerokość pasma B3dB: ', BxdB(widmo_zp_db, freq, 3))
print('\tSzerokość pasma B6dB: ', BxdB(widmo_zp_db, freq, 6))
print('\tSzerokość pasma B12dB: ', BxdB(widmo_zp_db, freq, 12))

print('\nSzerokość pasma zf(t):')
print('\tSzerokość pasma B3dB: ', BxdB(widmo_zf_db, freq, 3))
print('\tSzerokość pasma B6dB: ', BxdB(widmo_zf_db, freq, 6))
print('\tSzerokość pasma B12dB: ', BxdB(widmo_zf_db, freq, 12))


# ------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------
