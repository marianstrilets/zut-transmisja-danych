# =============================================================================================================
# laboratorium 3
# Modulacja amplitudy i kąta
# =============================================================================================================
# =============================================================================================================
import numpy as np
import matplotlib.pyplot as plt
# -------------------------------------------------------------------------------------------------------------
# parametry próbkowania
fm = 10     # częstotliwość modulująca
fn = 100    # częstotliwośc nośna
Tc = 1      # [s] czas trwania sygnału, (czas próbkowania w sekundach)
fs = 300    # [Hz] częstotliwosć probkowania

N = int(Tc * fs)        # liczba próbek przypadających na cały sygnal
Ts = 1 / fs             # okres próbkowania
t = np.arange(N) * Ts   # Tablica czasu od 0 do N, z krokiem ts

m = np.sin(2 * np.pi * fm * t)  # sygnał zmodulowany jednym tonem
# =============================================================================================================
#    Zadanie 1:
# 1. Wygenerować sygnały zmodulowane za(t), zp(t) oraz zf(t) dla nastempujących przypadków (fn oraz fm należy
#   dobrać tak aby był spełniony warunek fn>>fm):
#   --  modulacja amplitudy     :    a)1> ka >0;      b)12 > ka > 2;      c)ka > 20;
#   --  modulacja fazy          :    a)kp < 1;        b)pi > kp > 0;      c)kp > 2pi;
#   --  modulacja częstotliwości:    a)kf < 1;        b)pi > kf > 0;      c)kf > 2pi;
# =============================================================================================================
# modulacja amplitudy
za_a = []
za_b = []
za_c = []
ka = {'a': 0.5, 'b': 5, 'c': 25}  # Współczynnik modulacji amplitudy

for i in range(len(t)):
    za_a.append((ka['a'] * m[i] + 1) * np.cos(2 * np.pi * fn * t[i]))
    za_b.append((ka['b'] * m[i] + 1) * np.cos(2 * np.pi * fn * t[i]))
    za_c.append((ka['c'] * m[i] + 1) * np.cos(2 * np.pi * fn * t[i]))
# -------------------------------------------------------------------------------------------------------------
# modulacja fazy
zp_a = []
zp_b = []
zp_c = []
kp = {'a': 0.5, 'b': 2, 'c': 10}  # Współczynnik fazy amplitudy

for i in range(len(t)):
    zp_a.append((np.cos(2 * np.pi * fn * t[i] + kp['a'] * m[i])))
    zp_b.append((np.cos(2 * np.pi * fn * t[i] + kp['b'] * m[i])))
    zp_c.append((np.cos(2 * np.pi * fn * t[i] + kp['c'] * m[i])))
# -------------------------------------------------------------------------------------------------------------
# modulacja częstotliwości
zf_a = []
zf_b = []
zf_c = []
kf = {'a': 0.5, 'b': 2, 'c': 3}  # Współczynnik częstotliwości amplitudy

for i in range(len(t)):
    zf_a.append((np.cos(2 * np.pi * fn * t[i] + kf['a']/fm * m[i])))
    zf_b.append((np.cos(2 * np.pi * fn * t[i] + kf['b']/fm * m[i])))
    zf_c.append((np.cos(2 * np.pi * fn * t[i] + kf['c']/fm * m[i])))
# -------------------------------------------------------------------------------------------------------------
# =============================================================================================================
#    Zadanie 2:
# 2. Narysować widma amplitudowe w skali decydebolej sygnałow zmodulowanych za(t), zp(t) oraz zf(t).
#   Należy tak dobrać skałe osi częstotliwości (liniową lub logarytmiczną) aby jak najwięcej prażków widma było
#   widoczbych na wykresie.
# =============================================================================================================
# Obliczenie widm amplitudowych sygnałów zmodulowanych
widmo_za_a = np.abs(np.fft.fft(za_a))
widmo_za_b = np.abs(np.fft.fft(za_b))
widmo_za_c = np.abs(np.fft.fft(za_c))
widmo_zp_a = np.abs(np.fft.fft(zp_a))
widmo_zp_b = np.abs(np.fft.fft(zp_b))
widmo_zp_c = np.abs(np.fft.fft(zp_c))
widmo_zf_a = np.abs(np.fft.fft(zf_a))
widmo_zf_b = np.abs(np.fft.fft(zf_b))
widmo_zf_c = np.abs(np.fft.fft(zf_c))

# Skala osi częstotliwości w skali decybelowej
widmo_za_a_db = 10 * np.log10(widmo_za_a)
widmo_za_b_db = 10 * np.log10(widmo_za_b)
widmo_za_c_db = 10 * np.log10(widmo_za_c)
widmo_zp_a_db = 10 * np.log10(widmo_zp_a)
widmo_zp_b_db = 10 * np.log10(widmo_zp_b)
widmo_zp_c_db = 10 * np.log10(widmo_zp_c)
widmo_zf_a_db = 10 * np.log10(widmo_zf_a)
widmo_zf_b_db = 10 * np.log10(widmo_zf_b)
widmo_zf_c_db = 10 * np.log10(widmo_zf_c)

# -------------------------------------------------------------------------------------------------------------
# Wykresy widm amplitudowych w skali decybelowej
# -------------------------------------------------------------------------------------------------------------
freq = [] # skala_czestosci
for k in range(len(widmo_za_a)):
    freq.append(k * fs/N)
# ----------------------------------------
# Widmo amplitudowe: za_a
plt.figure("Widmo amplitudowe: za_a(t)")
plt.plot(freq, widmo_za_a_db, color="red")
plt.title("Widmo amplitudowe: za_a(t)")
plt.xlabel("Częstotliwość [Hz]")
plt.ylabel("Amplituda [dB]")
plt.savefig("./za_a.png")
plt.show()
# ----------------------------------------
# Widmo amplitudowe: za_b
plt.figure("Widmo amplitudowe: za_b(t)")
plt.plot(freq, widmo_za_b_db, color="red")
plt.title("Widmo amplitudowe: za_b(t)")
plt.xlabel("Częstotliwość [Hz]")
plt.ylabel("Amplituda [dB]")
plt.savefig("./za_b.png")
plt.show()
# ----------------------------------------
# Widmo amplitudowe: za_c
plt.figure("Widmo amplitudowe: za_c(t)")
plt.plot(freq, widmo_za_c_db, color="red")
plt.title("Widmo amplitudowe: za_c(t)")
plt.xlabel("Częstotliwość [Hz]")
plt.ylabel("Amplituda [dB]")
plt.savefig("./za_c.png")
plt.show()
# -------------------------------------------------------------------------------------------------------------
# Widmo amplitudowe: zp_a
plt.figure("Widmo amplitudowe: zp_a(t)")
plt.plot(freq, widmo_zp_a_db, color="green")
plt.title("Widmo amplitudowe: zp_a(t)")
plt.xlabel("Częstotliwość [Hz]")
plt.ylabel("Amplituda [dB]")
plt.savefig("./zp_a.png")
plt.show()
# ----------------------------------------
# Widmo amplitudowe: zp_b
plt.figure("Widmo amplitudowe: zp_b(t)")
plt.plot(freq, widmo_zp_b_db, color="green")
plt.title("Widmo amplitudowe: zp_b(t)")
plt.xlabel("Częstotliwość [Hz]")
plt.ylabel("Amplituda [dB]")
plt.savefig("./zp_b.png")
plt.show()
# ----------------------------------------
# Widmo amplitudowe: zp_c
plt.figure("Widmo amplitudowe: zp_c(t)")
plt.plot(freq, widmo_zp_c_db, color="green")
plt.title("Widmo amplitudowe: zp_c(t)")
plt.xlabel("Częstotliwość [Hz]")
plt.ylabel("Amplituda [dB]")
plt.savefig("./zp_c.png")
plt.show()
# -------------------------------------------------------------------------------------------------------------
# Widmo amplitudowe: zf_a
plt.figure("Widmo amplitudowe: zf_a(t)")
plt.plot(freq, widmo_zf_a_db, color="orange")
plt.title("Widmo amplitudowe: zf_a(t)")
plt.xlabel("Częstotliwość [Hz]")
plt.ylabel("Amplituda [dB]")
plt.savefig("./zf_a.png")
plt.show()
# ----------------------------------------
# Widmo amplitudowe: zf_b
plt.figure("Widmo amplitudowe: zf_b(t)")
plt.plot(freq, widmo_zf_b_db, color="orange")
plt.title("Widmo amplitudowe: zf_b(t)")
plt.xlabel("Częstotliwość [Hz]")
plt.ylabel("Amplituda [dB]")
plt.savefig("./zf_b.png")
plt.show()
# ----------------------------------------
# Widmo amplitudowe: zf_c
plt.figure("Widmo amplitudowe: zf_c(t)")
plt.plot(freq, widmo_zp_c_db, color="orange")
plt.title("Widmo amplitudowe: zf_c(t)")
plt.xlabel("Częstotliwość [Hz]")
plt.ylabel("Amplituda [dB]")
plt.savefig("./zf_c.png")
plt.show()
# -------------------------------------------------------------------------------------------------------------
# =============================================================================================================
#    Zadanie 3:
# 3. Oszacować szerokość pasma B3dB, B6dB, B12dB sygnału zmodulowanego w sposób praedstawiony na rys.1
# =============================================================================================================

# Funkcja oblicz szerokosc pasma
def BxdB(A, f, dB):
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

print('\nSzerokość pasma:')

print('\tza_a(t): ', BxdB(widmo_za_a_db, freq, 3))
print('\tza_b(t): ', BxdB(widmo_za_b_db, freq, 6))
print('\tza_c(t): ', BxdB(widmo_za_c_db, freq, 12))
print('\n')

print('\tzp_a(t): ', BxdB(widmo_zp_a_db, freq, 3))
print('\tzp_b(t): ', BxdB(widmo_zp_b_db, freq, 6))
print('\tzp_c(t): ', BxdB(widmo_zp_c_db, freq, 12))
print('\n')

print('\tzf_a(t): ', BxdB(widmo_zf_a_db, freq, 3))
print('\tzf_b(t): ', BxdB(widmo_zf_b_db, freq, 6))
print('\tzf_c(t): ', BxdB(widmo_zf_c_db, freq, 12))
print('\n')



