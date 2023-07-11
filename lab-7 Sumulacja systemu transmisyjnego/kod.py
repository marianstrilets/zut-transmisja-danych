import numpy as np
import matplotlib.pyplot as plt

# ================================================================================================================================
# 1. Zbudować model systemu transmisyjnego predstawionego na rys.1. Implementację należy zrealizować w taki sposób, aby poszczególne
# bloki były łatwe do podmiany. Wyjście z modulatora (punkt A) powinno być połaczone z wejściem demodulatora (punkt B). Weryfikacje
# dzialania należy wykonać poprzez porównanie strumienia bitowego wprowadzonego po stronie nadawczej ze strumieniem po stronie odbiorcze.
# Testy nalęzy wykonać dla jednej modulacji (ASK, PSK lub FSK) oraz wybranego kodu Hamminga.
#
# 2. Nałezy zmodyfikować  układ z rys.1 dodając między punktami A i B układ przedstawiony na rys.2. Układ generujący sygnał g(t) jest
# generatorem szumu białego (nałezy użyć dowolnej, gotowej implementacji). Sygnały x(t), oraz g(t) powinny być znormalizowany do zakresu
# [-1, 1]. W przygotowanym układzie nałezy zbadać załeżność współczynika BER od parametru alfa dla modulacji ASK, PSK oraz FSK.
#
# 3. Przeprowadzić modyfikację układu z rys.1 analogicznie do cziczenia 2 użyciem układu pokazanego na rys.3. W przygotowanym układzie nalęzy
# zbadać załezność współczynika BER od parametru beta dla modulacji ASK, PSK, FSK.
#
# 4. Rozbudować układ z rys.1 z wykorzystaniem połaczenia kaskadowego układow z rys.2 (1) oraz 3 (2)
# w dwóch konfiguracjach: (1) 1 + 2 oraz (2) 2+ 1. W przygotowanych układach nałezy zbadać załeznośc wspołczynnika BER od parametrów
# alfa oraz beta dla modulacji ASK, PSK i FSK.
# ================================================================================================================================


class Hamming():
    def __init__(self, binary_info, n, k):
        self.binary_info = binary_info
        self.n = n
        self.k = k
        self.encoded = None
        self.decoded = None

    def generate_column(self, n, m):
        base_pattern = [0]*(2**(m-1)) + [1]*(2**(m-1))
        repeats = n // (2**m)
        column = np.tile(base_pattern, repeats)
        return column

    def generate_P_matrix(self):
        column1 = self.generate_column(self.n+1, 1)
        column2 = self.generate_column(self.n+1, 2)
        column3 = self.generate_column(self.n+1, 3)
        column4 = self.generate_column(self.n+1, 4)
        gray_matrix = np.array([column1, column2, column3, column4]).T
        for i in range(3):
            gray_matrix = np.delete(gray_matrix, 0, axis=0)
        gray_matrix = np.delete(gray_matrix, [1, 4], axis=0)
        return gray_matrix

    def coding_hamming_15_11(self):
        P = self.generate_P_matrix()
        I_k = np.eye(self.k)
        G = np.hstack((P, I_k))

        b = np.array([int(bit) for bit in self.binary_info], dtype='int')

        c = np.dot(b, G) % 2
        c = c.astype(int)

        self.encoded = ''.join(map(str, c))

    def decoding_hamming_15_11(self, encoded):
        c = list(map(int, encoded))

        P = self.generate_P_matrix().T
        I_n_k = np.eye(self.n - self.k)
        H = np.hstack((I_n_k, P))

        s = np.dot(c, H.T) % 2
        S = sum(s[i]*2**i for i in range(4))

        self.decoded = ''
        if(S != 0):
            indices = [1, 2, 4, 8, 3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15]
            index = indices.index(S)
            c[index] = 1 - c[index]
        self.decoded = ''.join(map(str, c[4:]))


class Modulation:
    def __init__(self, bn):
        self.fs = 1988   # [Hz] częstotliwość próbkowania
        self.bn = bn
        self.bits = len(self.bn)
        self.noweBn = None
        self.B = len(self.bn)
        # [s] czas trwania sygnału (czas próbkowania w sekundach)
        self.Tc = 1
        # liczba próbek przypadających na cały sygnał
        self.N = int(self.Tc*self.fs)
        # czas trwania pojedynczego bitu Tc => czas trwania sygnału B => liczba bitów
        self.Tb = self.Tc/self.B
        self.Tbp = int(self.N/self.B)   # [probki]
        self.W = 2                      # przebieg czasowy
        self.A = 10
        self.A1 = 5
        self.A2 = 10
        # Częstotliwość nośna - fn
        # zależność częstotliwości w przypadku kluczowania PSK
        self.fn = self.W / self.Tb
        # zależność częstotliwości w przypadku kluczowania FSK fn1
        self.fn1 = (self.W + 1) / self.Tb
        # zależność częstotliwości w przypadku kluczowania FSK fn2
        self.fn2 = (self.W + 2) / self.Tb
        self.h = 1200

        self.sig_ask_z = self.create_signal('ASK')
        self.sig_ask_x = self.ask_x()
        self.sig_ask_p = self.ask_p()
        self.sig_ask_c = self.ask_c()

        self.sig_psk_z = self.create_signal('PSK')
        self.sig_psk_x = self.psk_x()
        self.sig_psk_p = self.psk_p()
        self.sig_psk_c = self.psk_c()

        self.sig_fsk_z = self.create_signal('FSK')
        self.sig_fsk_x1 = self.fsk_x(1)
        self.sig_fsk_x2 = self.fsk_x(2)
        self.sig_fsk_p1 = self.fsk_pn(self.sig_fsk_x1)
        self.sig_fsk_p2 = self.fsk_pn(self.sig_fsk_x2)
        self.sig_fsk_p = self.fsk_p()
        self.sig_fsk_c = self.fsk_c()

        self.sig_noisy_ask_c = None
        self.sig_noisy_psk_c = None
        self.sig_noisy_fsk_c = None

    @staticmethod
    def ascii_to_bit(bn):
        # funkcja zamienia ascii na strumień bitowy
        bits = ''
        for char in bn:
            # Sprawdzenie, czy kod ASCII mieści się w zakresie od 32 do 127
            if 32 <= ord(char) <= 127:
                # Zamiana kodu ASCII na 7-bitową reprezentację binarną
                binary = bin(ord(char))[2:].zfill(7)
                # Dodanie reprezentacji binarnej do strumienia bitowego
                bits += binary
        return bits

    def create_signal(self, set_signal):
        if set_signal == 'ASK':
            x = []
            index = 0
            tempTbp = 1
            for n in range(self.N):
                x.append(self.ask_z(n/self.fs, self.bn[index]))
                if(tempTbp == self.Tbp and index < self.B-1):
                    index += 1
                    tempTbp = 0
                tempTbp += 1
            return x
        elif set_signal == 'PSK':
            x = []
            index = 0
            tempTbp = 1
            for n in range(self.N):
                x.append(self.psk_z(n/self.fs, self.bn[index]))
                if(tempTbp == self.Tbp and index < self.B-1):
                    index += 1
                    tempTbp = 0
                tempTbp += 1
            return x
        elif set_signal == 'FSK':
            x = []
            index = 0
            tempTbp = 1
            for n in range(self.N):
                x.append(self.fsk_z(n/self.fs, self.bn[index]))
                if(tempTbp == self.Tbp and index < self.B-1):
                    index += 1
                    tempTbp = 0
                tempTbp += 1
            return x

    # ================================= ASK ===============================================
    # Kluczowanie z przesunięciem amplitudy (ASK)
    def ask_z(self, t, b):
        # parametry amplitudy przedział <A1, A2>
        if b == '0':
            return self.A1 * np.sin(2 * np.pi * self.fn * t)
        elif b == '1':
            return self.A2 * np.sin(2 * np.pi * self.fn * t)

    def ask_x(self):
        ask_x_ = []
        for n in range(self.N):
            ask_x_.append(self.sig_ask_z[n] * self.A *
                          np.sin(2*np.pi*self.fn*(n/self.fs)))
        return ask_x_

    def ask_p(self):
        ask_p_ = []
        index = 0
        for i in range(self.B):
            ask_p_.append(self.sig_ask_x[index])
            index += 1
            for j in range(self.Tbp-1):
                ask_p_.append(ask_p_[index-1]+self.sig_ask_x[index])
                index += 1
        return ask_p_

    def ask_c(self):
        ask_c_ = []
        for i in range(self.N):
            if i < len(self.sig_ask_p):
                if self.sig_ask_p[i] > self.h:
                    ask_c_.append(1)
                else:
                    ask_c_.append(0)
        return ask_c_

    # ================================== PSK ===============================================
    # Kluczowanie z przesunięciem fazy (PSK)
    def psk_z(self, t, b):
        if b == '0':
            return np.sin(2 * np.pi * self.fn * t)
        elif b == '1':
            return np.sin(2 * np.pi * self.fn * t + np.pi)

    def psk_x(self):
        psk_x_ = []
        for n in range(self.N):
            psk_x_.append(self.sig_psk_z[n] * self.A *
                          np.sin(2*np.pi*self.fn*(n/self.fs)))
        return psk_x_

    def psk_p(self):
        psk_p_ = []
        index = 0
        for i in range(self.B):
            psk_p_.append(self.sig_psk_x[index])
            index += 1
            for j in range(self.Tbp-1):
                psk_p_.append(psk_p_[index-1] + self.sig_psk_x[index])
                index += 1
        return psk_p_

    def psk_c(self):
        psk_c_ = []
        for i in range(self.N):
            if i < len(self.sig_psk_p):
                if self.sig_psk_p[i] < 0:
                    psk_c_.append(1)
                else:
                    psk_c_.append(0)
        return psk_c_

    # ================================== FSK ===============================================
    # Kluczowanie z przesunięciem częstotliwości (FSK)
    def fsk_z(self, t, b):
        if b == '0':
            return np.sin(2 * np.pi * self.fn1 * t)
        elif b == '1':
            return np.sin(2 * np.pi * self.fn2 * t)

    def fsk_x(self, num):
        fsk_x_ = []
        if(num == 1):
            for n in range(self.N):
                fsk_x_.append(
                    self.sig_fsk_z[n] * np.sin(2 * np.pi * self.fn1 * (n/self.fs)))
        elif(num == 2):
            for n in range(self.N):
                fsk_x_.append(
                    self.sig_fsk_z[n] * np.sin(2 * np.pi * self.fn2 * (n/self.fs)))
        return fsk_x_

    def fsk_pn(self, sig):
        fsk_pn_ = []
        index = 0
        for i in range(self.B):
            fsk_pn_.append(sig[index])
            index += 1
            for j in range(self.Tbp-1):
                fsk_pn_.append(fsk_pn_[index-1]+sig[index])
                index += 1
        return fsk_pn_

    def fsk_p(self):
        fsk_p_ = []
        for i in range(self.N):
            if i < len(self.sig_fsk_p1) and i < len(self.sig_fsk_p2):
                fsk_p_.append(self.sig_fsk_p2[i] - self.sig_fsk_p1[i])
        return fsk_p_

    def fsk_c(self):
        fsk_c = []
        for i in range(self.N):
            if i < len(self.sig_fsk_p):
                if self.sig_fsk_p[i] > 0:
                    fsk_c.append(1)
                else:
                    fsk_c.append(0)
        return fsk_c

    def add_noise(self, signal, alfa, beta):
        noise = np.random.normal(alfa, beta, len(signal))
        noisy_signal = signal + noise
        return noisy_signal

    def cascade_connection(self, signal1, signal2, alfa, beta):
        noisy_signal1 = self.add_noise(signal1, alfa, beta)
        noisy_signal2 = self.add_noise(signal2, alfa, beta)
        combined_signal = []
        for i in range(len(signal1)):
            combined_signal.append(noisy_signal1[i] + noisy_signal2[i])
        return combined_signal

    def calculate_ber(self, original_signal, received_signal):
        errors = np.sum(
            np.abs(np.array(original_signal) - np.array(received_signal)))
        ber = errors / len(original_signal)
        return ber

    def simulate_cascade_configuration(self, alfa, beta):
        self.sig_noisy_ask_c = self.cascade_connection(
            self.sig_ask_c, self.sig_psk_c, alfa, beta)
        self.sig_noisy_psk_c = self.cascade_connection(
            self.sig_psk_c, self.sig_fsk_c, alfa, beta)
        self.sig_noisy_fsk_c = self.cascade_connection(
            self.sig_fsk_c, self.sig_ask_c, alfa, beta)

        ber_ask_c = self.calculate_ber(self.sig_ask_c, self.sig_noisy_ask_c)
        ber_psk_c = self.calculate_ber(self.sig_psk_c, self.sig_noisy_psk_c)
        ber_fsk_c = self.calculate_ber(self.sig_fsk_c, self.sig_noisy_fsk_c)
        return ber_ask_c, ber_psk_c, ber_fsk_c

    def generate_noise_white(self, choose_modulate, alfa_values):
        n = 0
        if(choose_modulate == "ASK"):
            n = len(self.sig_ask_z)
        elif(choose_modulate == "PSK"):
            n = len(self.sig_psk_z)
        elif(choose_modulate == "FSK"):
            n = len(self.sig_fsk_z)

        noise = np.random.uniform(-1, 1, n)
        for i in range(n):
            if(choose_modulate == "ASK"):
                self.sig_ask_z[i] += alfa_values * noise[i]
            elif(choose_modulate == "PSK"):
                self.sig_psk_z[i] += alfa_values * noise[i]
            elif(choose_modulate == "FSK"):
                self.sig_fsk_z[i] += alfa_values * noise[i]

    def generate_noise_suppression(self, choose_modulate, beta_values):
        n = 0
        if(choose_modulate == "ASK"):
            n = len(self.sig_ask_z)
        elif(choose_modulate == "PSK"):
            n = len(self.sig_psk_z)
        elif(choose_modulate == "FSK"):
            n = len(self.sig_fsk_z)

        noise = []
        for i in range(n):
            t = i/self.fs
            noise.append(np.e ** (-beta_values * t))
            if(choose_modulate == "ASK"):
                self.sig_ask_z[i] *= noise[i]
            elif(choose_modulate == "PSK"):
                self.sig_psk_z[i] *= noise[i]
            elif(choose_modulate == "FSK"):
                self.sig_fsk_z[i] *= noise[i]


def demodulation_dekoder(modulation, choose_modulate):
    if(choose_modulate == 'ASK'):
        modulation.ask_x()
        modulation.ask_p()
        modulation.ask_c()
    elif(choose_modulate == 'PSK'):
        modulation.psk_x()
        modulation.psk_p()
        modulation.psk_c()
    elif(choose_modulate == 'FSK'):
        modulation.fsk_p()
        modulation.fsk_c()

def gen_mod(modulation, noise, choose_modulate, alfa, beta, res_all = False): 
    ber_results = []           
    ber_ask_c, ber_psk_c, ber_fsk_c = modulation.simulate_cascade_configuration(
        alfa, beta)
    ber_results.append([alfa, beta, ber_ask_c, ber_psk_c, ber_fsk_c])
    # Szum biały
    if(noise == 'noise_white'):
        modulation.generate_noise_white(choose_modulate, alfa)
    # Szum tłumienia
    elif(noise == 'noise_suppression'):
        modulation.generate_noise_suppression(choose_modulate, beta)
    # Szum konfiguracja 1
    elif(noise == 'configuration1'):
        modulation.generate_noise_white(choose_modulate, alfa)
        modulation.generate_noise_suppression(choose_modulate, beta)
    # Szum konfiguracja 2
    elif(noise == 'configuration2'):
        modulation.generate_noise_suppression(choose_modulate, beta)
        modulation.generate_noise_white(choose_modulate, alfa)
    if(res_all == True):
        return ber_results
    else:
        if(choose_modulate == "ASK"):
            return ber_results[2]
        elif(choose_modulate == "PSK"):
            return ber_results[3]
        elif(choose_modulate == "FSK"):
            return ber_results[4]

def generate_modulation(noise, choose_modulate, alfa_values, beta_values, output_res=True):
    modulation = Modulation(info_binary)
    ber_results = []

    ber_ask = []
    ber_psk = []
    ber_fsk = []
    for i in range(20):
        for alfa in alfa_values:
            for beta in beta_values:
                #ber_results = gen_mod(modulation, noise, choose_modulate, alfa, beta, True)
                ber_ask_c, ber_psk_c, ber_fsk_c = modulation.simulate_cascade_configuration(
                    alfa, beta)
                if(i == 0):
                    ber_results.append([alfa, beta, ber_ask_c, ber_psk_c, ber_fsk_c])
                # Szum biały
                if(noise == 'noise_white'):
                    modulation.generate_noise_white(choose_modulate, alfa)
                # Szum tłumienia
                elif(noise == 'noise_suppression'):
                    modulation.generate_noise_suppression(choose_modulate, beta)
                # Szum konfiguracja 1
                elif(noise == 'configuration1'):
                    modulation.generate_noise_white(choose_modulate, alfa)
                    modulation.generate_noise_suppression(choose_modulate, beta)
                # Szum konfiguracja 2
                elif(noise == 'configuration2'):
                    modulation.generate_noise_suppression(choose_modulate, beta)
                    modulation.generate_noise_white(choose_modulate, alfa)                   
                demodulation_dekoder(modulation, choose_modulate)
                if(choose_modulate == "ASK"):
                    ber_ask.append(ber_ask_c)
                elif(choose_modulate == "PSK"):
                    ber_psk.append(ber_psk_c)
                elif(choose_modulate == "FSK"):
                    ber_fsk.append(ber_fsk_c)
    if(output_res == True):
        # Wyświetlanie wyników
        print("\t" * 3, "Zależność współczynnika BER od parametrów alfa i beta:\n")
        for result in ber_results:
            print("Alfa:", result[0], "Beta:", result[1], "\tBER ASK:",
                  result[2], "\tBER PSK:", result[3], "\tBER FSK:", result[4])        

def line(letter, text):
    print(letter * 120)
    print('\t' * 6, text)
    print(letter * 120)


# Bity informacyjne
print('=' * 120)
info_binary = '10100101010'
print("bity informacyjne:\t", info_binary)

# Bity zakodowane za pomocą Hamminga 11-15
temp_info = ''
info_encoded = ''
for i in range(1, len(info_binary)+1):
    temp_info += info_binary[i-1]
    if i % 11 == 0:
        hamming = Hamming(temp_info, 15, 11)
        hamming.coding_hamming_15_11()
        temp_info = ''
        info_encoded += hamming.encoded
print("bity zakodowane:\t", info_encoded)


alfa_values = [0, 2]
beta_values = [0, 20]

# --------------------------------- Szum biały ------------------------------------
line('-', 'Szum bialy: ASK')
generate_modulation('noise_white', 'ASK', alfa_values, beta_values)
line('-', 'Szum bialy: PSK')
generate_modulation('noise_white', 'PSK', alfa_values, beta_values)
line('-', 'Szum bialy: FSK')
generate_modulation('noise_white', 'FSK', alfa_values, beta_values)
# --------------------------------- Szum  tłumienia ------------------------------------
line('-', 'Szum tlumienie: ASK')
generate_modulation('noise_suppression', 'ASK', alfa_values, beta_values)
line('-', 'Szum tlumienie: PSK')
generate_modulation('noise_suppression', 'PSK', alfa_values, beta_values)
line('-', 'Szum tlumienie: FSK')
generate_modulation('noise_suppression', 'FSK', alfa_values, beta_values)

# --------------------------------- Konfiguracja 1 ------------------------------------
line('-', 'Konfiguracja 1: ASK')
generate_modulation('configuration1', 'ASK', alfa_values, beta_values)
line('-', 'Konfiguracja 1: PSK')
generate_modulation('configuration1', 'PSK', alfa_values, beta_values)
line('-', 'Konfiguracja 1: FSK')
generate_modulation('configuration1', 'FSK', alfa_values, beta_values)
# --------------------------------- Konfiguracja 2 ------------------------------------
line('-', 'Konfiguracja 2: ASK')
generate_modulation('configuration2', 'ASK', alfa_values, beta_values)
line('-', 'Konfiguracja 2: PSK')
generate_modulation('configuration2', 'PSK', alfa_values, beta_values)
line('-', 'Konfiguracja 2: FSK')
generate_modulation('configuration2', 'FSK', alfa_values, beta_values)

print('=' * 120)
temp_info = ''
info_decoded = ''
for i in range(1, len(info_encoded)+1):
    temp_info += info_encoded[i-1]
    if i % 15 == 0:
        hamming = Hamming(None, 15, 11)
        hamming.decoding_hamming_15_11(temp_info)
        temp_info = ''
        info_decoded += hamming.decoded
print("bity zakodowane:\t", info_encoded)
print("bity zdekodowane:\t", info_decoded)
print('=' * 120)
print("bity zdekodowane są takie same jak bity informacyjne:\n\t",
      info_decoded == info_binary)
print('=' * 120)

