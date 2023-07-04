import numpy as np
import matplotlib.pyplot as plt

#================================================================================================================================
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
#================================================================================================================================

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

class Modulacja:
    def __init__(self, bn):
        self.fs = 1988   # [Hz] częstotliwość próbkowania
        self.bn = bn
        self.bits = len(self.bn)
        self.noweBn = None 
        self.B = len(self.bn)
        self.Tc = 1                     # [s] czas trwania sygnału (czas próbkowania w sekundach)
        self.N = int(self.Tc*self.fs)   # liczba próbek przypadających na cały sygnał
        self.Tb = self.Tc/self.B        # czas trwania pojedynczego bitu Tc => czas trwania sygnału B => liczba bitów
        self.Tbp = int(self.N/self.B)   # [probki]
        self.W = 2                      # przebieg czasowy
        self.A = 10
        self.A1 = 5
        self.A2 = 10  
        # Częstotliwość nośna - fn
        self.fn = self.W / self.Tb         # zależność częstotliwości w przypadku kluczowania PSK
        self.fn1 = (self.W + 1) / self.Tb  # zależność częstotliwości w przypadku kluczowania FSK fn1
        self.fn2 = (self.W + 2) / self.Tb  # zależność częstotliwości w przypadku kluczowania FSK fn2
        self.h = 1200

        self.sig_ask_z = self.create_signal('ASK')
        self.sig_ask_x = self.ask_x()
        self.sig_ask_p = self.ask_p()
        self.sig_ask_c = self.ask_c()  

        self.sig_psk_z = self.create_signal('PSK')
        self.sig_psk_x = self.psk_x()
        self.sig_psk_p = self.psk_p()
        self.sig_psk_c = self.psk_c()
        
        self.sig_fsk_z  = self.create_signal('FSK')
        self.sig_fsk_x1 = self.fsk_x(1)
        self.sig_fsk_x2 = self.fsk_x(2)
        self.sig_fsk_p1 = self.fsk_pn(self.sig_fsk_x1)
        self.sig_fsk_p2 = self.fsk_pn(self.sig_fsk_x2)
        self.sig_fsk_p  = self.fsk_p()
        self.sig_fsk_c  = self.fsk_c()

        self.sig_noisy_ask_c_1 = None
        self.sig_noisy_psk_c_1 = None
        self.sig_noisy_fsk_c_1 = None
        self.sig_noisy_ask_c_2 = None
        self.sig_noisy_psk_c_2 = None
        self.sig_noisy_fsk_c_2 = None
      
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

    #================================= ASK ===============================================
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
            ask_x_.append(self.sig_ask_z[n] * self.A * np.sin(2*np.pi*self.fn*(n/self.fs)))
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

    #================================== PSK ===============================================
    # Kluczowanie z przesunięciem fazy (PSK)
    def psk_z(self, t, b):
        if b == '0':
            return np.sin(2 * np.pi * self.fn * t)
        elif b == '1':
            return np.sin(2 * np.pi * self.fn * t + np.pi)

    def psk_x(self):
        psk_x_ = []
        for n in range(self.N):
            psk_x_.append(self.sig_psk_z[n] * self.A * np.sin(2*np.pi*self.fn*(n/self.fs)))
        return psk_x_
    
    def psk_p(self):
        psk_p_ = []
        index = 0
        for i in range(self.B):
            psk_p_.append(self.sig_psk_x[index])
            index +=1
            for j in range(self.Tbp-1):
                psk_p_.append(psk_p_[index-1] + self.sig_psk_x[index])
                index+=1
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

    #================================== FSK ===============================================
    # Kluczowanie z przesunięciem częstotliwości (FSK)
    def fsk_z(self, t, b):
        if b == '0':
            return np.sin(2 * np.pi * self.fn1 * t)
        elif b == '1':
            return np.sin(2 * np.pi * self.fn2 * t)

    def fsk_x(self, num):
        fsk_x_ = []    
        if(num==1):
            for n in range(self.N):
                fsk_x_.append(self.sig_fsk_z[n] * np.sin(2 * np.pi * self.fn1 * (n/self.fs)))
        elif(num==2):
            for n in range(self.N):
                fsk_x_.append(self.sig_fsk_z[n] * np.sin(2 * np.pi * self.fn2 * (n/self.fs)))
        return fsk_x_

    def fsk_pn(self, sig):
        fsk_pn_ = []
        index = 0
        for i in range(self.B):
            fsk_pn_.append(sig[index])
            index +=1
            for j in range(self.Tbp-1):
                fsk_pn_.append(fsk_pn_[index-1]+sig[index])
                index+=1
        return fsk_pn_

    def fsk_p(self):
        fsk_p_=[]
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
        errors = np.sum(np.abs(np.array(original_signal) - np.array(received_signal)))
        ber = errors / len(original_signal)
        return ber

    def simulate_cascade_configuration(self, alfa, beta):
        self.sig_noisy_ask_c_1 = self.cascade_connection(self.sig_ask_c, self.sig_psk_c, alfa, beta)
        self.sig_noisy_psk_c_1 = self.cascade_connection(self.sig_psk_c, self.sig_fsk_c, alfa, beta)
        self.sig_noisy_fsk_c_1 = self.cascade_connection(self.sig_fsk_c, self.sig_ask_c, alfa, beta)

        self.sig_noisy_ask_c_2 = self.cascade_connection(self.sig_psk_c, self.sig_ask_c, alfa, beta)
        self.sig_noisy_psk_c_2 = self.cascade_connection(self.sig_fsk_c, self.sig_psk_c, alfa, beta)
        self.sig_noisy_fsk_c_2 = self.cascade_connection(self.sig_ask_c, self.sig_fsk_c, alfa, beta)

        ber_ask_c_1 = self.calculate_ber(self.sig_ask_c, self.sig_noisy_ask_c_1)
        ber_psk_c_1 = self.calculate_ber(self.sig_psk_c, self.sig_noisy_psk_c_1)
        ber_fsk_c_1 = self.calculate_ber(self.sig_fsk_c, self.sig_noisy_fsk_c_1)

        ber_ask_c_2 = self.calculate_ber(self.sig_ask_c, self.sig_noisy_ask_c_2)
        ber_psk_c_2 = self.calculate_ber(self.sig_psk_c, self.sig_noisy_psk_c_2)
        ber_fsk_c_2 = self.calculate_ber(self.sig_fsk_c, self.sig_noisy_fsk_c_2)

        return ber_ask_c_1, ber_psk_c_1, ber_fsk_c_1, ber_ask_c_2, ber_psk_c_2, ber_fsk_c_2

print('=====================================\n')
info_binary = '10100101010' 
print("bity informacyjne:\t",info_binary)
temp_info = ''
info_encoded = ''
for i in range(1,len(info_binary)+1):
    temp_info += info_binary[i-1]
    if i%11 == 0:
        hamming = Hamming(temp_info, 15, 11)
        hamming.coding_hamming_15_11()
        temp_info = ''
        info_encoded += hamming.encoded
print("bity zakodowane:\t",info_encoded)
temp_info = ''
info_decoded = ''
for i in range(1,len(info_encoded)+1):
    temp_info += info_encoded[i-1]
    if i%15 == 0:
        hamming = Hamming(None, 15, 11)
        hamming.decoding_hamming_15_11(temp_info)
        temp_info = ''
        info_decoded += hamming.decoded
print("bity zdekodowane:\t",info_decoded)
print("bity zdekodowane są takie same jak bity informacyjne:\n\t", info_decoded == info_binary)
print('=====================================\n')

modulation = Modulacja(info_binary)
alfa_values = [0.1, 0.5, 1.0]
beta_values = [0.1, 0.5, 1.0]
ber_results_1_2 = []
ber_results_2_1 = []

for alfa in alfa_values:
    for beta in beta_values:
        ber_ask_c_1, ber_psk_c_1, ber_fsk_c_1, ber_ask_c_2, ber_psk_c_2, ber_fsk_c_2 = modulation.simulate_cascade_configuration(alfa, beta)
        ber_results_1_2.append([alfa, beta, ber_ask_c_1, ber_psk_c_1, ber_fsk_c_1])
        ber_results_2_1.append([alfa, beta, ber_ask_c_2, ber_psk_c_2, ber_fsk_c_2])

# Wyświetlanie wyników
print("Zależność współczynnika BER od parametrów alfa i beta (1+2):")
for result in ber_results_1_2:
    print("Alfa:", result[0], "Beta:", result[1], "\tBER ASK:", result[2], "\tBER PSK:", result[3], "\tBER FSK:", result[4])

print("\nZależność współczynnika BER od parametrów alfa i beta (2+1):")
for result in ber_results_2_1:
    print("Alfa:", result[0], "Beta:", result[1], "\tBER ASK:", result[2], "\tBER PSK:", result[3], "\tBER FSK:", result[4])

print('=====================================\n')
