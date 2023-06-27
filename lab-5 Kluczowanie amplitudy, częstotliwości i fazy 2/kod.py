# ======================================================================================
import numpy as np
import matplotlib.pyplot as plt
# ======================================================================================
# ======================================================================================


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

bn = 'test'  # strumień bitowy
Tc = 1      # [s] czas trwania sygnału, (czas próbkowania w sekundach)
bits = ascii_to_bit(bn)
B = len(bits)       # liczba bitów sygnalu informatycznego
Tb = Tc / B         # czas trwania pojedynszego bitu Tc=>czas trwania sygnalu B=>ilośc bitów
W = 2               # przebieg czasowy
# Czestotliwosc nosna - fn
fn = W / Tb         # zaleznosci cżestotliwosci w przypadku kluczowania PSK
fn1 = (W + 1) / Tb  # zaleznosci cżestotliwosci w przypadku kluczowania FSK fn1
fn2 = (W + 2) / Tb  # zaleznosci cżestotliwosci w przypadku kluczowania FSK fn2

fs = 1988   # [Hz] częstotliwosć probkowania

N = int(Tc * fs)        # liczba próbek przypadających na cały sygnal
A1 = 5
A2 = 10
A = 10

Tbp = int(N/B)  # [probki]
h = 1200

# ======================================================================================
set_signal = ['ASK', 'PSK', 'FSK']
def create_signal(set_signal):
    x = []
    indeks = 0
    tempTbp = 1
    match set_signal:
        case 'ASK':
            for n in range(N):        
                x.append(ask_z(n/fs, bits[indeks]))
                if(tempTbp == Tbp and indeks < B-1):
                    indeks += 1
                    tempTbp = 0
                tempTbp += 1
        case 'PSK':
            for n in range(N):        
                x.append(psk_z(n/fs, bits[indeks]))
                if(tempTbp == Tbp and indeks < B-1):
                    indeks += 1
                    tempTbp = 0
                tempTbp += 1
        case 'FSK':
            for n in range(N):        
                x.append(fsk_z(n/fs, bits[indeks]))
                if(tempTbp == Tbp and indeks < B-1):
                    indeks += 1
                    tempTbp = 0
                tempTbp += 1      
    return x
#================================== ASK ===============================================
# Kluczowanie z przesuwem amplitudy (ASK)
def ask_z(t, b, A1=5, A2=10):
    # parametry amplitudy  przedział <A1, A2>
    if b == '0':
        return A1 * np.sin(2 * np.pi * fn * t)
    elif b == '1':
        return A2 * np.sin(2 * np.pi * fn * t)    



def ask_x():
    ask_x_ = []
    for n in range(N):
        ask_x_.append(sig_ask_z[n] * A*np.sin(2*np.pi*fn*(n/fs)))
    return ask_x_


def ask_p():
    ask_p_ = []
    indeks = 0
    for i in range(B):
        ask_p_.append(sig_ask_x[indeks])
        indeks += 1
        for j in range(Tbp-1):
            ask_p_.append(ask_p_[indeks-1]+sig_ask_x[indeks])
            indeks += 1
    return ask_p_


def ask_c():
    ask_c_ = []
    for i in range(N):
        if(sig_ask_p[i] > h):
            ask_c_.append(1)
        else:
            ask_c_.append(0)
    return ask_c_

sig_ask_z = create_signal('ASK')
sig_ask_x = ask_x()
sig_ask_p = ask_p()
sig_ask_c = ask_c()


#================================== PSK ===============================================
# Kluczowanie z przesuwem fazy (PSK)
def psk_z(t, b):
    if b == '0':
        return np.sin(2 * np.pi * fn * t)
    elif b == '1':
        return np.sin(2 * np.pi * fn * t + np.pi)

def psk_x():
    psk_x_ = []
    for n in range(N):
        psk_x_.append(sig_psk_z[n] * A * np.sin(2*np.pi*fn*(n/fs)))
    return psk_x_

def psk_p():
   psk_p_ = []
   indeks = 0
   for i in range(B):
       psk_p_.append(sig_psk_x[indeks])
       indeks +=1
       for j in range(Tbp-1):
           psk_p_.append(psk_p_[indeks-1] + sig_psk_x[indeks])
           indeks+=1
   return psk_p_

def psk_c():
    psk_c_ = []
    for i in range(N):
        if(sig_psk_p[i] < 0):
            psk_c_.append(1)
        else:
            psk_c_.append(0)
    return psk_c_

sig_psk_z = create_signal('PSK')
sig_psk_x = psk_x()
sig_psk_p = psk_p()
sig_psk_c = psk_c()

#================================== FSK ===============================================
# Kluczowanie z przesuwem czestotliwosci (FSK)
def fsk_z(t, b):
    if b == '0':
        return np.sin(2 * np.pi * fn1 * t)
    elif b == '1':
        return np.sin(2 * np.pi * fn2 * t)

def fsk_x(num):
    fsk_x_ = []    
    if(num==1):
        for n in range(N):
            fsk_x_.append(sig_fsk_z[n] * np.sin(2 * np.pi * fn1 * (n/fs)))
    elif(num==2):
        for n in range(N):
            fsk_x_.append(sig_fsk_z[n] * np.sin(2 * np.pi * fn2 * (n/fs)))
    return fsk_x_

def fsk_pn(sig):
    fsk_pn_ = []
    indeks = 0
    for i in range(B):
        fsk_pn_.append(sig[indeks])
        indeks +=1
        for j in range(Tbp-1):
            fsk_pn_.append(fsk_pn_[indeks-1]+sig[indeks])
            indeks+=1
    return fsk_pn_

def fsk_p():
    fsk_p_=[]
    for i in range(N):
        fsk_p_.append(sig_fsk_p2[i] - sig_fsk_p1[i])
    return fsk_p_

def fsk_c():
    fsk_c = []
    for i in range(N):
        if(sig_fsk_p[i]>0):
            fsk_c.append(1)
        else:
            fsk_c.append(0)
    return fsk_c


sig_fsk_z  = create_signal('FSK')
sig_fsk_x1 = fsk_x(1)
sig_fsk_x2 = fsk_x(2)
sig_fsk_p1 = fsk_pn(sig_fsk_x1)
sig_fsk_p2 = fsk_pn(sig_fsk_x2)
sig_fsk_p  = fsk_p()
sig_fsk_c  = fsk_c()

#======================================================================================
#   2. Wygenerować wykresy przedstawiające proces demodulacji dla rozpatrzywanych 
#   sygnałow zmodulowanych ASK/PSK (z(t), x(t), p(t), c(t)) oraz 
#   FSK (z(t), x1(t), x2(t), p1(t), p2(t), c(t)).
#======================================================================================
#================================== ASK ===============================================
sig_ask = {
    'ASK Z': sig_ask_z,
    'ASK X': sig_ask_x,
    'ASK P': sig_ask_p,
    'ASK C': sig_ask_c
}
for key, value in sig_ask.items():
    plt.figure('Sygnał ' + key)
    plt.plot(value, color='red')
    plt.title('Sygnał ' + key)
    plt.xlabel('Czas [s]')
    plt.ylabel('Amplituda')
    plt.savefig('./' + key[:3].lower() + '_' + key[-1].lower() + '.png')
    plt.show()

#================================== PSK ===============================================

sig_psk = {
    'PSK Z': sig_psk_z,
    'PSK X': sig_psk_x,
    'PSK P': sig_psk_p,
    'PSK C': sig_psk_c
}
for key, value in sig_psk.items():
    plt.figure('Sygnał ' + key)
    plt.plot(value, color='green')
    plt.title('Sygnał ' + key)
    plt.xlabel('Czas [s]')
    plt.ylabel('Amplituda')
    plt.savefig('./' + key[:3].lower() + '_' + key[-1].lower() + '.png')
    plt.show()
    
#================================== FSK ===============================================

sig_fsk = {
    'FSK Z': sig_fsk_z,
    'FSK X1': sig_fsk_x1,
    'FSK X2': sig_fsk_x2,
    'FSK P1': sig_fsk_p1,
    'FSK P2': sig_fsk_p2,
    'FSK P': sig_fsk_p,
    'FSK C': sig_fsk_c
}
for key, value in sig_fsk.items():
    plt.figure('Sygnał ' + key)
    plt.plot(value, color='orange')
    plt.title('Sygnał ' + key)
    plt.xlabel('Czas [s]')
    plt.ylabel('Amplituda')
    plt.savefig('./' + key[:3].lower() + '_' + key[3:].lower() + '.png')
    plt.show()
    
#======================================================================================
#   3. Napisać funkcję zamieniającą sygnał c(t) na ciąg bitów
#======================================================================================
def gen_bits_string(signal_c):
    signal_c_ = ''
    sum = 0
    sum += signal_c[0]
    for i in range(1,N):
        sum += signal_c[i]
        if(i % Tbp==0 or i == N-1):
            if(sum >= Tbp // 2):
                signal_c_ += '1'
            else:
                signal_c_ += '0'
            sum = 0
    return signal_c_

def check_bits_string(new_bits):
   if(bits == new_bits):
       print('Nowe bity zgadzaja sie z poprzednimy bitamy')
   else:
       print('Blad')
       
       
new_bits_ask  = gen_bits_string(sig_ask_c)
new_bits_psk  = gen_bits_string(sig_psk_c)
new_bits_fsk  = gen_bits_string(sig_fsk_c)
check_bits_string(new_bits_ask)
check_bits_string(new_bits_psk)
check_bits_string(new_bits_fsk)

