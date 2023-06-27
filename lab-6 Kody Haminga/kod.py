import numpy as np
# ===========================================================================================
#   1.  Zaimplementować koder oraz dekoder kodu Haminga (7, 4). Z wykorzystaniem 
#   opracowanych implementacja nałeży zilustrować proces kodowania i dekodowania.
# ===========================================================================================
def array_to_string(array):
     return ''.join(str(i) for i in array)

# Funkcje dla kodowania Hamminga(7, 4)
def encode_hamming_74(bits):
    x3=int(bits[0])
    x5=int(bits[1])
    x6=int(bits[2])
    x7=int(bits[3])
    
    x1 = x3 ^ x5 ^ x7
    x2 = x3 ^ x6 ^ x7
    x4 = x5 ^ x6 ^ x7
    return  np.array([x1, x2, x3, x4, x5, x6, x7])

# Funkcje dla dekodowania Hamminga(7, 4)
def decode_hamming_74(encoded_bits):
    encoded = np.array([int(i) for i in encoded_bits])
    
    p1 = encoded[0]
    p2 = encoded[1]
    x3 = encoded[2]
    p4 = encoded[3]
    x5 = encoded[4]
    x6 = encoded[5]
    x7 = encoded[6]
    
    c1 = x3 ^ x5 ^ x7
    c2 = x3 ^ x6 ^ x7
    c4 = x5 ^ x6 ^ x7
    
    if (c1, c2, c4) != (p1, p2, p4):
        error = 4*c4 + 2*c2 + c1
        encoded[error - 1] ^= 1
    return  np.array([encoded[2], encoded[4], encoded[5], encoded[6]])

# ===========================================================================================
#   2.  Zaimplementować koder oraz dekoder kodu Hamming(15, 11) z wykorzystaniem rachunku 
#   macierzowego z operacją dodawania modulo 2.
# ===========================================================================================
def generate_G():
    # P jest macierzą okreslajaca bity parzystosci
    P = np.random.randint(2, size=(4, 11))
    # I jest macirzą jednostkową
    I = np.eye(11, dtype=int)
    # G jest macierzą generującą
    G = np.hstack((I, P.T))
    return G

def generate_H(G):
    # P jest macierzą okreslajaca bity parzystosci
    P = G[:, -4:].T  # Bierzemy ostatnie 4 kolumny macierzy G
    # I jest macirzą jednostkową
    I = np.eye(4, dtype=int)
    # H jest macierzą kontroli parzystosci
    H = np.hstack((P, I))
    return H


def encode_hamming_1511(bits, gen_matrix):
    return np.dot(bits, gen_matrix) % 2


def decode_hamming_1511(encoded_bits, gen_matrix, check_matrix):
    encoded = np.array([int(i) for i in encoded_bits])
    syndrome = np.dot(encoded, check_matrix.T) % 2
    if np.any(syndrome):  
        error = np.where(np.all(check_matrix.T == syndrome, axis=1))[0][0]
        encoded[error] ^= 1  
    return encoded[:11]


# Testowanie funkcji
def main():
    # Generowanie losowych bitów
    info_bits_7_4 = np.random.randint(2, size=4)
    info_bits_15_11 = np.random.randint(2, size=11)
    
    # kodowania i dekodowania Hamming 7_4
    encode_7_4 = encode_hamming_74(info_bits_7_4)    
    decoded_7_4 = decode_hamming_74(encode_7_4)
   
    # kodowania i dekodowania Hamming 15_11
    gen_matrix = generate_G()
    check_matrix = generate_H(gen_matrix)
    encode_15_11 = encode_hamming_1511(info_bits_15_11, gen_matrix)    
    decoded_15_11 = decode_hamming_1511(encode_15_11, gen_matrix, check_matrix)
    
    print('--------------- Hamming 7_4 -------------------')
    print('Bity informacyjne:\t', array_to_string(info_bits_7_4))
    print('Bity zakodowane:\t', array_to_string(encode_7_4))
    print('Bity dekodowane:\t', array_to_string(decoded_7_4))
    print('===============================================')

    print('--------------- Hamming 15_11 -----------------')
    print('Bity informacyjne:\t', array_to_string(info_bits_15_11))
    print('Bity zakodowane:\t', array_to_string(encode_15_11))
    print('Bity dekodowane:\t', array_to_string(decoded_15_11))
    print('===============================================')
    

    
    
if __name__ == "__main__":
    main()