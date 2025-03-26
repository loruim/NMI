import cv2 as cv
import numpy as np
import copy
import bitstring
from another_function import EC, PSNR, BER, MSE

def Interpolation(height, width, pixel_array):
    interpolated_array = np.zeros((2*height, 2*width), dtype=np.int16)
    for column in range(len(pixel_array)):
        for string in range(len(pixel_array[column])):
            interpolated_array[2 * column][2 * string] = pixel_array[column][string]
            try:
                interpolated_array[2 * column + 1][2 * string] = (pixel_array[column][string] + pixel_array[column + 1][string]) // 2
            except IndexError:
                interpolated_array[2 * column + 1][2 * string] = (pixel_array[column][string] + interpolated_array[2 * column - 1][2 * string]) // 2
            try:
                interpolated_array[2 * column][2 * string + 1] = (pixel_array[column][string] + pixel_array[column][string + 1]) // 2
            except IndexError:
                interpolated_array[2 * column][2 * string + 1] = (pixel_array[column][string] + interpolated_array[2 * column][2 * string - 1]) // 2
            interpolated_array[2 * column + 1][2 * string + 1] = (interpolated_array[2 * column + 1][2 * string] + interpolated_array[2 * column][2 * string] + interpolated_array[2 * column][2 * string + 1]) // 3

    return interpolated_array

def Embedding(interpolated_array, pixel_array, bit_stream):
    end_flag = False
    for column in range(len(pixel_array)):
        for string in range(len(pixel_array)):
            difference = abs(interpolated_array[2 * column][2 * string + 1] - interpolated_array[2 * column][2 * string])
            if difference != 0:
                quantity_of_bits = int(np.floor(np.log2(difference)))
                if quantity_of_bits != 0:
                    if bit_stream.pos + quantity_of_bits <= bit_stream.length:
                        bits = bit_stream.read(f'bits:{quantity_of_bits}').bin
                    else:
                        bits = bit_stream.read(f'bits:{bit_stream.length - bit_stream.pos}').bin
                    embedded_bits = int(bits, base=2)
                    interpolated_array[2 * column][2 * string + 1] += embedded_bits

                    if bit_stream.pos == bit_stream.length:
                        end_flag = True
                        break

            difference = abs(interpolated_array[2 * column + 1][2 * string] - interpolated_array[2 * column][2 * string])
            if difference != 0:
                quantity_of_bits = int(np.floor(np.log2(difference)))
                if quantity_of_bits != 0:
                    if bit_stream.pos + quantity_of_bits <= bit_stream.length:
                        bits = bit_stream.read(f'bits:{quantity_of_bits}').bin
                    else:
                        bits = bit_stream.read(f'bits:{bit_stream.length - bit_stream.pos}').bin
                    embedded_bits = int(bits, base=2)
                    interpolated_array[2 * column + 1][2 * string] += embedded_bits

                    if bit_stream.pos == bit_stream.length:
                        end_flag = True
                        break

            difference = abs(interpolated_array[2 * column + 1][2 * string + 1] - interpolated_array[2 * column][2 * string])
            if difference != 0:
                quantity_of_bits = int(np.floor(np.log2(difference)))
                if quantity_of_bits != 0:
                    if bit_stream.pos + quantity_of_bits <= bit_stream.length:
                        bits = bit_stream.read(f'bits:{quantity_of_bits}').bin
                    else:
                        bits = bit_stream.read(f'bits:{bit_stream.length - bit_stream.pos}').bin
                    embedded_bits = int(bits, base=2)
                    interpolated_array[2 * column + 1][2 * string + 1] += embedded_bits

                    if bit_stream.pos == bit_stream.length:
                        end_flag = True
                        break

        if end_flag:
            break

def extraction(original_pixels, encrypted_pixels, pixel_data):
    key_string = ''
    for column in range(len(pixel_data)):
        for string in range(len(pixel_data)):
            difference = abs(original_pixels[2 * column][2 * string + 1] - original_pixels[2 * column][2 * string])
            if difference != 0:
                quantity_of_bits = int(np.floor(np.log2(difference)))
                if quantity_of_bits != 0:
                    number = abs(encrypted_pixels[2 * column][2 * string + 1] - original_pixels[2 * column][2 * string + 1])
                    bin_number = bin(number)[2:]
                    if len(bin_number) < quantity_of_bits:
                        bin_number = "0" * (quantity_of_bits - len(bin_number)) + bin_number
                    key_string += bin_number

            difference = abs(original_pixels[2 * column + 1][2 * string] - original_pixels[2 * column][2 * string])
            if difference != 0:
                quantity_of_bits = int(np.floor(np.log2(difference)))
                if quantity_of_bits != 0:
                    number = abs(encrypted_pixels[2 * column + 1][2 * string] - original_pixels[2 * column + 1][2 * string])
                    bin_number = bin(number)[2:]
                    if len(bin_number) < quantity_of_bits:
                        bin_number = "0" * (quantity_of_bits - len(bin_number)) + bin_number
                    key_string += bin_number

            difference = abs(original_pixels[2 * column + 1][2 * string + 1] - original_pixels[2 * column][2 * string])
            if difference != 0:
                quantity_of_bits = int(np.floor(np.log2(difference)))
                if quantity_of_bits != 0:
                    number = abs(encrypted_pixels[2 * column + 1][2 * string + 1] - original_pixels[2 * column + 1][2 * string + 1])
                    bin_number = bin(number)[2:]
                    if len(bin_number) < quantity_of_bits:
                        bin_number = "0" * (quantity_of_bits - len(bin_number)) + bin_number
                    key_string += bin_number

    bit_stream, file, key_length = key()
    count = 0
    for i in range(key_length):
        try:
            if bit_stream.read(f'bits:{1}').bin != key_string[i]:
                count += 1
        except IndexError:
            break
        
    print(f"BER = {BER(len(key_string), count)}")

    hex_form = ""
    while len(key_string) != 0:
            hex_form += hex(int(key_string[:4], 2))[2:]
            key_string = key_string[4:]

    if len(hex_form) % 2 != 0:
        hex_form += "0"

    with open("output.txt", "wb") as f:
        f.write(bytes.fromhex(hex_form))

def key():
    f = open("Key/key.txt", "rb")
    bit_stream = bitstring.ConstBitStream(f)
    key_length = bit_stream.length
    return bit_stream, f, key_length

def Encryption():
# 1. Загрузка изображения
# Picture/Original/airplane.png
# Picture/Original/baboon.png
# Picture/Original/Lenna.png
    image = cv.imread("Picture/Original/Lenna.png")

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    height, width = gray.shape
    pixel_data = np.array(gray, dtype=np.int16)

    interpolated_image = Interpolation(height, width, pixel_data)
    interpolation = copy.deepcopy(interpolated_image)
    bit_stream, file, key_length = key()
    Embedding(interpolated_image, pixel_data, bit_stream)
    interpolated_image = interpolated_image.astype(np.uint8)

    print(f"EC = {EC(bit_stream.pos, width, height)}")
    print(f"MSE = {MSE(interpolation, interpolated_image)}")
    print(f"PSNR = {PSNR(interpolation, interpolated_image)}")

    cv.imwrite("Picture/Changed/modified_Lenna.png", interpolated_image)
    cv.imwrite("modified_Lenna.jpg", interpolated_image, [cv.IMWRITE_JPEG_QUALITY, 90])
    cv.imshow("Original", gray)
    cv.imshow("Modified", interpolated_image)
    cv.waitKey(0)
    cv.destroyAllWindows()

def Decryption():
# 2. Построение измененного изображения
# Picture/Changed/modified_airplane.png
# Picture/Changed/modified_baboon.png
# Picture/Changed/modified_Lenna.png
    image_original = cv.imread("Picture/Original/Lenna.png")
    gray = cv.cvtColor(image_original, cv.COLOR_BGR2GRAY)
    height, width = gray.shape
    pixel_data = np.array(gray, dtype=np.int16)
    interpolated_image = Interpolation(height, width, pixel_data)

    #image_code = cv.imread("Picture/Changed/modified_Lenna.png", cv.IMREAD_GRAYSCALE)
    image_code = cv.imread("modified_Lenna.jpg", cv.IMREAD_GRAYSCALE)
    encrypted_pixels = np.array(image_code, dtype=np.int16)
    
    extraction(interpolated_image, encrypted_pixels, pixel_data)

chouse = input('1 / 2: ')
if chouse == '1':
    Encryption()
elif chouse == '2':
    Decryption()
