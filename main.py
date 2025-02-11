import cv2 as cv
import numpy as np

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

def Embedding(interpolated_array, pixel_array, byte_string):
    end_flag = False
    for column in range(len(pixel_array)):
        for string in range(len(pixel_array)):
            difference = abs(interpolated_array[2 * column][2 * string + 1] - interpolated_array[2 * column][2 * string])
            if difference != 0:
                quantity_of_bits = int(np.floor(np.log2(difference)))
                if quantity_of_bits != 0:
                    embedded_bits = int(byte_string[:quantity_of_bits], base=2)
                    byte_string = byte_string[quantity_of_bits:]
                    interpolated_array[2 * column][2 * string + 1] += embedded_bits

                    if len(byte_string) == 0:
                        end_flag = True
                        break

            difference = abs(interpolated_array[2 * column + 1][2 * string] - interpolated_array[2 * column][2 * string])
            if difference != 0:
                quantity_of_bits = int(np.floor(np.log2(difference)))
                if quantity_of_bits != 0:
                    embedded_bits = int(byte_string[:quantity_of_bits], base=2)
                    byte_string = byte_string[quantity_of_bits:]
                    interpolated_array[2 * column + 1][2 * string] += embedded_bits

                    if len(byte_string) == 0:
                        end_flag = True
                        break

            difference = abs(interpolated_array[2 * column + 1][2 * string + 1] - interpolated_array[2 * column][2 * string])
            if difference != 0:
                quantity_of_bits = int(np.floor(np.log2(difference)))
                if quantity_of_bits != 0:
                    embedded_bits = int(byte_string[:quantity_of_bits], base=2)
                    byte_string = byte_string[quantity_of_bits:]
                    interpolated_array[2 * column + 1][2 * string + 1] += embedded_bits

                    if len(byte_string) == 0:
                        end_flag = True
                        break

        if end_flag:
            break

"""def extraction(original_pixels, encrypted_pixels, pixel_data):
    key_string = ''
    for column in range(len(pixel_data)):
        for string in range(len(pixel_data)):
            number = abs(encrypted_pixels[2 * column][2 * string + 1] - original_pixels[2 * column][2 * string + 1])
            bin_number = format(number, 'b')
            key_string += bin_number

            number = abs(encrypted_pixels[2 * column + 1][2 * string] - original_pixels[2 * column + 1][2 * string])
            bin_number = format(number, 'b')
            key_string += bin_number

            number = abs(encrypted_pixels[2 * column + 1][2 * string + 1] - original_pixels[2 * column + 1][2 * string + 1])
            bin_number = format(number, 'b')
            key_string += bin_number

    text = "".join(chr(int(key_string[i:i+8], 2)) for i in range(0, len(key_string), 8))
    
    with open("output.txt", "w", encoding="utf-8") as f:
        f.write(text)

    print(text)"""

def key():
    with open("Key/number_key.txt", "r", encoding="utf8") as f:
        data = f.read()
    byte_string = "".join(format(ord(char), "08b") for char in data)
    return byte_string

def Encryption():
# 1. Загрузка изображения
# Picture/Original/airplane.png
# Picture/Original/baboon.png
# Picture/Original/Lenna.png
    image = cv.imread("Picture/Original/baboon.png")

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    height, width = gray.shape
    pixel_data = np.array(gray, dtype=np.int16)

    interpolated_image = Interpolation(height, width, pixel_data)
    byte_string = key()
    Embedding(interpolated_image, pixel_data, byte_string)
    interpolated_image = interpolated_image.astype(np.uint8)

    cv.imwrite("Picture/Changed/modified_baboon.png", interpolated_image)
    cv.imshow("Original", gray)
    cv.imshow("Modified", interpolated_image)
    cv.waitKey(0)
    cv.destroyAllWindows()

# Not work yet
"""def Decryption():
# 2. Построение измененного изображения
# Picture/Changed/modified_airplane.png
# Picture/Changed/modified_baboon.png
# Picture/Changed/modified_Lenna.png
    image_original = cv.imread("Picture/Original/baboon.png")
    gray = cv.cvtColor(image_original, cv.COLOR_BGR2GRAY)
    height, width = gray.shape
    pixel_data = np.array(gray, dtype=np.int16)
    interpolated_image = Interpolation(height, width, pixel_data)

    image_code = cv.imread("Picture/Changed/modified_baboon.png", cv.IMREAD_GRAYSCALE)
    encrypted_pixels = np.array(image_code, dtype=np.int16)
    
    extraction(interpolated_image, encrypted_pixels, pixel_data)"""

chouse = input('1 / 2: ')
if chouse == '1':
    Encryption()
#elif chouse == '2':
#    Decryption()
