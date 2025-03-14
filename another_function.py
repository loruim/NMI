import numpy as np

def EC(message_bit_length, width, height):
    ec = message_bit_length / (width * height)
    return ec

def PSNR(original_image, stego_image):
    mse = MSE(original_image, stego_image)
    try:
        psnr = 10 * np.log10((255*2)/mse)
    except ZeroDivisionError:
        psnr = 100
    return psnr

def MSE(original_image, stego_image):
    height, width = original_image.shape
    first_part = 1 / (height * width)
    summ = 0
    for string in range(height):
        for column in range(width):
            summ += (original_image[string][column] - stego_image[string][column])**2
            
    mse = summ * first_part
    return mse

def BER(message_bit_length, error_count):
    ber = error_count / message_bit_length
    return ber