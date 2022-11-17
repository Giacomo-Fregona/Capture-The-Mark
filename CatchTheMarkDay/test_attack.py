from attacks import *
import cv2

# Returns the indices of the area (of size area_size) of the array containing the lowest values
def get_lowest_area(arr, area_size):
    min_sum = float("+inf")
    row_idx, col_idx = 0, 0
    for row in range(arr.shape[0]-area_size):
        for col in range(arr.shape[1]-area_size):
            curr_sum =  np.sum(arr[row:row+area_size, col:col+area_size])
            if curr_sum < min_sum:
                row_idx, col_idx = row, col
                min_sum = curr_sum
    # return arr[row_idx:row_idx+area_size, col_idx:col_idx+area_size]
    return row_idx, col_idx

# Attacks an image once, takes the <size> block least affected area of the image and attacks again on it
def attack_least_affected(image, size=64):
    print("Applying JPEG compression...")
    attacked = jpeg_compression(image, 75)

    # We calculate the difference between the original image and the attacked one
    # diff = abs(image - attacked)
    diff = image - attacked

    # We use the previously calculated diff the get the indices of the area with the lowest values, which means the least altered one
    row, col = get_lowest_area(diff, size)
    print("The lowest", size, "pixel area altered by the attack starts at coordinates (", row, col,")")

    # Attacks on the area defined by the previous calculation
    area_to_attack = attacked[row:row+size, col:col+size]
    print("Applying JPEG compression on this area...")
    attacked_area = jpeg_compression(area_to_attack, 75)
    
    # Replacing the area in the attacked image by the newly attacked one
    attacked[row:row+size, col:col+size] = attacked_area
    print("Done. Attacked image is:\n", attacked)
    return attacked

# Attacks an area of an image based on the input indices (row, col) and a size
def attack_area(image, row, col, size):
    area_to_attack = image[row:row+size, col:col+size]
    attacked = jpeg_compression(area_to_attack, 75)
    image[row:row+size, col:col+size] = attacked
    return image

image = cv2.imread('lena.bmp', 0)
attack_area(image, 0, 0, 20)
