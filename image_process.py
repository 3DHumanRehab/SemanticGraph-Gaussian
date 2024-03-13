import cv2
import os
import numpy as np

def canny(img, low_threshold = 100,high_threshold = 200):
    dst = cv2.Canny(img, low_threshold, high_threshold)
    dst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
    return dst

def fft_filter(source_image, cut_freq_rate=0.1):
    h, w, c = source_image.shape
    print(h,w)
    out = []
    cut_h = h/2*cut_freq_rate
    cut_w = w/2*cut_freq_rate
    for i in range(c):
        source_image_f = np.fft.fft2(source_image[:,:,i])
        source_image_fshift = np.fft.fftshift(source_image_f)

        source_image_fshift[int(h/2)-int(cut_h):int(h/2)+int(cut_h),
                            int(w/2)-int(cut_w):int(w/2)+int(cut_w)]=0
            
        source_image_ifshift = np.fft.ifftshift(source_image_fshift)
        source_image_if = np.fft.ifft2(source_image_ifshift)
        source_image_if = np.abs(source_image_if)
        
        source_image_if[source_image_if>255] = np.max(source_image[:,:,i])
        out.append(source_image_if)
    out = np.array(out)
    out = out.swapaxes(1,0).swapaxes(1,2)

    out = out.astype(np.uint8)
    return out

def brightness_enhance(image,brightness_factor=3):
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    brightness = hsv_img[:, :, 2]
    enhanced_brightness = brightness * brightness_factor
    
    max_value = 255
    enhanced_brightness[enhanced_brightness > max_value] = max_value
    
    hsv_img[:, :, 2] = enhanced_brightness

    output_image = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
    return output_image

if __name__=='__main__':
    img = cv2.imread('image.png')
    
    cv2.imwrite("canny.png",canny(img))
    cv2.imwrite("fourier_HPF.png",brightness_enhance(fft_filter(img,0.1)))