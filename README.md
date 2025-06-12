# XuLyAnhSo
 btth-207ct65680-VuDoanKhanhLinh

 lab1

    
    iio.imread : đọc ảnh
    ex: data = iio.imread('AnhLab1.jpg')

    iio.imshow : xem ảnh
    ex: plt.imshow(red, cmap='Reds')  -> lưu ảnh blue với bảng màu Blues

    iio.imwrite : lưu ảnh
    ex:iio.imwrite('AnhLab1-bai1-red.jpg', red)

    plt.tight_layout(): căn lề ảnh

bai1
    tách từng màu:
    red = data[:, :, 0]     
    green = data[:, :, 1]
    blue = data[:, :, 2]

bai2
    hoán đổi màu:
    swapped[:, :, 0] = rgb[:, :, 1]   Red <- Green
    swapped[:, :, 1] = rgb[:, :, 2]   Green <- Blue
    swapped[:, :, 2] = rgb[:, :, 0]   Blue <- Red

bai3
    tạo 3 mảng rỗng h,s,v:
    height, width, _ = rgb.shape
    h = np.zeros((height, width))
    s = np.zeros((height, width))
    v = np.zeros((height, width))

    tạo vòng lặp duyệt qua từng pixel:
    for i in range(height):
        for j in range(width):
            r, g, b = rgb[i, j]
            h[i, j], s[i, j], v[i, j] = colorsys.rgb_to_hsv(r, g, b)

    chuyển sang định dạng unint8 để lưu ảnh
    h_img = (h * 255).astype(np.uint8)
    s_img = (s * 255).astype(np.uint8)
    v_img = (v * 255).astype(np.uint8)

    hiển thị với colormap 
    ex: plt.imshow(h_img, cmap='hsv') -> colormap 7 màu cầu vồng

bai4
    tạo mảng hsv rỗng:
    hsv = np.zeros_like(rgb)

    tạo vòng lặp duyệt qua từng pixel:
    for i in range(height):
        for j in range(width):
            r, g, b = rgb[i, j]
            h[i, j], s[i, j], v[i, j] = colorsys.rgb_to_hsv(r, g, b)

    thay đổi giá trị h và v:
    h = (h / 3.0) % 1.0           # Chia hue cho 3
    v = min(v * 0.75, 1.0)        # Giảm độ sáng 3/4

    Chuyển ngược HSV -> RGB
    new_rgb = np.zeros_like(rgb)
    for i in range(height):
        for j in range(width):
            h, s, v = hsv[i, j]
            r, g, b = colorsys.hsv_to_rgb(h, s, v)
            new_rgb[i, j] = [r, g, b]
    
bai5
    tạo vòng lặp duyệt tất cả file ảnh
    for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp')):

    Áp dụng mean filter
        if img.ndim == 3:   ->nếu là ảnh màumàu
            filtered_img = np.zeros_like(img)
            for c in range(3):  ->Lọc từng kênh R, G, B
                filtered_img[:, :, c] = uniform_filter(img[:, :, c], size=3)
        else:  
            filtered_img = uniform_filter(img, size=3) -> nếu là ảnh xám


lab2


bai6
    đọc file Exercise xử lý ảnh trong file
    input_folder = 'Exercise'
    output_folder = 'Exercise_Output'
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp')):
            filepath = os.path.join(input_folder, filename)
            img = iio.imread(filepath)

    thêm filter max,min,median
    median_filtered = median_filter(noisy_image, size=3)
    max_filtered = maximum_filter(noisy_image, size=3)
    min_filtered = minimum_filter(noisy_image, size=3)

    lặp qua 4 ảnh và hiển thị
    for i in range(4):
        axs[i].imshow(images[i])
        axs[i].set_title(titles[i])
        axs[i].axis('off')

bai7
    thêm filter cho ảnh
    sobel = filters.sobel(img2).astype(np.uint8)
    prewitt = filters.prewitt(img2).astype(np.uint8)
    canny = feature.canny(img2, sigma=3).astype(np.uint8)
    laplace = sn.laplace(img2,mode='reflect').astype(np.uint8)

    xem và lưu từng ảnh với từng filter khác nhau
    plt.subplot(1, 4, 1)
    iio.imsave('Exercise_Output/anh-sobel.jpg', sobel)
    plt.title('sobel')
    plt.imshow(sobel) 

bai8
    tạo hàm random để chọn màu ngẫu nhiênnhiên
    random = [0,1,2]
    np.random.shuffle(random)

    tạo ảnh random màu
    random_rgb = rgb[:,:,random]

bai9
    chuyển ảnh RGB sang HSV
    hue_shift = np.random.uniform(-0.1, 0.1)
    sat_scale = np.random.uniform(0.5, 1.5)
    val_scale = np.random.uniform(0.5, 1.5)

    tạo hàm random các giá trị
    hsv_image[:, :, 0] = (hsv_image[:, :, 0] + hue_shift) % 1.0
    hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] * sat_scale, 0, 1)
    hsv_image[:, :, 2] = np.clip(hsv_image[:, :, 2] * val_scale, 0, 1)



lab3
    
bai1
    tạo các hàm xử lý ảnh

    xử lý đảo màu
    def inverse_image(img):
    return 255 - img


    Thay đổi chất lượnglượng
    def gamma_correction(img, gamma=2.2):
        img_float = img_as_float(img)
        corrected = np.power(img_float, 1 / gamma)
        return img_as_ubyte(corrected)

    tahy đổi cường độ điểm ảnhảnh
    def log_transform(img):
        img_float = img_as_float(img)
        log_img = np.log1p(img_float)
        log_img /= np.max(log_img)
        return img_as_ubyte(log_img)

    cải thiện độ tương phản (mạnh)
    def histogram_equalization(img):
        if len(img.shape) == 3:
            img = np.mean(img, axis=2).astype(np.uint8)  # Chuyển về ảnh xám
        return img_as_ubyte(exposure.equalize_hist(img))

    cải thiện độ tương phản (yếu)
    def contrast_stretching(img):
        p2, p98 = np.percentile(img, (2, 98))
        stretched = rescale_intensity(img, in_range=(p2, p98))
        return stretched

    tạo các lựa chọn để xử lý ảnh
    print("Chọn phương pháp biến đổi ảnh:")
    print("I - Image Inverse")
    print("G - Gamma Correction")
    print("L - Log Transformation")
    print("H - Histogram Equalization")
    print("C - Contrast Stretching")

    if choice == 'I':
            result = inverse_image(img)
        elif choice == 'G':
            result = gamma_correction(img)
        elif choice == 'L':
            result = log_transform(img)
        elif choice == 'H':
            result = histogram_equalization(img)
        elif choice == 'C':
            result = contrast_stretching(img)
        else:
            print("❌ Lựa chọn không hợp lệ.")
            break

    lưu ảnh trong file output
    save_name = f"{os.path.splitext(filename)[0]}_{choice}.jpg"
        iio.imwrite(os.path.join(output_folder, save_name), result)
        print(f"✅ Đã lưu: {save_name}")

bai2
    tạo Fourier
    def apply_fft(img):
        gray = to_gray(img)
        f = fft2(gray)
        fshift = fftshift(f)
        magnitude_spectrum = np.log(1 + np.abs(fshift))
        return magnitude_spectrum

   tạo filter Butterworth Low/High Pass
    def butterworth_filter(shape, cutoff, order=2, filter_type='low'):
        M, N = shape
        u = np.arange(M)
        v = np.arange(N)
        U, V = np.meshgrid(v - N//2, u - M//2)
        D = np.sqrt(U**2 + V**2)

        if filter_type == 'low':
            H = 1 / (1 + (D / cutoff)**(2 * order))
        else:
            H = 1 / (1 + (cutoff / D)**(2 * order))
        return H
    
    thêm filter Butterworth Filter vào ảnh
    def apply_butterworth(img, cutoff=30, filter_type='low'):
        gray = to_gray(img)
        f = fft2(gray)
        fshift = fftshift(f)
        H = butterworth_filter(gray.shape, cutoff, filter_type=filter_type)
        f_filtered = fshift * H
        img_back = np.abs(ifft2(ifftshift(f_filtered)))
        return np.clip(img_back, 0, 255).astype(np.uint8)

bai3

    tạo hàm đổi màu RGB ngẫu nhiên
    def random_rgb_swap(img):
        channels = [0, 1, 2]
        random.shuffle(channels)
        swapped = img[:, :, channels]
        return swapped, channels

    các lựa chọn để biến đổi
    transformations = {
        'I': inverse_image,
        'G': gamma_correction,
        'L': log_transform,
        'H': histogram_equalization,
        'C': contrast_stretching
    }

    lựa chọn biến đổi ngẫu nhiên
    transform_key = random.choice(list(transformations.keys()))
    transformed = transformations[transform_key](img_swapped)

bai4

    như bai3
    thêm bộ lọc Butterworth Filter
    def butterworth_filter(img_gray, d0=30, n=2, highpass=False):
        img_float = img_as_float(img_gray)
        f = np.fft.fft2(img_float)
        fshift = np.fft.fftshift(f)

        rows, cols = img_gray.shape
        crow, ccol = rows // 2, cols // 2
        u, v = np.meshgrid(np.arange(cols), np.arange(rows))
        D = np.sqrt((u - ccol)**2 + (v - crow)**2)

        if highpass:
            H = 1 / (1 + (d0 / (D + 1e-5))**(2 * n))
        else:
            H = 1 / (1 + (D / d0)**(2 * n))

        G = fshift * H
        ishift = np.fft.ifftshift(G)
        img_back = np.fft.ifft2(ishift)
        return img_as_ubyte(np.abs(img_back))
    
    thêm filter max/min
    def apply_min_filter(img, size=3):
        return minimum_filter(img, size=size)

    def apply_max_filter(img, size=3):
        return maximum_filter(img, size=size)
