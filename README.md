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

