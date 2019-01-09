# Ý tưởng

- Chuyển từng ảnh thành histogram, với số bins có thể điểu chỉnh được.
    - Từ 1 ảnh, lần lượt xoay 45 độ và 90 độ để có 2 ảnh khác. Gom lại thành 1 ảnh lớn hơn.
    - Cắt ảnh theo chiều dọc, số mảnh cắt bằng số bins của histogram
    - Với mỗi mảnh cắt của ảnh, tính tổng giá trị các pixel chứa chữ ký của ảnh.
      Tổng này sẽ là giá trị của từng bins.
- Chuyển histogram thành vector. Vector này sẽ là đầu vào của thuật toán.
- Sử dụng thuật toán k-nearest neighbors (KNN) được cài đặt sẵn của thư viện Sklearn để trainning và dự đoán

# Quá trình

- Thiết lập các tham số khi test.
    - Kích thước file ảnh chuẩn hóa: mặc định là (600,400)
    - Số lượng bins của histogram: mặc định là 100 bins.
    - K: mặc định là 3.
- Lấy dữ liệu trainning và test, với mỗi ảnh chuyển thành histogram.
    - Tiền xử lý
        - Lấy dữ liệu ảnh
        - Loại bỏ các chỉ tiết nhiễu và chuẩn hóa file ảnh.
        - Chuyển ảnh thành histogram đưa vào dữ liệu trainning và test.
- Thực hiện trainning và dự đoán:
    - Lần lược thay đổi các tham số đầu vào để thống kê độ chính xác.
- Đo độ chính xác Accuracy.

## Thực hiện trainning và test

    Thực hiện trainng và dự đoán trong từng tham số khác nhau.
    Dựa vào thống kê bên dưới, ta thấy độ chính xác cao nhất > 96% khi sử dụng 20 bins histogram cho hầu hết các size ảnh và số lượng K-neighbor

**Thống kê**

    Image size: (600, 200)
    K=5
    bins=[600, 500, 400, 300, 200, 100, 90, 80, 70, 60, 50, 40, 30, 20, 10]

Bin | Accuracy
--- | -------------
600 | 63.9175257732
500 | 67.0103092784
400 | 76.2886597938
300 | 70.1030927835
200 | 69.0721649485
100 | 77.3195876289
 90 | 78.3505154639
 80 | 78.3505154639
 70 | 80.412371134
 60 | 84.5360824742
 50 | 87.6288659794
 40 | 90.7216494845
 30 | 95.8762886598
 20 | 96.9072164948
 10 | 95.8762886598


    Image size: (600, 200)
    K=3
    bins=[600, 500, 400, 300, 200, 100, 90, 80, 70, 60, 50, 40, 30, 20, 10]
    
Bin | Accuracy
--- | -------------
600 | 70.1030927835
500 | 69.0721649485
400 | 80.412371134
300 | 72.1649484536
200 | 73.1958762887
100 | 80.412371134
 90 | 83.5051546392
 80 | 86.5979381443
 70 | 87.6288659794
 60 | 89.6907216495
 50 | 90.7216494845
 40 | 93.8144329897
 30 | 97.9381443299
 20 | 98.9690721649
 10 | 94.8453608247


    Image size: (400, 200)
    K=5
    bins=[200, 100, 90, 80, 70, 60, 50, 40, 30, 20, 10]

Bin | Accuracy
--- | -------------
200 | 69.0721649485
100 | 77.3195876289
 90 | 78.3505154639
 80 | 78.3505154639
 70 | 80.412371134
 60 | 84.5360824742
 50 | 87.6288659794
 40 | 90.7216494845
 30 | 95.8762886598
 20 | 96.9072164948
 10 | 95.8762886598


    Image size: (200, 100)
    K=3
    bins=[200, 100, 90, 80, 70, 60, 50, 40, 30, 20, 10]

Bin | Accuracy
--- | -------------
200 | 73.1958762887
100 | 80.412371134
 90 | 83.5051546392
 80 | 86.5979381443
 70 | 87.6288659794
 60 | 89.6907216495
 50 | 90.7216494845
 40 | 93.8144329897
 30 | 97.9381443299
 20 | 98.9690721649
 10 | 94.8453608247