# MODEL_RESULTS

Khi chạy thử nghiệm các mô hình deep learning, dù cho đã cài đặt seed ngẫu nhiên để đảm bảo tính tái lập, chúng tôi vẫn nhận thấy sự biến động nhẹ về kết quả giữa các lần chạy. Điều này có thể do nhiều yếu tố cụ thể:
- Khi chạy trên GPU, thư viện cuDNN của NVIDIA tối ưu hóa tốc độ bằng cách sử dụng các thao tác Atomic Add (Cộng nguyên tử) trên nhiều luồng (threads) chạy song song. Việc luồng nào tính xong trước và ghi dữ liệu vào bộ nhớ trước là **hoàn toàn ngẫu nhiên** tùy thuộc vào tình trạng phần cứng tại mili-giây đó.
- Mặc dù trong toán học lý tưởng $(A + B) + C = A + (B + C)$, nhưng trong tính toán dấu phẩy động (Floating-point) của máy tính, việc cộng theo các thứ tự khác nhau sẽ sinh ra sai số làm tròn siêu nhỏ (ví dụ lệch ở chữ số thập phân thứ 8). Qua hàng chục epochs và hàng nghìn tham số, các sai số siêu nhỏ này tích tụ lại (gọi là hiệu ứng cánh bướm) khiến kết quả F1 hoặc Loss cuối cùng bị lệch.

Vì vậy chúng tôi tiến hành chạy nhiều lần (4 lần) trên mỗi mô hình đề xuất và ghi lại kết quả tốt nhất (best), trung bình (average), độ lệch chuẩn (Variance) cũng như thời gian chạy để đảm bảo tính khách quan và minh bạch trong việc đánh giá hiệu suất của các mô hình.

**Lưu ý**: Thời gian chạy được đề cập trong file này là ước tính trung bình 4 lần chạy trên NVIDIA GPU Tesla T4 của Kaggle. Thời gian thực tế khi chạy có thể thay đổi, nhưng không vượt quá **1.5 lần** thời gian ước tính.

## TAANet: Temporal Actions Attention Network (54m46s)
| Metric | Run 1 | Run 2 | Run 3 | Run 4 | Best | Average | Variance |
|--------|-------|-------|-------|-------|------|---------|----------|
|Macro F1 Attribute 1| 0.9989 | 0.9989 | 1.0000 | 1.0000 | 1.0000 | 0.9995 | 0.0006 |
|Macro F1 Attribute 2| 0.9999 | 0.9999 | 0.9994 | 0.9994 | 0.9999 | 0.9997 | 0.0001 |
|Macro F1 Attribute 3| 0.9983 | 0.9983 | 0.9980 | 0.9980 | 0.9983 | 0.9982 | 0.0001 |
|Macro F1 Attribute 4| 0.9928 | 0.9928 | 0.9927 | 0.9927 | 0.9928 | 0.9928 | 0.0001 |
|Macro F1 Attribute 5| 0.9993 | 0.9993 | 0.9994 | 0.9994 | 0.9994 | 0.9993 | 0.0001 |
|Macro F1 Attribute 6| 0.9991 | 0.9991 | 0.9992 | 0.9992 | 0.9992 | 0.9991 | 0.0001 |
|Validation Exact Match| 0.9968 | 0.9968 | 0.9967 | 0.9967 | 0.9968 | 0.9967 | 0.0001 |
|Public Test Exact Match| 0.9574 | 0.9574 | 0.9579 | 0.9579 | 0.9579 | 0.9577 | 0.0003 |

## TACNet: Temporal Actions Convolutional Network (35m50s)
| Metric | Run 1 | Run 2 | Run 3 | Run 4 | Best | Average | Variance |
|--------|-------|-------|-------|-------|------|---------|----------|
|Macro F1 Attribute 1| 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
|Macro F1 Attribute 2| 1.0000 | 1.0000 | 0.9999 | 0.9999 | 1.0000 | 0.9999 | 0.0001 |
|Macro F1 Attribute 3| 0.9987 | 0.9986 | 0.9982 | 0.9982 | 0.9987 | 0.9984 | 0.0002 |
|Macro F1 Attribute 4| 0.9890 | 0.9924 | 0.9944 | 0.9972 | 0.9972 | 0.9932 | 0.0008 |
|Macro F1 Attribute 5| 0.9993 | 0.9985 | 0.9993 | 0.9993 | 0.9993 | 0.9991 | 0.0004 |
|Macro F1 Attribute 6| 0.9993 | 0.9989 | 0.9991 | 0.9991 | 0.9993 | 0.9991 | 0.0002 |
|Validation Exact Match| 0.9992 | 0.9968 | 0.9972 | 0.9969 | 0.9992 | 0.9975 | 0.0011 |
|Public Test Exact Match| 0.9675 | 0.9662 | 0.9581 | 0.9735 | 0.9735 | 0.9663 | 0.0061 |

## TARNet: Temporal Actions Recurrent Network (22m37s)
| Metric | Run 1 | Run 2 | Run 3 | Run 4 | Best | Average | Variance |
|--------|-------|-------|-------|-------|------|---------|----------|
|Macro F1 Attribute 1| 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
|Macro F1 Attribute 2| 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
|Macro F1 Attribute 3| 0.9987 | 0.9988 | 0.9987 | 0.9986 | 0.9988 | 0.9987 | 0.0001 |
|Macro F1 Attribute 4| 0.9972 | 0.9934 | 0.9945 | 0.9972 | 0.9972 | 0.9956 | 0.0017 |
|Macro F1 Attribute 5| 0.9993 | 0.9991 | 0.9993 | 0.9994 | 0.9994 | 0.9993 | 0.0001 |
|Macro F1 Attribute 6| 0.9992 | 0.9992 | 0.9989 | 0.9992 | 0.9992 | 0.9991 | 0.0001 |
|Validation Exact Match| 0.9976 | 0.9975 | 0.9974 | 0.9976 | 0.9976 | 0.9975 | 0.0001 |
|Public Test Exact Match| 0.9654 | 0.9703 | 0.9746 | 0.9703 | 0.9746 | 0.9701 | 0.0033 |

## TAGNet: Temporal Actions Graph Network (30m28s)
| Metric | Run 1 | Run 2 | Run 3 | Run 4 | Best | Average | Variance |
|--------|-------|-------|-------|-------|------|---------|----------|
|Macro F1 Attribute 1| 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
|Macro F1 Attribute 2| 1.0000 | 0.9998 | 1.0000 | 0.9998 | 1.0000 | 0.9999 | 0.0001 |
|Macro F1 Attribute 3| 0.9983 | 0.9984 | 0.9979 | 0.9982 | 0.9984 | 0.9982 | 0.0002 |
|Macro F1 Attribute 4| 0.9950 | 0.9949 | 0.9972 | 0.9949 | 0.9972 | 0.9958 | 0.0011 |
|Macro F1 Attribute 5| 0.9994 | 0.9994 | 0.9993 | 0.9993 | 0.9994 | 0.9993 | 0.0001 |
|Macro F1 Attribute 6| 0.9992 | 0.9989 | 0.9992 | 0.9990 | 0.9992 | 0.9991 | 0.0002 |
|Validation Exact Match| 0.9974 | 0.9971 | 0.9969 | 0.9968 | 0.9974 | 0.9971 | 0.0002 |
|Public Test Exact Match| 0.9692 | 0.9704 | 0.9725 | 0.9720 | 0.9725 | 0.9708 | 0.0013 |
