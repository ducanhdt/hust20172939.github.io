---
title: "Tìm Link Facebook đúng cách"
categories:
  - technology
tags:
  - AI
  - Experience
---
 
## Lời mở đầu

Ai cũng có lúc muốn tìm info của một chàng trai, cô gái xinh đẹp nào đó với một lượng thông tin vô cùng ít ỏi, thậm chí chỉ là một bức ảnh chụp vội lúc vô tình người đó lướt qua. Dẫu biết là khó nhưng không hẳn là không thể, hôm nay tôi sẽ hướng dẫn các bạn cách tìm info của một sắn Bách Khoa chỉ với 1 bức ảnh. Bài viết không quá quá sâu vào vấn đề kĩ thuật nên không yêu cầu kiến thức lập trình, nhưng ai đã hoàn thành 300 bài code thiếu nhi sẽ là một lợi thế. 


## Vào việc!!!

Tổng quan: chúng ta sẽ  xây dựng một kho dữ liệu thông tin về từng người kèm theo link facebook tương ứng, khi có một bức ảnh bất kì ta sẽ so sánh với các thông tin trong kho để tìm người giống nhất, nếu người cần tìm có trong cơ sở dữ liệu và trả về đúng người đó thì xem như là thành công. 

Các bước thực hiện:
B1: Thu thập thật nhiều link facebook và ảnh tương ứng với link facebook đó. Càng nhiều dữ liệu thì càng dễ tìm được người cần tìm. Dữ liệu càng tinh thì xử lý càng nhanh. VD bạn cần tìm info gái xinh thì đào link facebook mấy đứa con trai làm gì.
B2: Chuyển các bức ảnh thành các đặc trưng ứng với từng người và lưu trữ lại, Bạn A mắt to môi nhỏ, bạn B mắt nhỏ môi dày chẳng hạn.
B3: Phân tích đặc trưng của bức ảnh mới, tìm ra người tương đồng nhất trong cơ sở dữ liệu. Đưa ảnh 1 thằng mắt to vào thì khả năng là A đấy.

## Tiến hành từng bước nào:

### Bước 1: Xây dựng dữ liệu
Giờ làm sao để có thể thu thập được một bộ dữ liêu thật chất lượng đây. Đơn giản nhất đó chính là cào link Facebook kèm avata, cào hết bạn bè của mình rồi bạn của bạn bè, bạn bè của bạn bè của bạn bè, cứ public là lấy hết về . Như thế là có thể thu được một lượng lớn dữ liệu, rất đa dạng. Nhưng vấn đề gặp phải của phương pháp này đó chính là dữ liệu rất tệ: không phải ai cũng để avata rõ mặt, không kể avata hoạt hình, avata trắng, bla bla... Hơn nữa có quá nhiều người dùng trong khi đa số lại là những người ta không quan tâm gây lãng phí tài nguyên tính toán trong khi những người ta thực sự quan tâm lại thiếu VD tôi chỉ quan tâm gái Bách Khoa thôi thì sao, khó. Vậy làm sao để giải quyết vấn đề này. Thay vì cào dữ liệu một cách thiếu chọn lọc ta sẽ sử dụng 1 vài chiến thuật ở đây. Các bài phát DRL của HSV thu hút rất nhiều SV BK tham gia comment, chỉ cần crawl hết đống này là có kha khá link facebook rồi. VD như ở bài về thử thách 7 ngày có thử thách chụp ảnh đeo khẩu trang, chỉ cần 1 vài dòng code crawl thiếu nhi kết hợp với vài dòng regex là ta đã có được bộ dữ liệu ảnh đeo khẩu trang như hình.

{% capture fig_img1 %}
![Foo]({{ 'assets/image/post/find_link/dataset.png' | relative_url }})
{% endcapture %}

<figure>
  {{ fig_img1 | markdownify | remove: "<p>" | remove: "</p>" }}
  <figcaption>bộ dữ liệu gồm MSSV, Link facebook và link hình ảnh đeo khẩu trang của SVBK</figcaption>
</figure>


Ảnh đeo khẩu trang thì không được rõ mặt lắm. Có cách nào để có ảnh chất lượng cao hơn không. Chả biết làm như nào, tôi đành gọi điện cho hắc cơ giấu tên (T), nhờ sự giúp đỡ. Sau khi thấy được tính chính nghĩa của đề tài, ngài T đã thâm nhập vào hệ thống của trường HUST để lấy ra bộ dữ liệu ảnh mặt mộc của sinh viên đê phục vụ mục đích nghiên cứu.


### Bước 2: Trích đặc trưng:

Giờ có ảnh rồi, công việc của chúng ta tiếp theo là gán nhãn dữ liệu. Công việc rất đơn giản đó chính là ngồi xem từng ảnh một, ghi lại xem người này mắt, mũi, miệng,... như thế nào. :V. Đấy là chỗ nào người ta làm như vậy chứ ở đây chúng ta không làm như thế. Trong thời đại cách mạng công nghiệp 4.0 như thế này, chúng ta phải dùng học máy, dùng Deep learning này nọ, mới ngầu. Vì thế tôi sẽ sử dụng mô hình InceptionResnetV1 với pretrain VGG của thư viện facenet-pytorch để trích đặc trưng của khuân mặt.
Trước khi đưa vào mô hình, ta sẽ bức ảnh crop lấy khuân mặt người bằng MTCNN cuối cùng ta thu được một vector số 512 chiều chứa đặc trưng của khuôn mặt

**Cài đặt:**
```bash
pip install facenet-pytorch
```
**Code:**
```python
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
import numpy as np
# If required, create a face detection pipeline using MTCNN:
mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device
)

# Create an inception resnet (in eval mode):
resnet = InceptionResnetV1(pretrained='vggface2').eval()
def get_image_embedding(link):
    img = Image.open(link)
    # Get cropped and prewhitened image tensor
    img_cropped = mtcnn(img)
    img_embedding = resnet(img_cropped.unsqueeze(0))
    return img_embedding.detach().numpy()
```

Để có thể hình dung dễ hơn cách mô hình hoạt động tôi sẽ sử dụng hình ảnh của một bạn lễ tân trong lễ tuyên dương SV5T hôm qua để thử nghiệm. Ta có bức ảnh như sau.
{% capture fig_img2 %}
![Foo]({{ 'assets/image/post/find_link/anh_test.png' | relative_url }})
{% endcapture %}

<figure>
  {{ fig_img2 | markdownify | remove: "<p>" | remove: "</p>" }}
  <figcaption>Ảnh một bạn lễ tân X nào đó tại lễ tuyên dương SV5T 2021</figcaption>
</figure>

{% capture fig_img3 %}
![Foo]({{ 'assets/image/post/find_link/mtcnn.png' | relative_url }})
{% endcapture %}

<figure>
  {{ fig_img3 | markdownify | remove: "<p>" | remove: "</p>" }}
  <figcaption>Khuân mặt được crop bằng MTCNN</figcaption>
</figure>

### Bước 3: Tìm kiếm

Đầu tiên ta sẽ trích đặc trưng và lưu lại tất cả các khuân mặt đã có với mô hình trên, mình dự đoán bạn này là K64 nên sẽ chỉ chạy với dữ liệu của k64 cho nhanh:

```python
from os import listdir
best_match = ""
best_distance=10
mssv = []
distances = []
for name in listdir('/content/2019/'):
    url = "/content/2019/"+name
    try:
        test = get_image_embedding(url)
    except:
        continue
    distance = get_distance(test,example)
    distances.append(distance)
    mssv.append(name)
    if len(mssv)%500==0:
        print(len(mssv))
    if distance<best_distance:
        best_distance = distance
```

Dù duyệt tuần tự hơi lâu 1 chút nhưng Hà Nội là không được vội nên kệ vậy. Sau khi có khoảng cách đến tất cả các điểm thì ta sẽ chọn ra những điểm có khoảng cách nhỏ nhất để check lại xem mô hình có hoạt động tốt hay không.


{% capture fig_img4 %}
![Foo]({{ 'assets/image/post/find_link/ketqua.png' | relative_url }})
{% endcapture %}

<figure>
  {{ fig_img4 | markdownify | remove: "<p>" | remove: "</p>" }}
  <figcaption>Sinh viên K64 giống ảnh nhất</figcaption>
</figure>

Giờ việc cuối cùng là xem liệu ta có link facebook của sinh viên kia không nào :V.


{% capture fig_img5 %}
![Foo]({{ 'assets/image/post/find_link/bilua.png' | relative_url }})
{% endcapture %}

<figure>
  {{ fig_img5 | markdownify | remove: "<p>" | remove: "</p>" }}
  <figcaption>Tada!!! Không có kết quả nào như vậy cả</figcaption>
</figure>

Nếu bạn đã đọc đến đây thì chúc mừng bạn đã bị lừa.



