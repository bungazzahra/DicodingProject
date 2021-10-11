# Laporan Proyek Machine Learning - Fatimah Azzahra Arham


## Domain Proyek


Domain proyek yang dipilih dalam proyek machine learning ini adalah mengenai Business dan Industri dengan judul proyek "Leads Scoring Prediction".

![alt text](https://miro.medium.com/max/467/1*roT_nhFL9cdR5Dg0QfLR5A.png)

- Latar Belakang

Machine learning termasuk dalam artificial intelligence (AI). Sama seperti arti harfiahnya, teknologi machine learning dapat mengembangkan kemampuannya dengan mempelajari data empiris. Kini, teknologi ini dapat diterapkan dalam berbagai sektor industri, seperti dalam bisnis industri. Dalam dunia bisnis terdapat persaingan yang sangat ketat, dengan menggunakan data science para pebisnis dapat dibantu dalam mengambil keputusan/strategi  yang tepat dengan menggunakan data historis yang mereka miliki.

Untuk mengembangkan bisnis diperlukan berbagai macam strategi. Salah satunya, menggunakan lead scoring. Peran lead scoring sangat penting, namun sayangnya masih banyak pebisnis yang belum menerapkannya pada bisnis. Lead scoring adalah sebuah sistem yang dapat diterapkan oleh tim sales dan marketing untuk mengkategorikan dan memprioritaskan sebuah peluang. Lead scoring juga bisa mencerminkan peluang dan tingkat peminat pada bisnis Anda. Biasanya setiap perusahaan memiliki model lead scoring yang berbeda untuk menentukan poin penilaian lead scoring yang nanti juga berguna dalam membangun lead generation. Namun cara yang paling umum, menggunakan data historis yang sudah Anda miliki.

Beberapa manfaat yang bisa kita dapatkan ketika mengguenakan lead scoring adalah :
1. Marketing campaign yang lebih efektif
2. Sinkronisasi sales dan marketing
3. Peningkatan keuntungan

Sources : https://vidhutandon.medium.com/lead-conversion-score-prediction-using-python-a65d5bb7ccff

## Business Understanding

### Problem Statement
Berangkat dari latar belakang diatas, berikut ini merupakan rincian masalah yang dapat diselesaikan pada proyek ini :
- Bagaimana cara melakukan pra-pemrosesan data air agar dapat digunakan untuk membuat model yang baik?
- Bagaimana cara membuat model untuk mengetahui customer yang memiliki kemungkinan besar tertarik pada layanan atau product yang kita miliki?

### Goals
Berikut adalah tujuan dari dibuatnya proyek ini :

- Melakukan pra-pemrosesan data leads yang baik agar dapat digunakan dalam membuat model.
- Membuat model machine learning untuk mendeteksi customer yang memiliki kemungkinan besar tertarik pada layanan product yang kita miliki dengan  tingkat akurasi > 80%.


### Solution statements
Solusi yang dapat dilakukan untuk memenuhi tujuan dari proyek ini diantaranya :

Untuk pra-pemrosesan data dapat dilakukan beberapa teknik, diantaranya :
- Menghapus variabel yang nilai missing valuesnya > 30%
- Mengisi nilai yang kosong dengan nilai terbanyak jika merupakan variabel categoric dan nilai median jika variabel numerik.
- Melakukan One-Hot-Encoding pada data categorical
- Mengatasi data yang tidak seimbang jumlahnya dengan label lain menggunakan teknik resample (SMOTE)
- Melakukan pembagian dataset menjadi dua bagian dengan rasio 70% untuk data latih dan 30% untuk data uji
- Melakukan Standarisasi pada data numerik

Poin pra-pemrosesan data akan dibahas lebih lanjut pada bagian `Data Preparation`.

Pada pembuatan model digunanakan model XGBClassifier dan Random Forest dalam pembuatan baseline. Algoritma tersebut memiliki nilai akurasi yang baik dikarenakan merupakan salah satu model Ensamble yang cocok digunakan untuk memprediksi categorical target. Cara kerja algoritma ini adalah sebagai berikut https://towardsdatascience.com/xgboost-fine-tune-and-optimize-your-model-23d996fab663

 Selanjutnya dilakukan Hyperparameter Tunning dengan menggunakan Gridsearc
 
 ## Data Understanding

Data yang digunakan merupakan data Lead Conversion dataset dari Kaggle. kumpulan data berisi lebih dari 9.000 customer dengan fitur pelanggan seperti lead origin, source of lead, total time spent on the website, total visits on the website, demographics information dan kolom target Dikonversi (menunjukkan 1 untuk konversi dan 0 untuk tidak ada konversi).

Sumber data : https://www.kaggle.com/ashydv/leads-dataset

Variabel data :

| Variables| Heading 2 | 
|-----------|:-----------:|
Prospect ID|A unique ID with which the customer is identified.|
Lead Number|A lead number assigned to each lead procured.|
Lead Origin|The origin identifier with which the customer was identified to be a lead. Includes API, Landing Page Submission, etc.|
Lead Source|The source of the lead. Includes Google, Organic Search, Olark Chat, etc.|
Do Not Email|An indicator variable selected by the customer wherein they select whether of not they want to be emailed about the course or not.|
Do Not Call|An indicator variable selected by the customer wherein they select whether of not they want to be called about the course or not.|
Converted|The target variable. Indicates whether a lead has been successfully converted or not.|
TotalVisits|The total number of visits made by the customer on the website.|
Total Time Spent on Website|The total time spent by the customer on the website.|
Page Views Per Visit|Average number of pages on the website viewed during the visits.|
Last Activity|Last activity performed by the customer. Includes Email Opened, Olark Chat Conversation, etc.|
Country|The country of the customer.|
Specialization|The industry domain in which the customer worked before. Includes the level 'Select Specialization' which means the customer had not selected this option while filling the form.|
How did you hear about X Education|The source from which the customer heard about X Education.|
What is your current occupation|Indicates whether the customer is a student, umemployed or employed.|
What matters most to you in choosing this course|An option selected by the customer indicating what is their main motto behind doing this course.|
Search|Indicating whether the customer had seen the ad in any of the listed items.|
Magazine|Indicating whether the customer had seen the ad in any of the listed items.|
Newspaper Article|Indicating whether the customer had seen the ad in any of the listed items.|
X Education Forums|Indicating whether the customer had seen the ad in any of the listed items.|
Newspaper|Indicating whether the customer had seen the ad in any of the listed items.|
Digital Advertisement|Indicating whether the customer had seen the ad in any of the listed items.|
Through Recommendations|Indicates whether the customer came in through recommendations.|
Receive More Updates About Our Courses|Indicates whether the customer chose to receive more updates about the courses.|
Tags|Tags assigned to customers indicating the current status of the lead.|
Lead Quality|Indicates the quality of lead based on the data and intuition the the employee who has been assigned to the lead.|
Update me on Supply Chain Content|Indicates whether the customer wants updates on the Supply Chain Content.|
Get updates on DM Content|Indicates whether the customer wants updates on the DM Content.|
Lead Profile|A lead level assigned to each customer based on their profile.|
City|The city of the customer.|
Asymmetrique Activity Index|An index and score assigned to each customer based on their activity and their profile|
Asymmetrique Profile Index|An index and score assigned to each customer based on their activity and their profile|
Asymmetrique Activity Score|An index and score assigned to each customer based on their activity and their profile|
Asymmetrique Profile Score|An index and score assigned to each customer based on their activity and their profile|
I agree to pay the amount through cheque|Indicates whether the customer has agreed to pay the amount through cheque or not.|
a free copy of Mastering The Interview|Indicates whether the customer wants a free copy of 'Mastering the Interview' or not.|
Last Notable Activity|The last notable acitivity performed by the student.|


