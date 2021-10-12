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

## Data Preparation

Seperti yang disebutkan di Solution Statement, berikut adalah tahapan-tahapan dalam melakukan pra-pemrosesan data :
1. Handling Missing Values

![image](https://user-images.githubusercontent.com/73678966/136870223-5a4ce174-961b-4185-a452-56725e277fa9.png)

Karena pada data tersebut terdapat banyak sekali variabel yang memiliki nilai null, maka untuk mengatasinya dilakukan beberapa tahap untuk mengatasinya
  - Menghapus variabel yang nilai missing valuesnya > 30%
  - Mengisi nilai yang kosong dengan nilai terbanyak jika merupakan variabel categoric
  - Mengisi nilai yang kosong dengan nilai median jika variabel numerik karena variabel numerik tersebut memiliki distrivusi right skewed

![image](https://user-images.githubusercontent.com/73678966/136870427-0cc63ae5-f21c-44e8-93a4-66e38fbbb415.png)
2. Feature Engeneering
  - Menghapus variabel yang memiliki variasi yang sedikit
3. Persiapan data untuk modelling
  - Melakukan One-Hot- Encoding untuk data categorical
  One-Hot-Encode adalah proses untuk membuat kolom baru dari variabel kategorikal kita di mana setiap kategori menjadi kolom baru dengan nilai 0 atau 1 (0 mewakili tidak ada dan 1 mewakili ada). Tahapan ini dilakukan karena banyak teknik statistik atau persamaan machine learning yang hanya menerima nilai numerik, bukan nilai kategorik.
  - Melakukan pembagian dataset menjadi dua bagian dengan rasio 70% untuk data latih dan 30% untuk data uji
   Tahap ini dilakukan agar kita bisa menguji model kita menggunakan data train dan dataset sehingga model yang dihasilkan dapat memiliki performa yang sama baiknya pada data uji seperti pada data latih. 
   - Melakukan Standarisasi
   Tahap terakhir dengan melakukan standarisasi data. Hal ini akan membuat semua fitur numerik berada dalam skala data yang sama juga membuat komputasi dari pembuatan model dapat berjalan lebih cepat. Metode yang digunakan adalah StandarsScaler, sehingga data memiliki nilai mean 0 dan nilai standar deviasi adalah 1.
   
   ![image](https://user-images.githubusercontent.com/73678966/136871347-26bcba4a-4fea-47b7-afae-aefbbac5c162.png)
   
   - Mengatasi data yang tidak seimbang jumlahnya dengan label lain menggunakan teknik resample menggunakan SMOTE
  Target data yang kita miliki memiliki data yang tidak seimbang jumlahnya, sehingga untuk mengasilkan peforma model yang baik diperlukan melakukan teknik resampling data menggunkan SMOTE. Metode SMOTE merupakan salah satu teknik oversampling yang akan menambah jumlah data kelas minor agar setara dengan kelas mayor dengan cara membangkitkan data buatan. Data buatan atau sintesis tersebut dibuat berdasarkan k-tetangga terdekat (k-nearest neighbor).
  
  ## Modelling
  
  Setelah melakukan pra-pemrosesan data yang baik pada tahap modeling akan dilakukan dua hal, yakni tahap pembuatan model baseline dan pembuatan model yang dikembangkan.
  
 - Cek Model classification terbaik
 Pada tahap ini dicoba beberapa model untuk mengetahui model mana yang paling baik dalam memprediksi leads, Sehingga kita bisa menentukan model yang paling baik akan di kembangkan nantinya.
 
 ![image](https://user-images.githubusercontent.com/73678966/136874013-3870827f-f3c8-49fe-a781-b7925d063ef9.png)
 
 Dengan menggunakan teknik SMOTE model terbaik yang dihasilkan dalam memprediksi leads adalah XGBClassifier.
 
 - Baseline model
 Pada tahap ini saya membuat model dasar dengan menggunakan modul scikit-learn yakni XGBClassifier tanpa menggunakan parameter tambahan. Lalu melakukan prediksi kepada data ujinya.
 
 - Model yang dikembangkan
 Pada tahap ini saya melakukan hyperparameter tunning dengan menggunakan GridSearchCV dalam menentukan parameter terbaik untuk model XGBClassifier dalam memprediksi target.
 
 Hasilnya dapat dilihat sebagai berikut:
 
 ## Evaluasi
 
   1. Model Baseline
 
 ![image](https://user-images.githubusercontent.com/73678966/136874417-e50bb7dc-6169-470b-89ac-68758713def7.png)
 
 dengan Confussion Matriks sebagaik berikut :
 
 ![image](https://user-images.githubusercontent.com/73678966/136874839-9598d8bb-375d-4148-bbaf-56ce554be3de.png)

 
   2. Model Hyperparameter Tunning
 
 ![image](https://user-images.githubusercontent.com/73678966/136874495-fee70eb2-4dff-4dbd-a0b5-40ed12a646a0.png)
 
  dengan Confussion Matriks sebagaik berikut :
  
  ![image](https://user-images.githubusercontent.com/73678966/136874920-072c9c28-85f8-4388-ac39-73c0639ba924.png)

 
 Pada hasil tersebut tidak ada perbedaan yang sangat signifikan dari nilai akurasi tersebut.
 
 - 
 
 








