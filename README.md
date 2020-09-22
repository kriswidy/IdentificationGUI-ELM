# System Identification GUI with ELM
Sistem ini memanfaatkan metode Extreme Learning Machine untuk membantu mengklasifikasikan objek GUI yang teridentifikasi pada screenshoot aplikasi. Sistem ini digunakan untuk membantu proses otomasi dari citra menjadi kode. Inputan berupa potongan gambar dari screenshot aplikasi. Sistem dapat mendeteksi komponen yang telah di inputkan. Hasil prediksi berupa nama komponen dan xml code.

<b>Kebutuhan Library</b>
1. Install Python
- Linux
src: https://docs.python-guide.org/starting/install3/linux/
- WIndows
src: https://www.python.org/downloads/windows/

2. Install Flask
src: https://flask.palletsprojects.com/en/1.1.x/installation/

3. Install Numpy
src: https://www.edureka.co/blog/install-numpy/

4. Install OpenCV
- Linux
src: https://docs.opencv.org/master/d7/d9f/tutorial_linux_install.html

- Windows
src: https://www.learnopencv.com/install-opencv3-on-windows/

5. Install Tensorflow
src: https://www.tensorflow.org/install

<br>
<br>

<i>*Pastikan memiliki CodeEditor yang mendukung Python, semisal visual studio code</i>
<br>
<br>

<b>Cara menjalankan sistem:</b>
Jika kamu menggunakan linux, ikuti langkah sebagaimana berikut ini:
1. Download berkas
2. Unzip berkas pada folder direktory dimana kamu menginstall Python
3. Klik kanan folder "Huumeh", pilih Open with terminal
4. Run "export FLASK_APP=hummeh.py"
5. Run "flask run", website bisa diakses melalui url http://127.0.0.1:5000/ pada web browser
<br>
<br>

<b>Cara menggunakan sistem:</b>
1. Pertama kali saat kamu mengakses website melalui url yang didapatkan saat menjalankan perintah "flask run", kamu akan mendapatkan tampilan halaman depan dengan informasi cara penggunaan dan beberapa navigation bar. Terdapat navigation bar yang berbeda-beda dengan fungsinya, yakni ada Home, About, Testing, dan Cropping.
2. Navigation bar, akan membawamu kepada halaman utama.
3. Navigation bar About, akan membawamu pada halaman informasi mengenai sistem dan penulis.
4. Navigation bar Testing, akan membawamu pada halaman Testing. Dimana kamu dapat melakukan identifikasi komponen GUI dari gambar yang akan kamu inputkan. Input dapat dilakukan dengan cara drag and drop ataupun open file dialog. Jika file sudah dipilih, gambar akan di preview. Selanjutnya jika ingin membatalkan atau mengganti gambar yang ingin di inputkan dapat di cancel menggunakan fungsi tombol remove, dan jika ingin diidentifikasi, dapat digunakan tombol upload. Hasilnya akan ditampilkan pada halaman Prediksi. Dimana akan di munculkan jenis dari komponen yang terindentifikasi dari gambar inputan beserta kode programnya.
6. Navigation bar Cropping, akan membawamu pada halaman Cropping Image. Dimana kamu dapat melakukan pemotongan gambar pada hasil tangkapan layar penuh dari aplikasi Andoid yang ingin dijadikan referensi ataupun ingin diidentifikasi. Terdapat tombol pilih gambar yang akan membantumu memilih gambar yang akan dipotong, jika gambar sudah dipilih, ia akan di tampilkan pada halaman selection area. Kamu dapat memilih bagian mana yang ingin dipotong. Setelah itu klik tombol Done. Gambar yang sudah terpotong akan di preview, untuk kemudian dapat disimpan dengan cara unduh. Gambar yang sudah terunduh, dapat diinputkan untuk diidentifikasi.

<br>
<br>
<i>*Jika ada pertanyaan lebih lanjut dapat melalui email: kwf2102@gmail.com
