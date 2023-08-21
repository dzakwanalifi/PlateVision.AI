import streamlit as st
from ultralytics import YOLO
import json
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import time
import io
import zipfile

fav_logo = Image.open('assets/mini-logo.png')
st.set_page_config(
        page_title="PlateVision.AI",
        page_icon=fav_logo,
    )

if 'results_df' not in st.session_state:
    st.session_state.results_df = None
if 'image_outputs' not in st.session_state:
    st.session_state.image_outputs = []
if 'output_tab' not in st.session_state:
    st.session_state.output_tab = False
if 'input_type' not in st.session_state:
    st.session_state.input_type = "Satuan"

logo = Image.open('assets/logo.png')
st.image(logo)

st.success('Tim BDC - SD2023040000215', icon="💡")

with st.expander("Apa itu PlateVision.AI"):
    st.write("PlateVision.AI adalah sebuah aplikasi yang memanfaatkan teknologi kecerdasan buatan untuk prediksi plat nomor kendaraan dengan tingkat akurasi yang tinggi. Dengan menggunakan model YOLOv5x, PlateVision.AI mampu mengenali dan memprediksi plat nomor dengan akurasi mencapai 99%. Dengan kombinasi kekuatan teknologi YOLOv5x dan pelatihan dataset yang komprehensif, PlateVision.AI memberikan solusi efisien dan andal untuk mengenali plat nomor kendaraan secara otomatis dan akurat.")

with st.expander("Model YOLOv5x"):

    st.subheader("YOLOv5x")
    st.write("YOLOv5x merupakan varian YOLO dengan arsitektur xlarge. Sebagai perkembangan dalam metodologi deteksi objek, YOLOv5x menggabungkan elemen penting dari model YOLOv5 yang dikembangkan oleh Ultralytics, dengan integrasi head split anchor-free dan objectness-free yang sebelumnya diperkenalkan pada model YOLOv8. Adaptasi ini mempertajam arsitektur model, menghasilkan keseimbangan antara akurasi dan kecepatan yang lebih baik dalam tugas deteksi objek. Dengan hasil empiris dan fitur-fiturnya, YOLOv5x memberikan alternatif efisien bagi mereka yang mencari solusi tangguh baik dalam penelitian maupun aplikasi praktis.")

    st.subheader("Parameter Pelatihan Model")
    
    params = [
    ["epochs", 100, "Jumlah epoch pelatihan"],
    ["imgsz", 320, "Ukuran gambar input sebagai bilangan bulat"],
    ["batch", 16, "Jumlah gambar per batch"],
    ["lr0", 0.001, "Tingkat pembelajaran awal"],
    ["lrf", 0.01, "Tingkat pembelajaran akhir (lr0 * lrf)"],
    ["dropout", 0.2, "Penggunaan regulasi dropout (hanya untuk pelatihan klasifikasi)"],
    ["optimizer", "AdamW", "Optimizer yang digunakan"],
    ["momentum", 0.937, "Momentum SGD/beta1 Adam"],
    ["weight_decay", 0.0005, "Weight decay optimizer 5e-4"],
    ["warmup_epochs", 3.0, "Jumlah epoch pemanasan"],
    ["warmup_momentum", 0.8, "Momentum awal pemanasan"],
    ["warmup_bias_lr", 0.1, "Learning rate awal pemanasan"],
    ["iou", 0.7, "Intersection over Union"],
    ["max_det", 300, "Jumlah deteksi maksimal"]
    ]

    params_df = pd.DataFrame(params, columns=["Parameter", "Nilai", "Keterangan"])
    st.dataframe(data=params_df, hide_index=True, use_container_width=True)

    st.subheader("Metrik Pelatihan Model")

    st.image("assets/training-plot.png", caption="Kurva Pelatihan YoloV5x")

input_type = st.radio("Pilih jenis input:", ("Satuan", "Banyak"))

if st.session_state.input_type != input_type:
    st.session_state.results_df = None
    st.session_state.image_outputs = []
    st.session_state.output_tab = False
    st.session_state.input_type = input_type

if input_type == "Banyak":
    uploaded_images = st.file_uploader("Upload banyak gambar...", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

    if uploaded_images:
        st.subheader("Pratinjau Gambar")
        tab_labels = [uploaded_image.name for uploaded_image in uploaded_images]

        tabs = st.tabs(tab_labels)

        for idx, tab in enumerate(tabs):
            with tab:
                uploaded_image = uploaded_images[idx]
                image = Image.open(uploaded_image)
                st.image(image, use_column_width=True)

        if len(uploaded_images) > 5:
            st.warning("Anda mengunggah lebih dari 5 gambar. Hanya tabel hasil prediksi yang akan ditampilkan.")

        if st.button("Mulai Prediksi"):
            model = YOLO('model/best.pt')

            progress_text = "Melakukan prediksi..."
            my_bar = st.progress(0.0, text=progress_text)

            total_images = len(uploaded_images)
            results_list = []

            for idx, uploaded_image in enumerate(uploaded_images):
                image = Image.open(uploaded_image)

                results = model.predict(image,
                                        hide_conf=True, line_width=1, retina_masks=True,
                                        iou=0.5, augment=True, max_det=9, agnostic_nms=True)

                result = results[0]
                prediction_json = result.tojson()
                predictions = json.loads(prediction_json)

                with open('data/predefined_classes.txt', 'r') as f:
                    classes = f.read().split()

                filtered_predictions = [pred for pred in predictions if pred['name'] in classes]
                sorted_predictions = sorted(filtered_predictions, key=lambda x: x['box']['x1'])
                combined_classes = ''.join([pred['name'] for pred in sorted_predictions])

                file_name = uploaded_image.name

                results_list.append({"Nama File": file_name, "Vehicleregistrationplatebymodel": combined_classes})
                my_bar.progress((idx + 1) / total_images, text=progress_text)

                st.session_state.image_outputs.append((uploaded_image, sorted_predictions))

            st.subheader("Hasil Prediksi")

            results_df = pd.DataFrame(results_list)
            
            results_df['Numeric Part'] = results_df['Nama File'].str.extract(r'(\d+)', expand=False).astype(int)
            sorted_results_df = results_df.sort_values('Numeric Part')

            sorted_results_df = sorted_results_df.drop('Numeric Part', axis=1)

            st.session_state.results_df = pd.DataFrame(sorted_results_df)
            st.dataframe(st.session_state.results_df, use_container_width=True, hide_index=True)
            
            if len(uploaded_images) <= 5:
                st.subheader("Output Gambar")
                st.session_state.output_tab = True
                tabs_output = st.tabs(tab_labels)

                for idx, tab in enumerate(tabs_output):
                    with tab:
                        uploaded_image, sorted_predictions = st.session_state.image_outputs[idx]
                        image = Image.open(uploaded_image)

                        fig, ax = plt.subplots(figsize=(image.width / 100, image.height / 100))
                        ax.imshow(image)
                        for pred in sorted_predictions:
                            box = pred['box']
                            rect = plt.Rectangle((box['x1'], box['y1']), box['x2'] - box['x1'], box['y2'] - box['y1'],
                                                fill=False, color='red', linewidth=2)
                            ax.add_patch(rect)
                            font_size = int(image.width / 80)
                            plt.text(box['x1'], box['y1'], pred['name'], color='black', backgroundcolor='white', fontsize=font_size)
                        plt.axis('off')
                        st.pyplot(fig, bbox_inches='tight', pad_inches=0)

        if st.session_state.output_tab is not None and st.session_state.results_df is not None:
            csv_buffer = io.StringIO()
            st.session_state.results_df.to_csv(csv_buffer)
            csv_bytes = csv_buffer.getvalue()
            st.download_button("Unduh Hasil Prediksi (CSV)", data=csv_bytes, file_name="hasil_prediksi.csv", mime="text/csv")

            if st.session_state.image_outputs is not None:
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
                    for idx, (uploaded_image, sorted_predictions) in enumerate(st.session_state.image_outputs):
                        image = Image.open(uploaded_image)
                        fig, ax = plt.subplots(figsize=(image.width / 100, image.height / 100))
                        ax.imshow(image)
                        for pred in sorted_predictions:
                            box = pred['box']
                            rect = plt.Rectangle((box['x1'], box['y1']), box['x2'] - box['x1'], box['y2'] - box['y1'],
                                                fill=False, color='red', linewidth=2)
                            ax.add_patch(rect)
                            font_size = int(image.width / 80)
                            plt.text(box['x1'], box['y1'], pred['name'], color='black', backgroundcolor='white', fontsize=font_size)
                        plt.axis('off')
                        plt.savefig(f"{uploaded_image.name}", bbox_inches='tight', pad_inches=0)
                        zip_file.write(f"{uploaded_image.name}")
                st.download_button("Unduh Gambar Output (ZIP)", data=zip_buffer.getvalue(), file_name="output_images.zip", mime="application/zip")
        
        if st.session_state.output_tab is not None:
            if st.button("Reset Prediksi"):
                st.session_state.results_df = None
                st.session_state.image_outputs = []
                st.session_state.output_tab = False

            st.warning("Pastikan anda menekan tombol `Reset Prediksi` sebelum melakukan prediksi kembali.")

else:
    uploaded_image = st.file_uploader("Pilih gambar...", type=["jpg", "png", "jpeg"])
    
    if uploaded_image is not None:
        st.subheader("Pratinjau Gambar")
        image = Image.open(uploaded_image)
        st.image(image, use_column_width=True)

        if st.button("Mulai Prediksi"):
            model = YOLO('model/best.pt')

            progress_text = "Melakukan prediksi..."
            my_bar = st.progress(0.0, text=progress_text)

            results = model.predict(image,
                                    hide_conf=True, line_width=1, retina_masks=True,
                                    iou=0.5, augment=True, max_det=9, agnostic_nms=True)
            
            for percent_complete in range(100):
                time.sleep(0.02)
                my_bar.progress((percent_complete + 1) / 100, text=progress_text)

            result = results[0]
            prediction_json = result.tojson()
            predictions = json.loads(prediction_json)

            with open('data/predefined_classes.txt', 'r') as f:
                classes = f.read().split()

            filtered_predictions = [pred for pred in predictions if pred['name'] in classes]
            sorted_predictions = sorted(filtered_predictions, key=lambda x: x['box']['x1'])
            combined_classes = ''.join([pred['name'] for pred in sorted_predictions])

            st.subheader("Hasil Prediksi")

            st.metric(label="Plat Nomor", value=combined_classes)

            fig, ax = plt.subplots(figsize=(image.width / 100, image.height / 100))
            ax.imshow(image)
            for pred in sorted_predictions:
                box = pred['box']
                rect = plt.Rectangle((box['x1'], box['y1']), box['x2'] - box['x1'], box['y2'] - box['y1'],
                                    fill=False, color='red', linewidth=2)
                ax.add_patch(rect)
                font_size = int(image.width / 80)
                plt.text(box['x1'], box['y1'], pred['name'], color='black', backgroundcolor='white', fontsize=font_size)
            plt.axis('off')
            st.pyplot(fig, bbox_inches='tight', pad_inches=0)
