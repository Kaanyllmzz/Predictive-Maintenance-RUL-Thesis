# main_gui.py
import sys
import pandas as pd  # type: ignore
import joblib  # type: ignore
import os
import datetime
import numpy as np  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import subprocess
import tempfile
from PyQt5 import QtWidgets, QtGui, QtCore  # type: ignore

# Keras ve TensorFlow importları
import tensorflow as tf  # type: ignore
from tensorflow.keras.models import load_model  # type: ignore

# Modüllerin importları
from maintenance import maintenance_decision
from reporting import append_prediction_log, daily_report_to_excel, ensure_dirs
from shap_analysis import shap_summary_png, shap_local_png, load_sample_data
from constants import FilePaths, Messages, ColumnNames, Colors, MaintenanceStatus, DefaultThresholds
# =======================================================

MODELS = ["XGBoost", "LightGBM", "CatBoost", "LSTM"]
QUANTILES = [0.1, 0.5, 0.9] # p10, p50 (Medyan), p90
TIMESTEPS = 30 # LSTM sekans boyutu

# Sensörlerin min/max değerlerini Celsius'a dönüştürmek için Rankine'dan Celsius'a dönüşüm katsayıları
R_TO_C_OFFSET = 491.67
R_TO_C_FACTOR = 1.8
TEMPERATURE_SENSORS = [
    "sensor_measurement_2", 
    "sensor_measurement_3", 
    "sensor_measurement_4", 
    "sensor_measurement_20"
]

class PredictionApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Makine Kalan Ömür Tahmini (RUL) - Gelişmiş Analiz")
        self.setGeometry(200, 100, 700, 900)

        ensure_dirs()

        # Tüm modelleri ve scaler'ları yükle
        self.models = {}
        self.scalers = {}
        self.load_all_models()

        # Özellikleri oku
        with open(FilePaths.FEATURES_TXT, "r") as f:
            self.features = [line.strip() for line in f.readlines()]

        # Canlı akış için timer
        self.stream_timer = QtCore.QTimer()
        self.stream_timer.timeout.connect(self.update_from_stream)
        self.is_streaming = False
        self.last_row_count = 0
        
        # Ek model dosyasını kaldır (model.pkl yerine yeni dosyalar yüklenecek)
        if os.path.exists(FilePaths.MODEL_PKL):
            os.remove(FilePaths.MODEL_PKL)
        if os.path.exists(FilePaths.SCALER_PKL):
            os.remove(FilePaths.SCALER_PKL)


        self.init_ui()
    
    def load_all_models(self):
        """Tüm Quantile modellerini ve scaler'ları yükle"""
        
        all_loaded = True
        
        for model_name in MODELS:
            self.models[model_name] = {}
            # Scaler'ı yükle
            scaler_path = f"{model_name.lower()}_scaler.pkl"
            try:
                self.scalers[model_name] = joblib.load(scaler_path)
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, Messages.MODEL_LOAD_ERROR, f"{model_name} Scaler yüklenemedi:\n{str(e)}")
                all_loaded = False
                continue

            # Quantile modellerini yükle
            for q in QUANTILES:
                q_str = f"q{int(q*100)}"
                if model_name == "LSTM":
                    model_path = f"{model_name.lower()}_{q_str}_model.h5"
                    try:
                        # Keras modelini yüklerken custom_objects geçmek gerekiyor
                        self.models[model_name][q_str] = load_model(model_path, custom_objects={'loss': self.get_quantile_loss_func(q)})
                    except Exception as e:
                        print(f"LSTM {q_str} yüklenemedi: {e}")
                        self.models[model_name][q_str] = None
                        all_loaded = False
                else:
                    model_path = f"{model_name.lower()}_{q_str}_model.pkl"
                    try:
                        self.models[model_name][q_str] = joblib.load(model_path)
                    except Exception as e:
                        print(f"{model_name} {q_str} yüklenemedi: {e}")
                        self.models[model_name][q_str] = None
                        all_loaded = False
        
        if not all_loaded:
            QtWidgets.QMessageBox.critical(self, Messages.MODEL_LOAD_ERROR, "Tüm modeller yüklenemedi. Lütfen önce model_train.py'yi çalıştırın.")
            # sys.exit(1) # Hata olursa uygulamayı kapatmak yerine uyarı verelim

    def get_quantile_loss_func(self, q):
        """Keras model yükleme için Pinball Loss fonksiyonu (lambda yerine)"""
        def loss(y_true, y_pred):
            e = y_true - y_pred
            return tf.reduce_mean(tf.where(e >= 0, q * e, (q - 1) * e))
        return loss

    def init_ui(self):
        # ... (Arayüz oluşturma kodları - önceki koddan alınıp düzenlenmeli) ...
        # Bu kısım çok uzun olduğu için özetliyorum. Ana hatlar korunmalı ve yeni
        # model seçimi eklenmeli.

        layout = QtWidgets.QVBoxLayout()

        # Başlık
        title = QtWidgets.QLabel("📊 Makine Kalan Ömür Tahmini - Gelişmiş Analiz")
        title.setFont(QtGui.QFont("Arial", 16, QtGui.QFont.Bold))
        title.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(title)

        # Model Seçimi
        model_group = QtWidgets.QGroupBox("⚙️ Model Seçimi")
        model_layout = QtWidgets.QHBoxLayout()
        self.model_combo = QtWidgets.QComboBox()
        self.model_combo.addItems(MODELS)
        model_layout.addWidget(QtWidgets.QLabel("Tahmin Modeli:"))
        model_layout.addWidget(self.model_combo)
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)

        # Canlı Akış Kontrolleri
        stream_group = QtWidgets.QGroupBox("🔴 Canlı Veri Akışı")
        stream_layout = QtWidgets.QHBoxLayout()
        # ... (Önceki koddan start/stop butonlarını ekleyin) ...
        self.start_stream_btn = QtWidgets.QPushButton("Canlı Akış Başlat")
        self.start_stream_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        self.start_stream_btn.clicked.connect(self.start_stream)
        
        self.stop_stream_btn = QtWidgets.QPushButton("Canlı Akış Durdur")
        self.stop_stream_btn.setStyleSheet("background-color: #f44336; color: white; font-weight: bold;")
        self.stop_stream_btn.clicked.connect(self.stop_stream)
        self.stop_stream_btn.setEnabled(False)
        
        stream_layout.addWidget(self.start_stream_btn)
        stream_layout.addWidget(self.stop_stream_btn)
        stream_group.setLayout(stream_layout)
        layout.addWidget(stream_group)
        
        # Sensör Değerleri Grubu
        sensor_group = QtWidgets.QGroupBox("🌡️ Ana Sensörler")
        sensor_layout = QtWidgets.QFormLayout()

        self.sicaklik_input = QtWidgets.QLineEdit()
        self.sicaklik_input.setPlaceholderText("Sıcaklık değeri (°C)")
        sensor_layout.addRow("Sıcaklık (°C):", self.sicaklik_input)

        self.titresim_input = QtWidgets.QLineEdit()
        self.titresim_input.setPlaceholderText("Titreşim değeri")
        sensor_layout.addRow("Titreşim:", self.titresim_input)

        self.tork_input = QtWidgets.QLineEdit()
        self.tork_input.setPlaceholderText("Tork değeri")
        sensor_layout.addRow("Tork:", self.tork_input)

        sensor_group.setLayout(sensor_layout)
        layout.addWidget(sensor_group)

        # Ana İşlem Butonları
        main_buttons_group = QtWidgets.QGroupBox("🔧 Ana İşlemler")
        main_buttons_layout = QtWidgets.QVBoxLayout()

        self.predict_btn = QtWidgets.QPushButton("Tahmin Yap (Quantile Analizi ile)")
        self.predict_btn.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold; font-size: 14px; padding: 10px;")
        self.predict_btn.clicked.connect(self.make_prediction)
        main_buttons_layout.addWidget(self.predict_btn)
        
        # Rapor butonu
        self.report_btn = QtWidgets.QPushButton("Rapor İndir (Bugün)")
        self.report_btn.setStyleSheet("background-color: #FF9800; color: white; font-weight: bold;")
        self.report_btn.clicked.connect(self.download_daily_report)
        main_buttons_layout.addWidget(self.report_btn)

        main_buttons_group.setLayout(main_buttons_layout)
        layout.addWidget(main_buttons_group)


        # Analiz Butonları
        analysis_group = QtWidgets.QGroupBox("📈 Analiz Araçları")
        analysis_layout = QtWidgets.QHBoxLayout()

        self.lime_btn = QtWidgets.QPushButton("LIME Açıklaması (Lokal)")
        self.lime_btn.setStyleSheet("background-color: #9C27B0; color: white; font-weight: bold;")
        self.lime_btn.clicked.connect(self.show_lime_explanation)

        self.shap_summary_btn = QtWidgets.QPushButton("SHAP Özet (Global)")
        self.shap_summary_btn.setStyleSheet("background-color: #607D8B; color: white; font-weight: bold;")
        self.shap_summary_btn.clicked.connect(self.show_shap_summary)

        analysis_layout.addWidget(self.lime_btn)
        analysis_layout.addWidget(self.shap_summary_btn)
        analysis_group.setLayout(analysis_layout)
        layout.addWidget(analysis_group)

        # Sonuç Gösterimi
        result_group = QtWidgets.QGroupBox("📊 Tahmin ve Belirsizlik Sonuçları")
        result_layout = QtWidgets.QVBoxLayout()

        self.result_label = QtWidgets.QLabel("Tahmin sonucu burada görünecek")
        self.result_label.setFont(QtGui.QFont("Arial", 12, QtGui.QFont.Bold))
        self.result_label.setAlignment(QtCore.Qt.AlignCenter)
        self.result_label.setStyleSheet("padding: 20px; border: 2px solid #ddd; border-radius: 10px;")
        result_layout.addWidget(self.result_label)
        
        # Belirsizlik grafiği için yer tutucu
        self.graph_label = QtWidgets.QLabel("Belirsizlik Grafiği burada görünecek")
        self.graph_label.setAlignment(QtCore.Qt.AlignCenter)
        self.graph_label.setFixedSize(650, 300) # Sabit boyut
        result_layout.addWidget(self.graph_label)


        result_group.setLayout(result_layout)
        layout.addWidget(result_group)


        # Durum Çubuğu
        self.status_label = QtWidgets.QLabel("Hazır")
        self.status_label.setStyleSheet("color: #666; font-style: italic;")
        layout.addWidget(self.status_label)

        self.setLayout(layout)
    
    # ... (start_stream, stop_stream, update_from_stream, download_daily_report metotları önceki koddan aynen korunmalı) ...
    
    def start_stream(self):
        """Canlı akış başlat"""
        # GÜNCELLEME: Constants'ten STREAM_CSV al
        STREAM_CSV = FilePaths.STREAM_CSV
        if not os.path.exists(STREAM_CSV):
            QtWidgets.QMessageBox.warning(self, "Dosya Bulunamadı", 
                                        f"Akış dosyası bulunamadı: {STREAM_CSV}\n\nÖnce sim_stream.py ile veri üretmeyi başlatın.")
            return

        self.is_streaming = True
        self.stream_timer.start(1000)  # 1 saniyede bir
        
        self.start_stream_btn.setEnabled(False)
        self.stop_stream_btn.setEnabled(True)
        
        self.status_label.setText("🔴 Canlı akış aktif")
        self.status_label.setStyleSheet("color: #4CAF50; font-weight: bold;")

    def stop_stream(self):
        """Canlı akış durdur"""
        self.is_streaming = False
        self.stream_timer.stop()
        
        self.start_stream_btn.setEnabled(True)
        self.stop_stream_btn.setEnabled(False)
        
        self.status_label.setText("⏹️ Canlı akış durduruldu")
        self.status_label.setStyleSheet("color: #666; font-style: italic;")

    def update_from_stream(self):
        """Stream dosyasından son satırı oku ve formu güncelle"""
        # GÜNCELLEME: Constants'ten STREAM_CSV al
        STREAM_CSV = FilePaths.STREAM_CSV
        try:
            if not os.path.exists(STREAM_CSV):
                self.stop_stream()
                return

            df = pd.read_csv(STREAM_CSV)
            
            if len(df) == 0:
                return
                
            # Yeni satır var mı kontrol et
            if len(df) > self.last_row_count:
                self.last_row_count = len(df)
                
                # Son satırı al
                last_row = df.iloc[-1]
                
                # Form alanlarını güncelle
                self.sicaklik_input.setText(str(round(last_row[ColumnNames.SICAKLIK], 2)))
                self.titresim_input.setText(str(round(last_row[ColumnNames.TITRESIM], 3)))
                self.tork_input.setText(str(round(last_row[ColumnNames.TORK], 2)))
                
                # Tahmin yap
                self.make_prediction(log_only=True)
                
                # Durum güncelle
                self.status_label.setText(f"🔄 Son güncelleme ve otomatik tahmin: {last_row[ColumnNames.TIMESTAMP]}")

        except Exception as e:
            self.status_label.setText(f"⚠️ Akış hatası: {str(e)}")

    def download_daily_report(self):
        """Günlük rapor indir"""
        try:
            excel_file = daily_report_to_excel()
            
            QtWidgets.QMessageBox.information(self, "Rapor Oluşturuldu", 
                                            f"Günlük Excel raporu başarıyla oluşturuldu!\n\nDosya: {excel_file}")
            
            self.status_label.setText(f"📊 Rapor oluşturuldu: {excel_file}")

        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Rapor Hatası", 
                                        f"Günlük rapor oluşturulamadı:\n{str(e)}")
                                        
    def get_raw_input(self):
        """Kullanıcı girdilerini alıp DataFrame'e dönüştürür."""
        
        sicaklik = float(self.sicaklik_input.text() or "0")
        titresim = float(self.titresim_input.text() or "0")
        tork = float(self.tork_input.text() or "0")
        
        if sicaklik == 0 and titresim == 0 and tork == 0:
            raise ValueError(Messages.ENTER_SENSOR_VALUES)

        # GERÇEK SENSÖR DEĞERLERİNE DÖNÜŞÜM MANTIKLARI
        # Bu kısım, train_rul.csv'deki 10 sensörün gerçek dünyadaki 3 ana değere 
        # (sıcaklık, titreşim, tork) bağımlı olduğunu simüle eder.
        input_data = {
            'sensor_measurement_11': [47.5],
            'sensor_measurement_12': [521.0 + sicaklik/10],
            'sensor_measurement_4': [(1400.0 + sicaklik) * R_TO_C_FACTOR + R_TO_C_OFFSET], # °C -> Rankine dönüşümü
            'sensor_measurement_7': [553.0],
            'sensor_measurement_15': [8.4 + titresim],
            'sensor_measurement_9': [9050.0 + tork*10],
            'sensor_measurement_21': [23.3],
            'sensor_measurement_20': [(38.9 + sicaklik/5) * R_TO_C_FACTOR + R_TO_C_OFFSET],
            'sensor_measurement_2': [(642.0 + sicaklik/5) * R_TO_C_FACTOR + R_TO_C_OFFSET],
            'sensor_measurement_3': [(1585.0 + sicaklik*2) * R_TO_C_FACTOR + R_TO_C_OFFSET]
        }
        
        # Sadece modelde kullanılan 10 özelliği al
        raw_df = pd.DataFrame(input_data)[self.features]
        return raw_df, sicaklik, titresim, tork

    def make_prediction(self, log_only=False):
        """Tahmin yap (p10, p50, p90) ve sonucu göster"""
        selected_model_name = self.model_combo.currentText()
        scaler = self.scalers.get(selected_model_name)
        
        if not scaler:
            self.result_label.setText(f"Hata: {selected_model_name} modeli için Scaler yüklenemedi.")
            return

        try:
            raw_df, sicaklik, titresim, tork = self.get_raw_input()
        except ValueError as e:
            if not log_only: # Sadece butona basıldığında uyarı ver
                QtWidgets.QMessageBox.warning(self, Messages.MISSING_DATA, str(e))
            return
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Veri Hatası", f"Girdi verisi işlenemedi:\n{str(e)}")
            return

        # Veriyi ölçekle
        X_scaled = scaler.transform(raw_df)
        
        quantile_predictions = {}
        for q in QUANTILES:
            q_str = f"q{int(q*100)}"
            model = self.models[selected_model_name].get(q_str)
            
            if not model:
                QtWidgets.QMessageBox.warning(self, Messages.MODEL_LOAD_ERROR, f"Hata: {selected_model_name} {q_str} modeli yüklenemedi.")
                return

            if selected_model_name == "LSTM":
                # LSTM için 3D input formatı
                X_seq = np.zeros((1, TIMESTEPS, X_scaled.shape[1]))
                X_seq[0, -1, :] = X_scaled[0, :]
                pred = model.predict(X_seq, verbose=0)[0][0]
            else:
                pred = model.predict(X_scaled)[0]
            
            quantile_predictions[q] = pred
        
        rul_p10 = quantile_predictions.get(0.1, np.nan)
        rul_p50 = quantile_predictions.get(0.5, np.nan)
        rul_p90 = quantile_predictions.get(0.9, np.nan)
        
        # Ana tahmin p50 (Medyan) alınır
        rul = rul_p50
        maintenance_info = maintenance_decision(rul)
        status = maintenance_info["status"]
        message = maintenance_info["message"]
        color = maintenance_info["color"]
        
        # Sonucu göster
        result_text = f"📌 Ana Tahmin (p50 / Medyan): {rul_p50:.2f} döngü\n"
        result_text += f"Belirsizlik Aralığı: [{rul_p10:.2f} – {rul_p90:.2f}] (p10-p90)\n"
        result_text += f"🔧 Bakım Durumu: {status}\n"
        result_text += f"💬 {message}"

        self.result_label.setText(result_text)
        self.result_label.setStyleSheet(f"color: {color}; font-weight: bold; padding: 20px; border: 2px solid {color}; border-radius: 10px;")

        # Belirsizlik grafiği oluştur ve göster
        self.create_uncertainty_graph(rul_p10, rul_p50, rul_p90, selected_model_name, color)

        # Loglama
        if log_only or self.is_streaming:
            log_data = {
                ColumnNames.TIMESTAMP: datetime.datetime.now().isoformat(),
                ColumnNames.SICAKLIK: sicaklik,
                ColumnNames.TITRESIM: titresim,
                ColumnNames.TORK: tork,
                ColumnNames.RUL: rul,
                "RUL_p10": rul_p10,
                "RUL_p90": rul_p90,
                ColumnNames.STATUS: status
            }
            append_prediction_log(log_data)
            self.status_label.setText(f"✅ Tahmin tamamlandı ve loglandı ({selected_model_name})")
        else:
             self.status_label.setText(f"✅ Tahmin tamamlandı ({selected_model_name})")


    def create_uncertainty_graph(self, p10, p50, p90, model_name, color):
        """Tahmin belirsizliğini gösteren basit bir grafik oluşturur"""
        fig, ax = plt.subplots(figsize=(6.5, 3))
        
        # Belirsizlik aralığı (p10-p90)
        ax.barh([0], [p90 - p10], left=[p10], color=color, alpha=0.3, label='80% Güven Aralığı')
        
        # P50 tahmini (nokta)
        ax.plot(p50, 0, marker='o', markersize=10, color=color, label='Medyan Tahmin (p50)')
        
        # Eşikler
        critical_th = DefaultThresholds.CRITICAL
        planned_th = DefaultThresholds.PLANNED
        
        ax.axvline(critical_th, color=Colors.CRITICAL, linestyle='--', label=f'Kritik Eşik ({critical_th})')
        ax.axvline(planned_th, color=Colors.PLANNED, linestyle=':', label=f'Planlı Eşik ({planned_th})')

        # Etiketler ve Başlık
        ax.set_yticks([])
        ax.set_xlabel("Kalan Ömür (RUL) Döngü Sayısı")
        ax.set_title(f"RUL Tahmin Belirsizliği - {model_name}")
        
        # Değerleri grafiğe yaz
        ax.text(p50, 0.1, f"p50: {p50:.1f}", ha='center', va='bottom', color='black', fontsize=9, fontweight='bold')
        ax.text(p10, -0.15, f"p10: {p10:.1f}", ha='center', va='top', color='gray', fontsize=8)
        ax.text(p90, -0.15, f"p90: {p90:.1f}", ha='center', va='top', color='gray', fontsize=8)

        ax.legend(loc='upper right', fontsize=8)
        ax.grid(axis='x', linestyle='--')
        plt.tight_layout()

        # Grafiği geçici bir dosyaya kaydet
        temp_file = os.path.join(tempfile.gettempdir(), f"rul_quantile_graph_{model_name}.png")
        plt.savefig(temp_file, dpi=100)
        plt.close(fig)
        
        # Resmi QLabel'a yükle
        pixmap = QtGui.QPixmap(temp_file)
        self.graph_label.setPixmap(pixmap.scaled(self.graph_label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

    def show_lime_explanation(self):
        """LIME açıklaması oluştur ve göster"""
        # Sadece ağaç tabanlı modeller için LIME (LSTM için LIME farklı kurgu gerektirir)
        selected_model_name = self.model_combo.currentText()
        if selected_model_name not in ["XGBoost", "LightGBM", "CatBoost"]:
            QtWidgets.QMessageBox.warning(self, "LIME Hatası", "LIME açıklaması sadece ağaç tabanlı modeller için uygundur.")
            return

        # ... (Önceki main_gui.py'den LIME kodu) ...
        try:
            # Form değerlerini al ve DF oluştur
            raw_df, _, _, _ = self.get_raw_input()

            self.status_label.setText("🔍 LIME açıklaması oluşturuluyor...")
            
            # LIME açıklaması oluştur
            from lime_explain import explain_instance, open_html_in_browser
            html_file = explain_instance(
                self.models[selected_model_name]['q50'], # p50 modelini kullan
                self.scalers[selected_model_name], 
                raw_df, 
                self.features,
                "reports/lime_explanation.html"
            )
            
            open_html_in_browser(html_file)
            
            QtWidgets.QMessageBox.information(self, "LIME Açıklaması", 
                                            f"LIME açıklaması başarıyla oluşturuldu ve tarayıcıda açıldı:\n{html_file}")
            
            self.status_label.setText("✅ LIME açıklaması oluşturuldu")

        except ValueError:
            QtWidgets.QMessageBox.warning(self, Messages.INVALID_DATA, Messages.ENTER_VALID_NUMBERS)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "LIME Hatası", 
                                         f"LIME açıklaması oluşturulamadı:\n{str(e)}")
            self.status_label.setText("❌ LIME hatası")

    def show_shap_summary(self):
        """SHAP Özet (Global) analizi oluştur"""
        selected_model_name = self.model_combo.currentText()
        if selected_model_name not in ["XGBoost", "LightGBM", "CatBoost"]:
            QtWidgets.QMessageBox.warning(self, "SHAP Hatası", "SHAP analizi sadece ağaç tabanlı modeller için uygundur.")
            return
        
        try:
            self.status_label.setText("🔍 SHAP özet analizi başlıyor...")
            
            # Örnek veri yükle (300 satır)
            x_sample = load_sample_data(csv_path=FilePaths.TRAIN_RUL_CSV, features_path=FilePaths.FEATURES_TXT, sample_size=300)
            
            # Veriyi ölçekle
            scaler = self.scalers[selected_model_name]
            x_scaled = pd.DataFrame(
                scaler.transform(x_sample), 
                columns=x_sample.columns,
                index=x_sample.index
            )
            
            # SHAP summary oluştur
            # p50 modelini kullan
            model = self.models[selected_model_name]['q50'] 
            summary_path = shap_summary_png(model, x_scaled, out_path=f"reports/{selected_model_name.lower()}_shap_summary.png")
            
            self.open_image_file(summary_path)
            
            QtWidgets.QMessageBox.information(self, "SHAP Özet Analizi", 
                                            f"SHAP özet analizi başarıyla oluşturuldu!\n\nDosya: {summary_path}")
            
            self.status_label.setText("✅ SHAP özet analizi tamamlandı")

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "SHAP Özet Hatası", 
                                         f"SHAP özet analizi oluşturulamadı:\n{str(e)}")
            self.status_label.setText("❌ SHAP özet hatası")

    # ... (open_image_file ve closeEvent metotları önceki koddan aynen korunmalı) ...
    def open_image_file(self, image_path):
        """Görsel dosyasını sistem görüntüleyicisiyle aç"""
        try:
            import sys
            if sys.platform.startswith("darwin"):  # macOS
                subprocess.run(["open", image_path])
            elif sys.platform.startswith("win"):  # Windows
                os.startfile(image_path)
            elif sys.platform.startswith("linux"):  # Linux
                subprocess.run(["xdg-open", image_path])
            else:
                print(f"Görsel dosyası: {image_path}")
                
        except Exception as e:
            print(f"⚠️ Görsel dosyası açılamadı: {e}")

    def closeEvent(self, event):
        """Uygulama kapatılırken canlı akışı durdur"""
        if self.is_streaming:
            self.stop_stream()
        event.accept()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = PredictionApp()
    window.show()
    sys.exit(app.exec_())