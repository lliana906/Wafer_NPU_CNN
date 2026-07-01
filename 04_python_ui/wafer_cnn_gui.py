"""
wafer_gui.py — Wafer Defect Classifier PyQt5 GUI
PC ↔ Zybo Z7-20 NPU

실제 보드 펌웨어(main.c)가 쓰는 프로토콜에 맞춤:

송신 프로토콜:
  0xAA(시작 바이트) + 2304바이트 (48×48, row-major, 활성값 스케일 적용된 값)

수신 프로토콜:
  결과 1바이트(클래스 0~8) + 0x55(종료 바이트)  -  총 2바이트, 텍스트 아님

입력 포맷 : 원본 이미지를 48×48로 리사이즈 후, 0~127 범위(활성값 스케일 x4)로
            변환한 uint8. 보드는 받은 바이트를 그대로 부호 있는 8bit 활성값으로
            쓰기 때문에, 0~255 원본 그레이스케일을 그대로 보내면 큰 값이 음수로
            뒤집혀 추론이 깨진다 (wafer_npu_gui.py의 preprocess_to_bytes와 동일한
            변환을 여기서도 그대로 적용).
보드측    : 리사이즈나 양자화를 하지 않는다 - PC가 보낸 2304바이트를 그대로 씀
Baud rate : 115200 (보드 내장 USB-UART, PS UART1)
"""

import sys
import time
import serial
import serial.tools.list_ports
import numpy as np
from PIL import Image

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLabel, QFileDialog,
    QComboBox, QTextEdit, QGroupBox, QProgressBar, QDoubleSpinBox
)
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import Qt, QThread, pyqtSignal

# ── 설정 ──────────────────────────────────────────────────────────
BAUD_RATE = 115200   # PS UART(보드 내장 USB 단일 포트) 기준, 커스텀 UART IP 미사용
DEFAULT_TIMEOUT_S = 30.0
IMG_W      = 48        # 보드가 실제로 받는 고정 크기 (main.c 기준, 리사이즈는 PC가 함)
IMG_H      = 48
IMG_PIXELS = IMG_W * IMG_H
START_BYTE = 0xAA
END_BYTE   = 0x55
INPUT_SCALE = 4.0       # = 2^ACT_FRAC_BITS (ACT_FRAC_BITS=2, quant_meta.json 기준)
MAX_UPLOAD_W = 1024      # 업로드 원본 이미지 크기 상한(용량 방지용, 어차피 48×48로 줄임)
MAX_UPLOAD_H = 1024

CLASS_NAMES = [
    "Center", "Donut", "Edge-Loc", "Edge-Ring",
    "Loc", "Near-full", "Random", "Scratch", "none"
]
CLASS_COLORS = [
    "#E74C3C", "#9B59B6", "#3498DB", "#1ABC9C",
    "#F39C12", "#E67E22", "#27AE60", "#2980B9", "#95A5A6"
]


# ── 이미지 → binary 변환 ────────────────────────────────────────────
def image_to_bin(path: str):
    """
    원본 이미지를 열어 48x48로 리사이즈하고, RTL이 그대로 읽는 활성값 스케일
    (0~127, x4)로 변환한다.

    보드는 IMG_DATA로 받은 바이트를 그 어떤 변환도 없이 곧바로 부호 있는 8bit
    활성값으로 사용한다(cnn_core_fsm의 conv1 입력). 그래서 여기서 이미 최종
    스케일까지 다 맞춰서 보내야 한다 - wafer_npu_gui.py의 preprocess_to_bytes와
    동일한 규칙(activation_format.frac_bits=2, quant_meta.json)이다.

    반환: (bin_bytes, orig_w, orig_h) - orig_w/h는 화면 표시용 원본 크기.
    """
    img = Image.open(path).convert('L')
    orig_w, orig_h = img.size
    if orig_w > MAX_UPLOAD_W or orig_h > MAX_UPLOAD_H:
        raise ValueError(f"이미지가 너무 큽니다 ({orig_w}×{orig_h}). 최대 {MAX_UPLOAD_W}×{MAX_UPLOAD_H}")

    resized = img.resize((IMG_W, IMG_H), Image.BILINEAR)
    arr = np.array(resized, dtype=np.float32)  # 0~255

    byte_vals = np.clip(np.round((arr / 255.0) * INPUT_SCALE), 0, 127).astype(np.uint8)
    return byte_vals.tobytes(), orig_w, orig_h


# ── UART 통신 워커 ──────────────────────────────────────────────────
class UartWorker(QThread):
    sig_result   = pyqtSignal(int, int)  # (pred_label, elapsed_us) - elapsed_us는 PC측 왕복 측정값
    sig_error    = pyqtSignal(str)
    sig_log      = pyqtSignal(str)
    sig_progress = pyqtSignal(int)       # 전송 진행률 0~100

    def __init__(self, ser, bin_bytes, timeout_sec):
        super().__init__()
        self.ser        = ser   # 이미 연결되어 있는 serial.Serial 객체 (여기서 열고 닫지 않음)
        self.bin_bytes  = bin_bytes
        self.timeout_sec = timeout_sec

    def run(self):
        try:
            ser = self.ser
            ser.timeout = self.timeout_sec
            ser.reset_input_buffer()
            ser.reset_output_buffer()
            t0 = time.time()

            # ① 시작 바이트 + 고정 2304바이트 픽셀 전송 (main.c와 동일한 프로토콜)
            ser.write(bytes([START_BYTE]))
            self.sig_log.emit(f"전송 시작: 0xAA + {len(self.bin_bytes):,} bytes")

            CHUNK = 512
            total = len(self.bin_bytes)
            sent  = 0
            while sent < total:
                end = min(sent + CHUNK, total)
                ser.write(self.bin_bytes[sent:end])
                sent = end
                self.sig_progress.emit(int(sent / total * 100))
            ser.flush()

            self.sig_log.emit(f"전송 완료 ({total:,} bytes) | 보드 응답 대기 중...")
            self.sig_progress.emit(100)

            # ② 결과 수신: 클래스 1바이트 + 0x55 - 총 2바이트, 텍스트가 아님
            resp = ser.read(2)
            elapsed_us = int((time.time() - t0) * 1_000_000)

            if len(resp) < 2:
                self.sig_error.emit(
                    f"응답 타임아웃: {self.timeout_sec:.1f}초 내에 {len(resp)}/2 바이트만 수신했습니다."
                )
                return

            pred, end_byte = resp[0], resp[1]
            self.sig_log.emit(f"수신: 0x{pred:02X} 0x{end_byte:02X}")

            if end_byte != END_BYTE:
                self.sig_error.emit(f"프로토콜 오류: 종료 바이트가 0x{END_BYTE:02X}가 아니라 0x{end_byte:02X}입니다.")
                return

            if pred < 0 or pred >= len(CLASS_NAMES):
                self.sig_error.emit(f"잘못된 pred: {pred}")
                return

            self.sig_result.emit(pred, elapsed_us)

        except serial.SerialException as e:
            self.sig_error.emit(f"포트 오류: {e}")
        except Exception as e:
            self.sig_error.emit(f"오류: {e}")


# ── 메인 윈도우 ──────────────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Wafer Defect Classifier — Zybo Z7-20 NPU")
        self.setMinimumSize(860, 620)
        self.img_path  = None
        self.bin_bytes = None
        self.img_w     = 0
        self.img_h     = 0
        self.worker    = None
        self.ser       = None
        self._build_ui()
        self._refresh_ports()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setSpacing(12)
        root.setContentsMargins(12, 12, 12, 12)

        # ── 좌측 패널 ──────────────────────────────────────────────
        left = QVBoxLayout()
        left.setSpacing(8)

        img_grp = QGroupBox("입력 이미지")
        ig_lay  = QVBoxLayout(img_grp)

        self.img_label = QLabel("이미지를 선택하세요")
        self.img_label.setAlignment(Qt.AlignCenter)
        self.img_label.setFixedSize(320, 320)
        self.img_label.setStyleSheet(
            "border:2px dashed #555; background:#1a1a1a; color:#666;"
        )
        ig_lay.addWidget(self.img_label, alignment=Qt.AlignCenter)

        self.img_info = QLabel("")
        self.img_info.setAlignment(Qt.AlignCenter)
        self.img_info.setStyleSheet("color:#888; font-size:11px;")
        ig_lay.addWidget(self.img_info)

        btn_open = QPushButton("📂  이미지 열기")
        btn_open.setFixedHeight(36)
        btn_open.clicked.connect(self._open_image)
        ig_lay.addWidget(btn_open)
        left.addWidget(img_grp)

        port_grp = QGroupBox("UART 연결  (115200 baud)")
        pl = QHBoxLayout(port_grp)

        self.port_combo = QComboBox()
        self.port_combo.setMinimumWidth(130)
        pl.addWidget(self.port_combo)

        btn_rf = QPushButton("새로고침")
        btn_rf.setObjectName("secondary")
        btn_rf.clicked.connect(self._refresh_ports)
        pl.addWidget(btn_rf)

        self.btn_connect = QPushButton("연결")
        self.btn_connect.clicked.connect(self._toggle_connection)
        pl.addWidget(self.btn_connect)

        self.status_label = QLabel("● 연결 안 됨")
        self.status_label.setStyleSheet("color:#888; font-weight:600;")
        pl.addWidget(self.status_label)

        pl.addStretch()

        pl.addWidget(QLabel("응답 타임아웃(초):"))
        self.timeout_spin = QDoubleSpinBox()
        self.timeout_spin.setRange(1.0, 300.0)
        self.timeout_spin.setSingleStep(5.0)
        self.timeout_spin.setValue(DEFAULT_TIMEOUT_S)
        self.timeout_spin.setFixedWidth(70)
        pl.addWidget(self.timeout_spin)

        left.addWidget(port_grp)

        self.btn_run = QPushButton("▶  추론 실행")
        self.btn_run.setFixedHeight(46)
        self.btn_run.setEnabled(False)
        self.btn_run.setStyleSheet(
            "QPushButton{background:#1abc9c;color:white;font-size:15px;"
            "border-radius:6px;font-weight:bold;}"
            "QPushButton:disabled{background:#3a3a3a;color:#666;}"
            "QPushButton:hover{background:#16a085;}"
        )
        self.btn_run.clicked.connect(self._run_inference)
        left.addWidget(self.btn_run)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setVisible(False)
        self.progress.setFixedHeight(8)
        self.progress.setTextVisible(False)
        self.progress.setStyleSheet(
            "QProgressBar{border:none;background:#333;border-radius:4px;}"
            "QProgressBar::chunk{background:#1abc9c;border-radius:4px;}"
        )
        left.addWidget(self.progress)
        left.addStretch()
        root.addLayout(left, 4)

        # ── 우측 패널 ──────────────────────────────────────────────
        right = QVBoxLayout()
        right.setSpacing(8)

        res_grp = QGroupBox("추론 결과")
        rl = QVBoxLayout(res_grp)

        self.lbl_class = QLabel("—")
        self.lbl_class.setAlignment(Qt.AlignCenter)
        f = QFont(); f.setPointSize(30); f.setBold(True)
        self.lbl_class.setFont(f)
        self.lbl_class.setStyleSheet("color:#1abc9c;")
        rl.addWidget(self.lbl_class)

        self.lbl_time = QLabel("")
        self.lbl_time.setAlignment(Qt.AlignCenter)
        self.lbl_time.setStyleSheet("color:#888; font-size:13px;")
        rl.addWidget(self.lbl_time)

        self.bars = {}
        grid = QGridLayout()
        grid.setVerticalSpacing(4)
        for i, (name, color) in enumerate(zip(CLASS_NAMES, CLASS_COLORS)):
            lbl = QLabel(name)
            lbl.setFixedWidth(78)
            lbl.setStyleSheet("color:#ccc;font-size:11px;")
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setFixedHeight(14)
            bar.setTextVisible(False)
            bar.setStyleSheet(
                f"QProgressBar{{border:none;background:#2a2a2a;border-radius:3px;}}"
                f"QProgressBar::chunk{{background:{color};border-radius:3px;}}"
            )
            grid.addWidget(lbl, i, 0)
            grid.addWidget(bar, i, 1)
            self.bars[i] = bar
        rl.addLayout(grid)
        right.addWidget(res_grp)

        log_grp = QGroupBox("로그")
        ll = QVBoxLayout(log_grp)
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMaximumHeight(160)
        self.log_box.setStyleSheet(
            "background:#0d0d0d;color:#0f0;"
            "font-family:monospace;font-size:11px;"
        )
        ll.addWidget(self.log_box)
        right.addWidget(log_grp)
        root.addLayout(right, 3)

        self.setStyleSheet("""
            QMainWindow,QWidget{background:#252525;color:#ddd;}
            QGroupBox{border:1px solid #444;border-radius:6px;
                      margin-top:8px;padding-top:8px;color:#eee;}
            QGroupBox::title{subcontrol-origin:margin;left:8px;}
            QComboBox,QDoubleSpinBox{background:#333;color:#ddd;border:1px solid #555;
                      padding:3px;border-radius:4px;}
            QPushButton{background:#3a3a3a;color:#ddd;
                        border:1px solid #555;border-radius:4px;padding:4px 8px;}
            QPushButton:hover{background:#4a4a4a;}
            QPushButton#secondary{background:transparent;color:#1abc9c;
                        border:1px solid #1abc9c;}
            QPushButton#secondary:hover{background:#173832;}
        """)

        self.btn_connect.setStyleSheet(
            "QPushButton{background:#1abc9c;color:white;border:none;"
            "border-radius:4px;padding:6px 14px;font-weight:bold;}"
            "QPushButton:hover{background:#16a085;}"
        )

    def _refresh_ports(self):
        self.port_combo.clear()
        for p in serial.tools.list_ports.comports():
            self.port_combo.addItem(p.device)
        if self.port_combo.count() == 0:
            self.port_combo.addItem("/dev/ttyUSB1")

    def _toggle_connection(self):
        if self.ser is not None and self.ser.is_open:
            self._disconnect_serial()
        else:
            self._connect_serial()

    def _connect_serial(self):
        port = self.port_combo.currentText().strip()
        if not port:
            self._log("[오류] 연결할 포트를 선택하세요")
            return
        try:
            self.ser = serial.Serial(port, BAUD_RATE, timeout=self.timeout_spin.value())
            self.btn_connect.setText("연결 해제")
            self.status_label.setText("● 연결됨")
            self.status_label.setStyleSheet("color:#1abc9c; font-weight:600;")
            self._log(f"[연결] {port} @ {BAUD_RATE:,} baud 연결 성공")
        except serial.SerialException as e:
            self.ser = None
            self._log(f"[오류] 연결 실패: {e}")
        self._update_run_button_state()

    def _disconnect_serial(self):
        try:
            if self.ser:
                self.ser.close()
        finally:
            self.ser = None
            self.btn_connect.setText("연결")
            self.status_label.setText("● 연결 안 됨")
            self.status_label.setStyleSheet("color:#888; font-weight:600;")
            self._log("[연결] 포트 연결을 해제했습니다")
        self._update_run_button_state()

    def _update_run_button_state(self):
        connected = self.ser is not None and self.ser.is_open
        busy = self.worker is not None and self.worker.isRunning()
        self.btn_run.setEnabled(connected and self.bin_bytes is not None and not busy)

    def _open_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "이미지 선택", "",
            "Images (*.png *.bmp *.jpg *.jpeg *.tif *.tiff)"
        )
        if not path:
            return

        try:
            self.bin_bytes, self.img_w, self.img_h = image_to_bin(path)
        except Exception as e:
            self._log(f"[오류] {e}")
            return

        self.img_path = path

        pil = Image.open(path).convert('RGB')
        pil.thumbnail((310, 310))
        qi = QImage(pil.tobytes(), pil.width, pil.height,
                    pil.width * 3, QImage.Format_RGB888)
        self.img_label.setPixmap(
            QPixmap.fromImage(qi).scaled(
                310, 310, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        )

        tx_sec = len(self.bin_bytes) / (BAUD_RATE / 10)
        self.img_info.setText(
            f"원본 {self.img_w}×{self.img_h} px → 48×48로 리사이즈  |  "
            f"전송 {len(self.bin_bytes):,} bytes (+시작바이트 1)  |  "
            f"전송 약 {tx_sec:.2f}초"
        )
        self._update_run_button_state()
        self._log(
            f"[로드] {path.split('/')[-1]} "
            f"(원본 {self.img_w}×{self.img_h} → 48×48, {len(self.bin_bytes):,}B)"
        )
        self._reset_result()

    def _run_inference(self):
        if self.ser is None or not self.ser.is_open:
            self._log("[오류] 먼저 보드에 연결하세요")
            return

        self.btn_run.setEnabled(False)
        self.progress.setValue(0)
        self.progress.setVisible(True)
        self._reset_result()
        self._log("[추론] 이미지 전송 시작")

        self.worker = UartWorker(
            self.ser, self.bin_bytes, self.timeout_spin.value(),
        )
        self.worker.sig_result.connect(self._on_result)
        self.worker.sig_error.connect(self._on_error)
        self.worker.sig_log.connect(self._log)
        self.worker.sig_progress.connect(self.progress.setValue)
        self.worker.finished.connect(self._on_done)
        self.worker.start()

    def _on_result(self, pred: int, elapsed_us: int):
        pred  = max(0, min(pred, len(CLASS_NAMES) - 1))
        name  = CLASS_NAMES[pred]
        color = CLASS_COLORS[pred]

        self.lbl_class.setText(name)
        self.lbl_class.setStyleSheet(f"color:{color};font-weight:bold;")

        ms = elapsed_us / 1000.0
        self.lbl_time.setText(f"총 소요시간: {ms:.1f} ms  (전송+추론+응답 왕복, PC 측 측정)")

        for i in range(len(CLASS_NAMES)):
            self.bars[i].setValue(100 if i == pred else 0)

        self._log(f"[결과] {name}")

    def _on_error(self, msg: str):
        self.lbl_class.setText("오류")
        self.lbl_class.setStyleSheet("color:#e74c3c;")
        self.lbl_time.setText(msg)
        self._log(f"[오류] {msg}")

    def _on_done(self):
        self.progress.setVisible(False)
        self._update_run_button_state()

    def _reset_result(self):
        self.lbl_class.setText("—")
        self.lbl_class.setStyleSheet("color:#1abc9c;")
        self.lbl_time.setText("")
        for bar in self.bars.values():
            bar.setValue(0)

    def _log(self, msg: str):
        self.log_box.append(msg)
        sb = self.log_box.verticalScrollBar()
        sb.setValue(sb.maximum())

    def closeEvent(self, event):
        if self.worker is not None and self.worker.isRunning():
            self.worker.wait(2000)
        if self.ser is not None and self.ser.is_open:
            self.ser.close()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())