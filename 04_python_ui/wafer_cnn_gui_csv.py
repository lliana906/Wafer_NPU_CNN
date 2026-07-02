"""
wafer_cnn_gui_csv.py — Wafer Defect Classifier PyQt5 GUI (통합본)
PC ↔ Zybo Z7-20 NPU

원래 wafer_cnn_gui_many.py(BatchWindow 기반 클래스)와
wafer_cnn_gui_add.py(FinalWindow/FinalAddWindow 확장 클래스)로
나뉘어 있던 두 파일을 하나로 병합했습니다.
FinalAddWindow가 FinalWindow를 상속하고, FinalWindow가 BatchWindow를
상속하는 구조라 굳이 파일을 분리해 둘 필요가 없어 통합했습니다.

이 파일 하나로 다음이 모두 가능합니다:
  - UART로 배치(다중 이미지) 추론 실행
  - 결과 테이블 + 요약 패널(수율/불량률/클래스 분포/Pareto TOP3)
  - 이미지 더블클릭 시 미리보기 다이얼로그
  - 불량률 임계값(%) 초과 시 경고 배너 + 팝업
  - "배치 결과" CSV, "배치 요약" CSV, PDF 리포트 3종 저장
      batch_result_{YYYYmmdd}_{HHMMSS}.csv   (파일별 상세 결과)
      batch_summary_{YYYYmmdd}_{HHMMSS}.csv  (요약 통계, wafer_cnn_gui_compare.py가 이 파일을 읽음)
      batch_summary_{YYYYmmdd}_{HHMMSS}.pdf  (사람이 보는 리포트)

⚠️ wafer_cnn_gui_compare.py는 CLASS_NAMES, NORMAL_IDX를
   이 파일에서 import해야 합니다 (기존 wafer_cnn_gui_many.py 대신):

   from wafer_cnn_gui_csv import CLASS_NAMES, NORMAL_IDX

실행: python wafer_cnn_gui_csv.py
"""

import os
import sys
import csv
import time
from collections import Counter
from datetime import datetime
from io import BytesIO

import serial
import serial.tools.list_ports
import numpy as np
from PIL import Image

import matplotlib
matplotlib.use("Agg")  # Qt 이벤트 루프와 충돌 없이 오프스크린 렌더링
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib import font_manager as fm

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage,
    PageBreak,
)

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog,
    QComboBox, QTextEdit, QGroupBox, QProgressBar, QDoubleSpinBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
    QMessageBox, QDialog,
)
from PyQt5.QtGui import QColor, QBrush, QImage, QPixmap
from PyQt5.QtCore import Qt, QThread, pyqtSignal


# ==================================================================
# 공통 설정값 (원래 wafer_cnn_gui_many.py 상단)
# ==================================================================
BAUD_RATE   = 115200
DEFAULT_TIMEOUT_S = 30.0
IMG_W       = 48
IMG_H       = 48
IMG_PIXELS  = IMG_W * IMG_H
START_BYTE  = 0xAA
END_BYTE    = 0x55
INPUT_SCALE = 4.0        # = 2^ACT_FRAC_BITS (ACT_FRAC_BITS=2, quant_meta.json 참고)
MAX_UPLOAD_W = 1024
MAX_UPLOAD_H = 1024

CLASS_NAMES = [
    "Center", "Donut", "Edge-Loc", "Edge-Ring",
    "Loc", "Near-full", "Random", "Scratch", "Normal"
]
CLASS_COLORS = [
    "#E74C3C", "#9B59B6", "#3498DB", "#1ABC9C",
    "#F39C12", "#E67E22", "#27AE60", "#2980B9", "#95A5A6"
]
NORMAL_IDX = 8   # 정상 클래스 인덱스
NORMAL_NAME = CLASS_NAMES[NORMAL_IDX]  # "Normal"

IMG_EXTS = (".png", ".bmp", ".jpg", ".jpeg", ".tif", ".tiff")

# 결함 클래스별 공정 단계 / 원인 / 조치 매핑
PROCESS_MAP = {
    "Center": {
        "stage":  "CMP / 증착(CVD·PVD) / 세정 공정",
        "cause":  "웨이퍼 중심부 두께 편차 - 슬러리 분포 불균일, CMP 압력 편차, "
                  "증착 시 중심부 온도 편차, 세정 공정 균일도 저하.",
        "action": "슬러리 유량/압력 재조정, CMP 헤드 압력 균일화 점검, "
                  "증착 온도 프로파일 점검 및 세정 레시피 재검토.",
    },
    "Donut": {
        "stage":  "식각(에칭·건식) / 도핑 공정",
        "cause":  "웨이퍼 중앙과 외곽부에서 대칭적인 편차 - 가스 유량 불균일, "
                  "이온주입 각도 편차로 인한 도넛 형태 결함.",
        "action": "가스 유량·플라즈마 균일도 점검 및 EBR 마진 재조정, "
                  "이온주입 각도 균일도 재점검(설비 캘리브레이션) 진행.",
    },
    "Edge-Loc": {
        "stage":  "가장자리 공정 / 노광 초점 / 식각",
        "cause":  "웨이퍼 외곽부 국지적 결함 발생 - 가장자리 초점/노광 균일도 "
                  "저하, 엣지비드제거(EBR) 편차, 가장자리 파티클 흡착.",
        "action": "가장자리 노광 초점/에너지 균일도 재조정, EBR 레시피 점검, "
                  "가장자리 파티클 저감 세정 강화.",
    },
    "Edge-Ring": {
        "stage":  "식각 / 정전척 균일도 / 온도(챔버) 관리",
        "cause":  "웨이퍼 가장자리 전체 링 형태 결함 - 정전척 흡착 불균일, "
                  "정전척(ESC) 온도 편차, 챔버 링 부품 마모.",
        "action": "정전척 흡착 균일도 점검 및 클리닝, 챔버 소모품(엣지링) "
                  "교체 주기 확인, 온도 프로파일 재보정.",
    },
    "Loc": {
        "stage":  "포토 공정 / 파티클 / 이물 오염",
        "cause":  "웨이퍼 임의 위치에 국지적(localized) 결함 발생 - 노광 파티클 "
                  "오염, 마스크 결함 전사, 웨이퍼 이물 흡착.",
        "action": "노광 클린룸 파티클 모니터링 강화, 마스크/펠리클 점검, "
                  "이물 검사 후 세정 레시피 강화.",
    },
    "Near-full": {
        "stage":  "웨이퍼(전) 공정 전체 / 설비·계측 이상",
        "cause":  "웨이퍼 전면 결함 발생 - 설비 이상 정지(온도·압력·유량 급변), "
                  "레시피 오적용/오류, 계측 오류/센서 이상으로 전면 fail 발생.",
        "action": "해당 로트 즉시 격리 및 설비 로그/레시피 이력 점검, 설비·센서 "
                  "정밀 점검, 재발 방지 위해 인터록 조건 재검토.",
    },
    "Random": {
        "stage":  "다수/복합 공정 (특정 단계 특정 불가)",
        "cause":  "특정 공정 단계로 특정하기 어려운 무작위 결함 - 설비 상태 "
                  "불안정, 환경 오염, 세정/건조 공정 등 다양한 요인 복합.",
        "action": "설비 예방점검 이력·주기 점검, 클린룸/환경 파티클 점검, 발생 "
                  "빈도가 높은 로트 공통 공정 단계 교차 분석(공정 추적 필요).",
    },
    "Scratch": {
        "stage":  "이송 / CMP / 핸들링 공정",
        "cause":  "선(line) 형태의 물리적 결함 - 웨이퍼 이송·핸들링 중 접촉 스크래치, "
                  "CMP 패드 상 이물/스크래치, 카세트 접촉 결함.",
        "action": "웨이퍼 핸들러/이송 로봇 정렬 점검, CMP 패드 컨디셔닝·이물 "
                  "제거 강화, 카세트 슬롯 상태 점검.",
    },
}


def _first_existing(paths):
    for p in paths:
        if os.path.isfile(p):
            return p
    return None


def _register_korean_fonts():
    """
    matplotlib(차트)와 reportlab(PDF 텍스트)에 한글이 깨지지 않게 시스템에
    설치된 한글 폰트(나눔고딕/Noto)를 등록한다.

    반환: (pdf_font_regular, pdf_font_bold) - reportlab에서 사용할 폰트 이름.
    폰트가 하나도 없으면 Helvetica로 폴백한다(한글은 깨질 수 있음),
    프로세스는 계속 진행한다.
    """
    _KOREAN_FONT_CANDIDATES = [
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ]
    _KOREAN_BOLD_CANDIDATES = [
        "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    ]

    reg_path = _first_existing(_KOREAN_FONT_CANDIDATES)
    bold_path = _first_existing(_KOREAN_BOLD_CANDIDATES)

    if reg_path:
        pdfmetrics.registerFont(TTFont("Korean", reg_path))
        fm.fontManager.addfont(reg_path)
        matplotlib.rcParams["font.family"] = fm.FontProperties(fname=reg_path).get_name()
        matplotlib.rcParams["axes.unicode_minus"] = False
    if bold_path:
        pdfmetrics.registerFont(TTFont("Korean-Bold", bold_path))

    pdf_regular = "Korean" if reg_path else "Helvetica"
    pdf_bold = "Korean-Bold" if bold_path else ("Korean" if reg_path else "Helvetica-Bold")
    return pdf_regular, pdf_bold


FONT_REGULAR, FONT_BOLD = _register_korean_fonts()


# ==================================================================
# 이미지 → binary 변환 (baseline image_to_bin과 동일)
# ==================================================================
def image_to_bin(path: str):
    """
    입력 이미지를 흑백 48x48로 리사이즈하고, RTL이 기대하는 고정소수점
    포맷(0~127, x4 스케일)의 바이트로 변환한다.

    반환: (bin_bytes, orig_w, orig_h)
    """
    img = Image.open(path).convert('L')
    orig_w, orig_h = img.size
    if orig_w > MAX_UPLOAD_W or orig_h > MAX_UPLOAD_H:
        raise ValueError(f"이미지가 너무 큽니다 ({orig_w}×{orig_h}). 최대 {MAX_UPLOAD_W}×{MAX_UPLOAD_H}")

    resized = img.resize((IMG_W, IMG_H), Image.BILINEAR)
    arr = np.array(resized, dtype=np.float32)  # 0~255

    byte_vals = np.clip(np.round((arr / 255.0) * INPUT_SCALE), 0, 127).astype(np.uint8)
    return byte_vals.tobytes(), orig_w, orig_h


# ==================================================================
# 배치 UART 워커 스레드
# ==================================================================
class BatchUartWorker(QThread):
    """
    baseline UartWorker의 단일 프레임 프로토콜(0xAA + 2304바이트 → 2바이트 응답)을
    파일 목록에 대해 반복하며, 진행 상황과 결과를 시그널로 보고한다.
    """
    sig_item     = pyqtSignal(int, int, float)  # (index, pred_label, elapsed_ms) - 성공
    sig_item_err = pyqtSignal(int, str)          # (index, error_msg)              - 실패
    sig_progress = pyqtSignal(int, int)          # (done_count, total)
    sig_log      = pyqtSignal(str)

    def __init__(self, ser, file_paths, timeout_sec):
        super().__init__()
        self.ser         = ser
        self.file_paths  = file_paths   # 처리할 이미지 파일 경로 목록
        self.timeout_sec = timeout_sec
        self._abort      = False

    def abort(self):
        self._abort = True

    def _infer_one(self, bin_bytes):
        """baseline UartWorker.run()과 동일한 단일 송수신. 반환 (pred, elapsed_ms)."""
        ser = self.ser
        ser.timeout = self.timeout_sec
        ser.reset_input_buffer()
        ser.reset_output_buffer()
        t0 = time.time()

        ser.write(bytes([START_BYTE]))
        CHUNK = 512
        total = len(bin_bytes)
        sent  = 0
        while sent < total:
            end = min(sent + CHUNK, total)
            ser.write(bin_bytes[sent:end])
            sent = end
        ser.flush()

        resp = ser.read(2)
        elapsed_ms = (time.time() - t0) * 1000.0

        if len(resp) < 2:
            raise TimeoutError(
                f"응답 타임아웃: {self.timeout_sec:.1f}초 내 {len(resp)}/2 바이트만 수신"
            )
        pred, end_byte = resp[0], resp[1]
        if end_byte != END_BYTE:
            raise ValueError(f"종료바이트 불일치: 예상 0x{END_BYTE:02X}, 수신 0x{end_byte:02X}")
        if pred < 0 or pred >= len(CLASS_NAMES):
            raise ValueError(f"잘못된 pred: {pred}")
        return pred, elapsed_ms

    def run(self):
        total = len(self.file_paths)
        for i, path in enumerate(self.file_paths):
            if self._abort:
                self.sig_log.emit("[중단] 사용자 요청으로 중단됨")
                break
            fname = os.path.basename(path)
            try:
                bin_bytes, _, _ = image_to_bin(path)
                self.sig_log.emit(f"[{i + 1}/{total}] 전송: {fname}")
                pred, elapsed_ms = self._infer_one(bin_bytes)
                self.sig_item.emit(i, pred, elapsed_ms)
                self.sig_log.emit(f"[{i + 1}/{total}] 결과: {CLASS_NAMES[pred]} ({elapsed_ms:.1f} ms)")
            except serial.SerialException as e:
                self.sig_item_err.emit(i, f"시리얼 오류: {e}")
                self.sig_log.emit(f"[{i + 1}/{total}] 치명적 오류 - 시리얼 오류: {e}")
                break  # 연결이 끊긴 경우 배치 중단
            except Exception as e:
                self.sig_item_err.emit(i, str(e))
                self.sig_log.emit(f"[{i + 1}/{total}] 오류: {e}")
            self.sig_progress.emit(i + 1, total)


# ==================================================================
# 배치 기본 윈도우 (baseline)
# ==================================================================
class BatchWindow(QMainWindow):
    COL_FILE, COL_CLASS, COL_IDX, COL_STAGE, COL_ACTION, COL_MS = range(6)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Wafer Defect Classifier (Batch) — Zybo Z7-20 NPU")
        self.setMinimumSize(1080, 680)
        self.file_paths = []       # 배치 대상 이미지 파일 목록
        self.results    = []       # dict per file: 저장/CSV용
        self.worker     = None
        self.ser        = None
        self._build_ui()
        self._refresh_ports()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(10)
        root.setContentsMargins(12, 12, 12, 12)

        port_grp = QGroupBox("UART 연결  (115200 baud)")
        pl = QHBoxLayout(port_grp)

        self.port_combo = QComboBox()
        self.port_combo.setMinimumWidth(140)
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
        root.addWidget(port_grp)

        sel_grp = QGroupBox("이미지 선택 & 실행")
        sl = QHBoxLayout(sel_grp)

        btn_files = QPushButton("여러 이미지 선택")
        btn_files.setFixedHeight(34)
        btn_files.clicked.connect(self._select_files)
        sl.addWidget(btn_files)

        btn_folder = QPushButton("폴더 통째 선택")
        btn_folder.setFixedHeight(34)
        btn_folder.clicked.connect(self._select_folder)
        sl.addWidget(btn_folder)

        btn_clear = QPushButton("🗑 목록 지우기")
        btn_clear.setObjectName("secondary")
        btn_clear.setFixedHeight(34)
        btn_clear.clicked.connect(self._clear_list)
        sl.addWidget(btn_clear)

        sl.addStretch()

        self.sel_info = QLabel("선택된 파일: 0개")
        self.sel_info.setStyleSheet("color:#aaa; font-size:12px;")
        sl.addWidget(self.sel_info)

        self.btn_run = QPushButton("▶ 배치 추론 실행")
        self.btn_run.setFixedHeight(40)
        self.btn_run.setEnabled(False)
        self.btn_run.setStyleSheet(
            "QPushButton{background:#1abc9c;color:white;font-size:14px;"
            "border-radius:6px;font-weight:bold;padding:0 16px;}"
            "QPushButton:disabled{background:#3a3a3a;color:#666;}"
            "QPushButton:hover{background:#16a085;}"
        )
        self.btn_run.clicked.connect(self._run_batch)
        sl.addWidget(self.btn_run)

        self.btn_stop = QPushButton("■ 중단")
        self.btn_stop.setFixedHeight(40)
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop_batch)
        sl.addWidget(self.btn_stop)

        self.btn_save = QPushButton("CSV 저장")
        self.btn_save.setFixedHeight(40)
        self.btn_save.setEnabled(False)
        self.btn_save.clicked.connect(self._save_csv)
        sl.addWidget(self.btn_save)
        root.addWidget(sel_grp)

        prog_row = QHBoxLayout()
        self.progress_lbl = QLabel("대기 중")
        self.progress_lbl.setStyleSheet("color:#aaa; font-size:12px;")
        self.progress_lbl.setFixedWidth(160)
        prog_row.addWidget(self.progress_lbl)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setFixedHeight(12)
        self.progress.setTextVisible(False)
        self.progress.setStyleSheet(
            "QProgressBar{border:none;background:#333;border-radius:6px;}"
            "QProgressBar::chunk{background:#1abc9c;border-radius:6px;}"
        )
        prog_row.addWidget(self.progress)
        root.addLayout(prog_row)

        res_grp = QGroupBox("추론 결과")
        rl = QVBoxLayout(res_grp)
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels([
            "파일명", "예측 클래스", "클래스 인덱스", "관련 공정 단계", "권장 조치 방안", "지연(ms)"
        ])
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.verticalHeader().setVisible(False)
        hdr = self.table.horizontalHeader()
        hdr.setSectionResizeMode(self.COL_FILE, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(self.COL_CLASS, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(self.COL_IDX, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(self.COL_STAGE, QHeaderView.Stretch)
        hdr.setSectionResizeMode(self.COL_ACTION, QHeaderView.Stretch)
        hdr.setSectionResizeMode(self.COL_MS, QHeaderView.ResizeToContents)
        self.table.setStyleSheet(
            "QTableWidget{background:#1a1a1a;color:#ddd;gridline-color:#3a3a3a;"
            "border:1px solid #444;}"
            "QHeaderView::section{background:#333;color:#eee;border:none;"
            "padding:5px;font-weight:bold;}"
        )
        rl.addWidget(self.table)
        root.addWidget(res_grp, 3)

        log_grp = QGroupBox("로그")
        ll = QVBoxLayout(log_grp)
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMaximumHeight(120)
        self.log_box.setStyleSheet(
            "background:#0d0d0d;color:#0f0;font-family:monospace;font-size:11px;"
        )
        ll.addWidget(self.log_box)
        root.addWidget(log_grp, 1)

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
            QPushButton:disabled{color:#666;}
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
            self._log("[오류] 연결할 포트가 없습니다")
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
            self._log("[연결] 포트 연결이 해제되었습니다")
        self._update_run_button_state()

    def _update_run_button_state(self):
        connected = self.ser is not None and self.ser.is_open
        busy = self.worker is not None and self.worker.isRunning()
        self.btn_run.setEnabled(connected and len(self.file_paths) > 0 and not busy)

    def _select_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "이미지 여러 개 선택", "",
            "Images (*.png *.bmp *.jpg *.jpeg *.tif *.tiff)"
        )
        if paths:
            self._set_file_list(paths)

    def _select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "이미지 폴더 선택", "")
        if not folder:
            return
        paths = [
            os.path.join(folder, f)
            for f in sorted(os.listdir(folder))
            if f.lower().endswith(IMG_EXTS)
        ]
        if not paths:
            self._log(f"[경고] 이미지 파일이 없습니다: {folder}")
            return
        self._set_file_list(paths)

    def _set_file_list(self, paths):
        self.file_paths = paths
        self.results = [None] * len(paths)
        self.sel_info.setText(f"선택된 파일: {len(paths)}개")
        self.btn_save.setEnabled(False)
        self._populate_table_pending()
        self.progress.setValue(0)
        self.progress_lbl.setText("대기 중")
        self._log(f"[선택] 파일 {len(paths)}개 등록")
        self._update_run_button_state()

    def _clear_list(self):
        if self.worker is not None and self.worker.isRunning():
            return
        self.file_paths = []
        self.results = []
        self.table.setRowCount(0)
        self.sel_info.setText("선택된 파일: 0개")
        self.btn_save.setEnabled(False)
        self.progress.setValue(0)
        self.progress_lbl.setText("대기 중")
        self._update_run_button_state()

    def _populate_table_pending(self):
        self.table.setRowCount(len(self.file_paths))
        for row, path in enumerate(self.file_paths):
            self.table.setItem(row, self.COL_FILE, QTableWidgetItem(os.path.basename(path)))
            self.table.setItem(row, self.COL_CLASS, QTableWidgetItem("대기"))
            for c in (self.COL_IDX, self.COL_STAGE, self.COL_ACTION, self.COL_MS):
                self.table.setItem(row, c, QTableWidgetItem(""))

    def _run_batch(self):
        if self.ser is None or not self.ser.is_open:
            self._log("[오류] 보드가 연결되지 않았습니다")
            return
        if not self.file_paths:
            self._log("[오류] 선택된 이미지가 없습니다")
            return

        self.results = [None] * len(self.file_paths)
        self._populate_table_pending()
        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_save.setEnabled(False)
        self.progress.setValue(0)
        self.progress_lbl.setText(f"0 / {len(self.file_paths)}")
        self._log(f"[시작] {len(self.file_paths)}개 배치 시작")

        self.worker = BatchUartWorker(self.ser, self.file_paths, self.timeout_spin.value())
        self.worker.sig_item.connect(self._on_item)
        self.worker.sig_item_err.connect(self._on_item_err)
        self.worker.sig_progress.connect(self._on_progress)
        self.worker.sig_log.connect(self._log)
        self.worker.finished.connect(self._on_batch_done)
        self.worker.start()

    def _stop_batch(self):
        if self.worker is not None and self.worker.isRunning():
            self.worker.abort()
            self.btn_stop.setEnabled(False)
            self._log("[중단] 현재 항목 완료 후 중단됩니다")

    def _on_item(self, index: int, pred: int, elapsed_ms: float):
        pred = max(0, min(pred, len(CLASS_NAMES) - 1))
        name = CLASS_NAMES[pred]
        color = CLASS_COLORS[pred]
        is_defect = (pred != NORMAL_IDX)
        stage  = PROCESS_MAP[name]["stage"]  if is_defect else ""
        action = PROCESS_MAP[name]["action"] if is_defect else ""

        now = datetime.now()
        self.results[index] = {
            "file":   os.path.basename(self.file_paths[index]),
            "class":  name,
            "idx":    pred,
            "stage":  stage,
            "action": action,
            "date":   now.strftime("%Y-%m-%d"),
            "time":   now.strftime("%H:%M:%S"),
            "ms":     round(elapsed_ms, 1),
        }

        cls_item = QTableWidgetItem(name)
        cls_item.setForeground(QBrush(QColor(color)))
        self.table.setItem(index, self.COL_CLASS, cls_item)
        self.table.setItem(index, self.COL_IDX, QTableWidgetItem(str(pred)))
        self.table.setItem(index, self.COL_STAGE, QTableWidgetItem(stage))
        self.table.setItem(index, self.COL_ACTION, QTableWidgetItem(action))
        self.table.setItem(index, self.COL_MS, QTableWidgetItem(f"{elapsed_ms:.1f}"))
        self.table.scrollToItem(cls_item)

    def _on_item_err(self, index: int, msg: str):
        now = datetime.now()
        self.results[index] = {
            "file":   os.path.basename(self.file_paths[index]),
            "class":  "ERROR",
            "idx":    -1,
            "stage":  "",
            "action": msg,
            "date":   now.strftime("%Y-%m-%d"),
            "time":   now.strftime("%H:%M:%S"),
            "ms":     0.0,
        }
        err_item = QTableWidgetItem("ERROR")
        err_item.setForeground(QBrush(QColor("#e74c3c")))
        self.table.setItem(index, self.COL_CLASS, err_item)
        self.table.setItem(index, self.COL_IDX, QTableWidgetItem("-1"))
        self.table.setItem(index, self.COL_ACTION, QTableWidgetItem(msg))

    def _on_progress(self, done: int, total: int):
        self.progress.setValue(int(done / total * 100) if total else 0)
        self.progress_lbl.setText(f"{done} / {total}")

    def _on_batch_done(self):
        self.btn_stop.setEnabled(False)
        self._update_run_button_state()
        done = sum(1 for r in self.results if r is not None)
        self.btn_save.setEnabled(done > 0)
        self.progress_lbl.setText(f"완료: {done} / {len(self.file_paths)}")
        self._log(f"[완료] 배치 종료 - {done}개 처리됨")

    def _save_csv(self):
        rows = [r for r in self.results if r is not None]
        if not rows:
            self._log("[오류] 저장할 결과가 없습니다")
            return

        now = datetime.now()
        default_name = f"batch_result_{now.strftime('%Y%m%d')}_{now.strftime('%H%M%S')}.csv"
        path, _ = QFileDialog.getSaveFileName(
            self, "CSV 결과 파일 저장", default_name, "CSV Files (*.csv)"
        )
        if not path:
            return
        if not path.lower().endswith(".csv"):
            path += ".csv"

        header = [
            "파일명", "예측 클래스", "예측 클래스 인덱스",
            "관련 공정 단계", "권장 조치 방안",
            "실행 날짜", "실행 시각", "추론 지연시간(ms)",
        ]
        try:
            with open(path, "w", newline="", encoding="utf-8-sig") as f:
                w = csv.writer(f)
                w.writerow(header)
                for r in rows:
                    w.writerow([
                        r["file"], r["class"], r["idx"],
                        r["stage"], r["action"],
                        r["date"], r["time"], r["ms"],
                    ])
            self._log(f"[저장] CSV 저장 완료: {path} ({len(rows)}건)")
            QMessageBox.information(self, "저장 완료",
                                    f"CSV 저장 완료:\n{path}\n\n{len(rows)}건")
        except Exception as e:
            self._log(f"[오류] CSV 저장 실패: {e}")
            QMessageBox.critical(self, "저장 실패", str(e))

    def _log(self, msg: str):
        self.log_box.append(msg)
        sb = self.log_box.verticalScrollBar()
        sb.setValue(sb.maximum())

    def closeEvent(self, event):
        if self.worker is not None and self.worker.isRunning():
            self.worker.abort()
            self.worker.wait(3000)
        if self.ser is not None and self.ser.is_open:
            self.ser.close()
        event.accept()


# ==================================================================
# 배치 통계 계산
# ==================================================================
def compute_batch_stats(rows):
    """
    rows: BatchWindow.results에서 None을 제외한 dict 목록
          (각 dict는 BatchWindow._on_item / _on_item_err 결과와 동일:
           file, class, idx, stage, action, date, time, ms)

    반환 dict:
      total_attempted   전체 시도 파일 수(오류 포함)
      error_count       오류(응답 실패 등)로 처리된 항목 수
      total_classified  정상적으로 분류된 파일 수(오류 제외, 정상/불량 모두 포함)
      class_counts      {클래스명: 개수} - CLASS_NAMES 9개 전부 포함(0인 경우도)
      class_pct         {클래스명: 비율(%)}
      yield_pct         수율(Normal 비율, %)
      defect_rate_pct   불량률(100 - 수율, %)
      pareto            [(클래스명, 개수, 비율%), ...] 불량 클래스만, 내림차순
      avg_ms            평균 추론 지연시간(ms), 정상 분류 건 기준
    """
    error_rows = [r for r in rows if r["idx"] == -1]
    valid_rows = [r for r in rows if r["idx"] != -1]

    total_attempted = len(rows)
    total_classified = len(valid_rows)

    counts = Counter(r["class"] for r in valid_rows)
    class_counts = {name: counts.get(name, 0) for name in CLASS_NAMES}
    class_pct = {
        name: (cnt / total_classified * 100.0 if total_classified else 0.0)
        for name, cnt in class_counts.items()
    }

    normal_count = class_counts.get(NORMAL_NAME, 0)
    yield_pct = (normal_count / total_classified * 100.0) if total_classified else 0.0
    defect_rate_pct = 100.0 - yield_pct if total_classified else 0.0

    pareto = sorted(
        (
            (name, cnt, class_pct[name])
            for name, cnt in class_counts.items()
            if name != NORMAL_NAME and cnt > 0
        ),
        key=lambda t: t[1],
        reverse=True,
    )

    avg_ms = (sum(r["ms"] for r in valid_rows) / total_classified) if total_classified else 0.0

    return {
        "total_attempted": total_attempted,
        "error_count": len(error_rows),
        "total_classified": total_classified,
        "class_counts": class_counts,
        "class_pct": class_pct,
        "yield_pct": yield_pct,
        "defect_rate_pct": defect_rate_pct,
        "pareto": pareto,
        "avg_ms": avg_ms,
    }


def render_class_distribution_png(stats, width_in=7.2, height_in=4.4, dpi=150):
    """클래스 분포(9개 클래스, Normal 포함) 막대그래프를 PNG 바이트로 렌더링."""
    fig = Figure(figsize=(width_in, height_in), dpi=dpi)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)

    counts = [stats["class_counts"][name] for name in CLASS_NAMES]
    bars = ax.bar(CLASS_NAMES, counts, color=CLASS_COLORS)
    ax.set_title("클래스 분포 (배치)", pad=14)
    ax.set_ylabel("개수")
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(30)
        lbl.set_ha("right")
        lbl.set_rotation_mode("anchor")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bar, cnt in zip(bars, counts):
        if cnt > 0:
            ax.annotate(
                str(cnt),
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3), textcoords="offset points",
                ha="center", va="bottom", fontsize=8,
            )

    fig.subplots_adjust(bottom=0.28, top=0.90)
    buf = BytesIO()
    canvas.print_png(buf)
    buf.seek(0)
    return buf.read()


def generate_summary_pdf(pdf_path, stats, meta, chart_png_bytes):
    """
    meta: {"date": "YYYY-MM-DD", "time": "HH:MM:SS", "source_note": str}
    """
    doc = SimpleDocTemplate(
        pdf_path, pagesize=A4,
        topMargin=18 * mm, bottomMargin=16 * mm,
        leftMargin=18 * mm, rightMargin=18 * mm,
    )

    title_style = ParagraphStyle(
        "Title", fontName=FONT_BOLD, fontSize=20, alignment=TA_CENTER,
        spaceAfter=4 * mm, textColor=colors.HexColor("#1a1a1a"),
    )
    meta_style = ParagraphStyle(
        "Meta", fontName=FONT_REGULAR, fontSize=10, alignment=TA_CENTER,
        spaceAfter=6 * mm, textColor=colors.HexColor("#555555"),
    )
    heading_style = ParagraphStyle(
        "Heading", fontName=FONT_BOLD, fontSize=13,
        spaceBefore=6 * mm, spaceAfter=3 * mm, textColor=colors.HexColor("#1abc9c"),
    )
    body_style = ParagraphStyle(
        "Body", fontName=FONT_REGULAR, fontSize=10, leading=14,
    )

    story = []

    story.append(Paragraph("Wafer NPU 배치 검사 리포트", title_style))
    story.append(Paragraph(
        f"생성 일시: {meta['date']} {meta['time']}  |  "
        f"전체 시도 건수: {stats['total_attempted']}건"
        + (f" (오류 {stats['error_count']}건 제외 {stats['total_classified']}건 정상 분류)"
           if stats["error_count"] else ""),
        meta_style,
    ))

    story.append(Paragraph("수율(Yield) / 불량률 요약", heading_style))
    yield_defect_tbl = Table(
        [
            ["수율 (Yield)", f"{stats['yield_pct']:.1f} %"],
            ["불량률 (Defect Rate)", f"{stats['defect_rate_pct']:.1f} %"],
        ],
        colWidths=[70 * mm, 40 * mm],
    )
    yield_defect_tbl.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), FONT_BOLD),
        ("FONTSIZE", (0, 0), (-1, -1), 14),
        ("BACKGROUND", (0, 0), (1, 0), colors.HexColor("#d5f5e3")),
        ("BACKGROUND", (0, 1), (1, 1), colors.HexColor("#fadbd8")),
        ("TEXTCOLOR", (0, 0), (1, 0), colors.HexColor("#196f3d")),
        ("TEXTCOLOR", (0, 1), (1, 1), colors.HexColor("#943126")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#999999")),
        ("ALIGN", (1, 0), (1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(yield_defect_tbl)
    story.append(Spacer(1, 8 * mm))

    story.append(Paragraph("클래스 분포 요약", heading_style))
    dist_header = ["클래스명", "개수", "비율(%)"]
    dist_rows = [
        [name, str(stats["class_counts"][name]), f"{stats['class_pct'][name]:.1f}"]
        for name in CLASS_NAMES
    ]
    dist_tbl = Table([dist_header] + dist_rows, colWidths=[60 * mm, 30 * mm, 30 * mm])
    dist_style = [
        ("FONTNAME", (0, 0), (-1, 0), FONT_BOLD),
        ("FONTNAME", (0, 1), (-1, -1), FONT_REGULAR),
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#333333")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#bbbbbb")),
        ("ALIGN", (1, 0), (2, -1), "CENTER"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f5f5")]),
    ]
    normal_row = 1 + CLASS_NAMES.index(NORMAL_NAME)
    dist_style.append(("BACKGROUND", (0, normal_row), (-1, normal_row), colors.HexColor("#d5f5e3")))
    dist_tbl.setStyle(TableStyle(dist_style))
    story.append(dist_tbl)

    story.append(Spacer(1, 8 * mm))
    story.append(RLImage(BytesIO(chart_png_bytes), width=170 * mm, height=104 * mm))

    story.append(PageBreak())

    story.append(Paragraph("불량 유형 Pareto 분석", heading_style))
    pareto = stats["pareto"]
    if not pareto:
        story.append(Paragraph("불량으로 분류된 항목이 없습니다 (전량 정상 판정).", body_style))
    else:
        top3 = pareto[:3]
        rank_colors = ["#f9e79f", "#eaeded", "#f5cba7"]  # gold / silver / bronze
        top3_lines = "<br/>".join(
            f"<b>{i + 1}위. {name}</b> — {cnt}건 ({pct:.1f}%)"
            for i, (name, cnt, pct) in enumerate(top3)
        )
        story.append(Paragraph(
            f"<b>가장 빈도가 높은 불량 유형 TOP {len(top3)}</b><br/>{top3_lines}",
            body_style,
        ))
        story.append(Spacer(1, 7 * mm))

        pareto_header = ["순위", "클래스명", "개수", "비율(%)"]
        pareto_rows = [
            [str(i + 1), name, str(cnt), f"{pct:.1f}"]
            for i, (name, cnt, pct) in enumerate(pareto)
        ]
        pareto_tbl = Table([pareto_header] + pareto_rows,
                            colWidths=[18 * mm, 60 * mm, 25 * mm, 25 * mm])
        pareto_style = [
            ("FONTNAME", (0, 0), (-1, 0), FONT_BOLD),
            ("FONTNAME", (0, 1), (-1, -1), FONT_REGULAR),
            ("FONTSIZE", (0, 0), (-1, -1), 9.5),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#333333")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#bbbbbb")),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ]
        for i in range(min(3, len(pareto_rows))):
            pareto_style.append(
                ("BACKGROUND", (0, i + 1), (-1, i + 1), colors.HexColor(rank_colors[i]))
            )
        pareto_tbl.setStyle(TableStyle(pareto_style))
        story.append(pareto_tbl)

    story.append(Spacer(1, 10 * mm))
    story.append(Paragraph(
        f"평균 추론 지연시간: <b>{stats['avg_ms']:.1f} ms</b> "
        f"(정상 분류 {stats['total_classified']}건 기준, 송신+추론+수신 포함)",
        body_style,
    ))
    if meta.get("source_note"):
        story.append(Spacer(1, 4 * mm))
        story.append(Paragraph(meta["source_note"], meta_style))

    doc.build(story)


# ==================================================================
# 최종 윈도우: 배치 baseline + 요약 패널 + CSV/PDF 저장
# ==================================================================
class FinalWindow(BatchWindow):
    def __init__(self):
        self._last_rows = []
        super().__init__()
        self.setWindowTitle("Wafer Defect Classifier (Final) — Zybo Z7-20 NPU")

    def _build_ui(self):
        super()._build_ui()

        screen = QApplication.desktop().availableGeometry(self)
        target_w = min(1400, screen.width() - 40)
        target_h = min(900, screen.height() - 60)
        self.setMinimumSize(min(1100, target_w), min(760, target_h))
        self.resize(target_w, target_h)

        self.btn_save.setText("CSV + PDF 저장")

        self.table.setWordWrap(True)
        self.table.verticalHeader().setDefaultSectionSize(28)

        root = self.centralWidget().layout()
        log_grp = root.itemAt(root.count() - 1).widget()
        log_grp.setVisible(False)
        root.setStretch(root.indexOf(log_grp), 0)

        res_grp = self.table.parentWidget()
        root.setStretch(root.indexOf(res_grp), 1)

        summary_grp = QGroupBox("배치 요약 (수율 · 불량률 · 클래스 분포)")
        sl = QVBoxLayout(summary_grp)

        self.summary_headline = QLabel("배치를 실행하면 요약이 여기에 표시됩니다.")
        self.summary_headline.setStyleSheet("color:#ccc; font-size:12px;")
        self.summary_headline.setWordWrap(True)
        sl.addWidget(self.summary_headline)

        row = QHBoxLayout()

        self.summary_table = QTableWidget(len(CLASS_NAMES), 3)
        self.summary_table.setHorizontalHeaderLabels(["클래스명", "개수", "비율(%)"])
        self.summary_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.summary_table.setSelectionMode(QAbstractItemView.NoSelection)
        self.summary_table.verticalHeader().setVisible(False)
        ROW_H = 26
        self.summary_table.verticalHeader().setDefaultSectionSize(ROW_H)
        self.summary_table.verticalHeader().setSectionResizeMode(QHeaderView.Fixed)
        self.summary_table.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        hdr = self.summary_table.horizontalHeader()
        hdr.setSectionResizeMode(QHeaderView.Stretch)
        self.summary_table.setStyleSheet(
            "QTableWidget{background:#1a1a1a;color:#ddd;gridline-color:#3a3a3a;"
            "border:1px solid #444;}"
            "QHeaderView::section{background:#333;color:#eee;border:none;"
            "padding:4px;font-weight:bold;}"
        )
        for i, name in enumerate(CLASS_NAMES):
            self.summary_table.setItem(i, 0, QTableWidgetItem(name))
            self.summary_table.setItem(i, 1, QTableWidgetItem("-"))
            self.summary_table.setItem(i, 2, QTableWidgetItem("-"))
            self.summary_table.setRowHeight(i, ROW_H)

        needed_h = 40 + self.summary_table.rowCount() * ROW_H + 2 * self.summary_table.frameWidth() + 24
        self.summary_table.setFixedHeight(needed_h)
        row.addWidget(self.summary_table, 2)

        self.pareto_label = QLabel("Pareto 요약: -")
        self.pareto_label.setAlignment(Qt.AlignTop)
        self.pareto_label.setWordWrap(True)
        self.pareto_label.setStyleSheet(
            "color:#ddd; font-size:12px; background:#1a1a1a; border:1px solid #444;"
            "border-radius:6px; padding:8px;"
        )
        row.addWidget(self.pareto_label, 1)

        sl.addLayout(row)

        root.insertWidget(root.indexOf(log_grp), summary_grp, 0)

    def _on_item(self, index, pred, elapsed_ms):
        super()._on_item(index, pred, elapsed_ms)
        self.table.resizeRowToContents(index)

    def _on_item_err(self, index, msg):
        super()._on_item_err(index, msg)
        self.table.resizeRowToContents(index)

    def _on_batch_done(self):
        super()._on_batch_done()
        self._last_rows = [r for r in self.results if r is not None]
        if self._last_rows:
            stats = compute_batch_stats(self._last_rows)
            self._update_summary_panel(stats)

    def _update_summary_panel(self, stats):
        self.summary_headline.setText(
            f"총 시도 {stats['total_attempted']}건"
            + (f" (오류 {stats['error_count']}건)" if stats["error_count"] else "")
            + f"  |  수율(Yield) <b>{stats['yield_pct']:.1f}%</b>"
            f"  |  불량률 <b>{stats['defect_rate_pct']:.1f}%</b>"
            f"  |  평균 지연시간 {stats['avg_ms']:.1f} ms"
        )

        for i, name in enumerate(CLASS_NAMES):
            cnt = stats["class_counts"][name]
            pct = stats["class_pct"][name]
            self.summary_table.item(i, 1).setText(str(cnt))
            self.summary_table.item(i, 2).setText(f"{pct:.1f}")

        pareto = stats["pareto"]
        if not pareto:
            self.pareto_label.setText("<b>불량 유형 Pareto 분석</b><br/>불량 없음 (전량 정상)")
        else:
            lines = "<br/>".join(
                f"<b>{i + 1}위 {name}</b> — {cnt}건 ({pct:.1f}%)"
                for i, (name, cnt, pct) in enumerate(pareto[:3])
            )
            self.pareto_label.setText(f"<b>가장 빈도가 높은 불량 유형 TOP {min(3, len(pareto))}</b><br/>{lines}")

    def _save_csv(self):
        rows = [r for r in self.results if r is not None]
        if not rows:
            self._log("[오류] 저장할 결과가 없습니다")
            return

        default_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
        os.makedirs(default_dir, exist_ok=True)
        folder = QFileDialog.getExistingDirectory(self, "CSV + PDF 저장 폴더 선택", default_dir)
        if not folder:
            return

        now = datetime.now()
        token = f"{now.strftime('%Y%m%d')}_{now.strftime('%H%M%S')}"
        csv_path = os.path.join(folder, f"result_{token}.csv")
        pdf_path = os.path.join(folder, f"summary_{token}.pdf")

        # CSV는 "추론 결과" 테이블(파일명/예측 클래스/클래스 인덱스/
        # 관련 공정 단계/권장 조치 방안/지연시간)을 그대로 저장한다.
        # 수율/불량률 같은 배치 통계는 PDF 리포트에만 담는다(중복 제거).
        try:
            self._write_result_csv(csv_path, rows)
        except Exception as e:
            self._log(f"[오류] CSV 저장 실패: {e}")
            QMessageBox.critical(self, "저장 실패", str(e))
            return

        stats = compute_batch_stats(rows)
        meta = {
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H:%M:%S"),
            "source_note": f"원본 파일: {os.path.basename(csv_path)}",
        }

        try:
            self._update_summary_panel(stats)
            chart_png = render_class_distribution_png(stats)
            generate_summary_pdf(pdf_path, stats, meta, chart_png)
        except Exception as e:
            self._log(f"[오류] PDF 생성 실패: {e}")
            QMessageBox.critical(self, "저장 실패", f"CSV는 저장됐지만 PDF 생성에 실패했습니다:\n{e}")
            return

        self._log(f"[저장] CSV 저장 완료: {csv_path} ({len(rows)}건)")
        self._log(f"[저장] PDF 리포트 저장 완료: {pdf_path}")
        QMessageBox.information(
            self, "저장 완료",
            f"CSV: {csv_path}\nPDF: {pdf_path}\n\n{len(rows)}건 저장됨",
        )

    @staticmethod
    def _write_result_csv(path, rows):
        header = [
            "파일명", "예측 클래스", "예측 클래스 인덱스",
            "관련 공정 단계", "권장 조치 방안",
            "실행 날짜", "실행 시각", "추론 지연시간(ms)",
        ]
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(header)
            for r in rows:
                w.writerow([
                    r["file"], r["class"], r["idx"],
                    r["stage"], r["action"],
                    r["date"], r["time"], r["ms"],
                ])


# ==================================================================
# 이미지 미리보기 다이얼로그
# ==================================================================
class ImagePreviewDialog(QDialog):
    """테이블 행을 더블클릭하면 원본 이미지와 판정 상세 정보를 보여준다."""

    PREVIEW_SIZE = 440

    def __init__(self, parent, result):
        super().__init__(parent)
        self.setWindowTitle(f"이미지 미리보기 — {result['file']}")
        self.setModal(True)
        self.setMinimumSize(520, 620)
        self.resize(540, 640)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(14)

        img_label = QLabel()
        img_label.setAlignment(Qt.AlignCenter)
        img_label.setFixedSize(self.PREVIEW_SIZE, self.PREVIEW_SIZE)
        img_label.setStyleSheet("border:1px solid #444; background:#1a1a1a; color:#888;")
        pixmap = self._load_pixmap(result.get("path"))
        if pixmap is not None:
            img_label.setPixmap(pixmap)
        else:
            img_label.setText("이미지를 불러올 수 없습니다\n(원본 파일이 이동/삭제됨)")
            img_label.setWordWrap(True)
        layout.addWidget(img_label, alignment=Qt.AlignCenter)

        info_label = QLabel(self._build_info_html(result))
        info_label.setWordWrap(True)
        info_label.setStyleSheet(
            "color:#ddd; font-size:16px; line-height:150%;"
            "background:#1a1a1a; border:1px solid #444; border-radius:6px;"
            "padding:14px;"
        )
        layout.addWidget(info_label)
        layout.addStretch()

        btn_close = QPushButton("닫기")
        btn_close.setFixedWidth(90)
        btn_close.clicked.connect(self.accept)
        layout.addWidget(btn_close, alignment=Qt.AlignRight)

        self.setStyleSheet(
            "QDialog{background:#252525;color:#ddd;}"
            "QPushButton{background:#3a3a3a;color:#ddd;border:1px solid #555;"
            "border-radius:4px;padding:5px 10px;}"
            "QPushButton:hover{background:#4a4a4a;}"
        )

    def _load_pixmap(self, path):
        """
        원본 흑백 이미지를 픽셀값 0~255 범위로 min-max 정규화한
        뒤, Qt.SmoothTransformation으로 확대해서 표시한다. 원본이
        48×48처럼 작아도 PIL의 thumbnail()로 축소할 때 뭉개지지 않게,
        여기선 QPixmap.scaled()로 확대한다.
        """
        if not path or not os.path.isfile(path):
            return None
        try:
            arr = np.array(Image.open(path).convert("L"), dtype=np.float32)
            lo, hi = float(arr.min()), float(arr.max())
            if hi > lo:
                arr = (arr - lo) / (hi - lo) * 255.0
            else:
                arr = np.zeros_like(arr)
            arr = arr.astype(np.uint8)

            norm_img = Image.fromarray(arr, mode="L").convert("RGB")
            qi = QImage(norm_img.tobytes(), norm_img.width, norm_img.height,
                        norm_img.width * 3, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qi)
            return pixmap.scaled(
                self.PREVIEW_SIZE, self.PREVIEW_SIZE,
                Qt.KeepAspectRatio, Qt.SmoothTransformation,
            )
        except Exception:
            return None

    @staticmethod
    def _build_info_html(result):
        pred = result["idx"]
        lines = [
            f"<b>파일명:</b> {result['file']}",
        ]
        if pred == -1:
            lines.append(f"<b>오류:</b> {result.get('action', '')}")
        else:
            lines.append(f"<b>예측 클래스:</b> {result['class']} (인덱스: {pred})")
            if pred != NORMAL_IDX:
                lines.append(f"<b>관련 공정 단계:</b> {result.get('stage', '')}")
                lines.append(f"<b>권장 조치 방안:</b> {result.get('action', '')}")
        lines.append(f"<b>추론 지연시간:</b> {result['ms']:.1f} ms")
        return "<br/>".join(lines)


# ==================================================================
# FinalWindow 확장 + 미리보기 다이얼로그 / 불량률 경계값 경고 배너
# ==================================================================
class FinalAddWindow(FinalWindow):
    DEFAULT_DEFECT_THRESHOLD_PCT = 20.0

    def _build_ui(self):
        super()._build_ui()
        self.table.cellDoubleClicked.connect(self._show_image_preview)

        sel_grp = self.btn_run.parentWidget()
        sel_layout = sel_grp.layout()
        sel_layout.addSpacing(16)
        sel_layout.addWidget(QLabel("불량률 경계값:"))
        self.defect_threshold_spin = QDoubleSpinBox()
        self.defect_threshold_spin.setRange(0.0, 100.0)
        self.defect_threshold_spin.setSingleStep(1.0)
        self.defect_threshold_spin.setValue(self.DEFAULT_DEFECT_THRESHOLD_PCT)
        self.defect_threshold_spin.setSuffix(" %")
        self.defect_threshold_spin.setFixedWidth(90)
        self.defect_threshold_spin.valueChanged.connect(self._on_threshold_changed)
        sel_layout.addWidget(self.defect_threshold_spin)

        summary_grp = self.summary_headline.parentWidget()
        summary_grp.layout().insertWidget(0, self._make_warning_banner())

    def _make_warning_banner(self):
        self.warning_banner = QLabel("배치를 실행하면 경계값 초과 여부가 여기에 표시됩니다.")
        self.warning_banner.setAlignment(Qt.AlignCenter)
        self.warning_banner.setWordWrap(True)
        self.warning_banner.setStyleSheet(
            "background:#3a3a3a; color:#ccc; font-size:12px; font-weight:bold;"
            "padding:8px; border-radius:6px;"
        )
        return self.warning_banner

    def _on_item(self, index, pred, elapsed_ms):
        super()._on_item(index, pred, elapsed_ms)
        if self.results[index] is not None:
            self.results[index]["path"] = self.file_paths[index]

    def _on_item_err(self, index, msg):
        super()._on_item_err(index, msg)
        if self.results[index] is not None:
            self.results[index]["path"] = self.file_paths[index]

    def _show_image_preview(self, row, _column):
        if row < 0 or row >= len(self.results):
            return
        result = self.results[row]
        if result is None:
            return
        dialog = ImagePreviewDialog(self, result)
        dialog.exec_()

    def _update_summary_panel(self, stats):
        super()._update_summary_panel(stats)
        threshold = self.defect_threshold_spin.value()
        defect_pct = stats["defect_rate_pct"]
        if defect_pct > threshold:
            self.warning_banner.setText(
                f"⚠ 불량률 {defect_pct:.1f}%가 경계값 {threshold:.1f}%를 초과했습니다"
            )
            self.warning_banner.setStyleSheet(
                "background:#c0392b; color:white; font-size:13px; font-weight:bold;"
                "padding:8px; border-radius:6px;"
            )
        else:
            self.warning_banner.setText(
                f"✓ 불량률 {defect_pct:.1f}% — 경계값 {threshold:.1f}% 이내 (정상 범위)"
            )
            self.warning_banner.setStyleSheet(
                "background:#1e824c; color:white; font-size:13px; font-weight:bold;"
                "padding:8px; border-radius:6px;"
            )

    def _on_threshold_changed(self, _value):
        if self._last_rows:
            self._update_summary_panel(compute_batch_stats(self._last_rows))

    def _on_batch_done(self):
        super()._on_batch_done()
        if not self._last_rows:
            return
        stats = compute_batch_stats(self._last_rows)
        threshold = self.defect_threshold_spin.value()
        if stats["defect_rate_pct"] > threshold:
            QMessageBox.warning(
                self, "불량률 경계값 초과",
                f"불량률 {stats['defect_rate_pct']:.1f}%가 "
                f"경계값 {threshold:.1f}%를 초과했습니다.",
            )


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = FinalAddWindow()
    win.show()
    sys.exit(app.exec_())