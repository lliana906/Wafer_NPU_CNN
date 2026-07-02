"""
wafer_cnn_gui_launcher.py — Wafer NPU 통합 런처

기존에 따로따로 실행하던 3개의 독립 GUI를 하나의 진입점에서 버튼으로 골라
열 수 있게 묶은 런처다. 각 GUI의 코드는 전혀 수정하지 않고 그대로 import해서
그 MainWindow를 띄우기만 한다:

  ① 단일 이미지 추론   → wafer_cnn_gui.py       (MainWindow)
  ② 배치 추론(다중 이미지, CSV+PDF 저장, 이미지 미리보기, 불량률 임계값 경고)
                        → wafer_cnn_gui_csv.py   (FinalAddWindow)
  ③ 배치 결과 비교(과거 batch_summary_*.csv 여러 개 비교/추이 그래프)
                        → wafer_cnn_gui_compare.py (CompareWindow)

wafer_cnn_gui.py / wafer_cnn_gui_csv.py / wafer_cnn_gui_csv.py /
wafer_cnn_gui_compare.py 는 이 파일에서 손대지 않는다(그대로 import만 함).

실행: python wafer_cnn_gui_launcher.py
"""

import sys

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel, QPushButton,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

from wafer_cnn_gui import MainWindow as SingleInferenceWindow
from wafer_cnn_gui_csv import FinalAddWindow
from wafer_cnn_gui_compare import CompareWindow


class LauncherWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Wafer NPU Defect Classifier — 통합 런처")
        self.setMinimumSize(520, 460)
        self._open_windows = []  # 생성한 하위 창 참조 유지(가비지 컬렉션 방지)
        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(24, 24, 24, 24)
        root.setSpacing(16)

        title = QLabel("Wafer NPU Defect Classifier")
        f = QFont()
        f.setPointSize(18)
        f.setBold(True)
        title.setFont(f)
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color:#1abc9c;")
        root.addWidget(title)

        subtitle = QLabel("원하는 작업을 선택하세요")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("color:#888; font-size:12px;")
        root.addWidget(subtitle)

        root.addWidget(self._make_launch_button(
            "①  단일 이미지 추론",
            "이미지 한 장을 보드에 보내 즉시 결과 확인 (UART)",
            self._open_single,
        ))
        root.addWidget(self._make_launch_button(
            "②  배치 추론 (여러 이미지 → CSV + PDF)",
            "폴더/여러 파일을 한 번에 추론, 통계 요약·이미지 미리보기·불량률 경고 포함",
            self._open_batch,
        ))
        root.addWidget(self._make_launch_button(
            "③  배치 결과 비교",
            "저장된 과거 배치 요약(csv) 여러 개를 골라 수율/불량률 추이 비교",
            self._open_compare,
        ))

        root.addStretch()

        self.setStyleSheet("""
            QMainWindow,QWidget{background:#252525;color:#ddd;}
        """)

    def _make_launch_button(self, label, desc, slot):
        wrapper = QWidget()
        wl = QVBoxLayout(wrapper)
        wl.setContentsMargins(0, 0, 0, 0)
        wl.setSpacing(4)

        btn = QPushButton(label)
        btn.setFixedHeight(52)
        btn.setStyleSheet(
            "QPushButton{background:#1abc9c;color:white;font-size:15px;"
            "font-weight:bold;border-radius:8px;text-align:left;padding-left:16px;}"
            "QPushButton:hover{background:#16a085;}"
        )
        btn.clicked.connect(slot)
        wl.addWidget(btn)

        desc_label = QLabel(desc)
        desc_label.setStyleSheet("color:#999; font-size:11px; padding-left:4px;")
        desc_label.setWordWrap(True)
        wl.addWidget(desc_label)

        return wrapper

    # ── 버튼 ① : 단일 이미지 추론 창 ────────────────────────────────
    def _open_single(self):
        win = SingleInferenceWindow()
        self._open_windows.append(win)
        win.show()

    # ── 버튼 ② : 배치 추론(다중 이미지, CSV+PDF) 창 ──────────────────
    def _open_batch(self):
        win = FinalAddWindow()
        self._open_windows.append(win)
        win.show()

    # ── 버튼 ③ : 배치 결과 비교 창 ───────────────────────────────────
    def _open_compare(self):
        win = CompareWindow()
        self._open_windows.append(win)
        win.show()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = LauncherWindow()
    win.show()
    sys.exit(app.exec_())
