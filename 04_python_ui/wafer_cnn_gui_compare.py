"""
wafer_cnn_gui_compare.py — Wafer NPU 배치 결과 비교/분석 도구 (독립 실행)

wafer_cnn_gui_add.py와는 독립된 별도 도구다. 실시간 UART 추론이나 배치
처리는 전혀 하지 않고, 이미 저장된 과거 배치 결과 폴더의
batch_summary_{YYYYmmdd}_{HHMMSS}.csv 파일들만 읽어서 배치 간 수율/불량률을
비교·분석한다(오프라인 사후 분석 전용).

다크 테마 색상 팔레트(#252525/#1a1a1a/#1abc9c 등)와 클래스 정보(CLASS_NAMES 등)는
wafer_cnn_gui_many.py / wafer_cnn_gui_add.py와 통일감을 맞추기 위해
그대로 재사용한다.

batch_summary_*.csv 포맷(wafer_cnn_gui_add.py가 저장):
  검증 날짜, 검증 시간, 전체 처리 장수, 오류 장수, 정상분류 장수,
  수율(%), 불량률(%), 평균 처리시간(ms), 클래스명, 개수, 비율(%)
  (클래스 9개 행이 있고, 배치 단위 통계는 모든 행에 동일하게 반복 저장되어
   있어 첫 행만 읽으면 배치 전체 통계를 알 수 있다.)

실행: python wafer_cnn_gui_compare.py
"""

import os
import sys
import csv
import glob

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QPushButton, QLabel, QFileDialog,
    QListWidget, QListWidgetItem, QTableWidget, QTableWidgetItem,
    QHeaderView, QAbstractItemView, QMessageBox,
)
from PyQt5.QtCore import Qt, QEvent
from PyQt5.QtGui import QColor

import matplotlib
from matplotlib import font_manager as fm
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from wafer_cnn_gui_csv import CLASS_NAMES, NORMAL_IDX  # 팔레트/클래스 정보 재사용

NORMAL_NAME = CLASS_NAMES[NORMAL_IDX]
SUMMARY_CSV_GLOB = "batch_summary_*.csv"

# ── 한글 폰트 등록 (수율 추이 그래프의 "수율(%)" 등 한글 라벨용) ────────
_KOREAN_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
]
for _font_path in _KOREAN_FONT_CANDIDATES:
    if os.path.isfile(_font_path):
        fm.fontManager.addfont(_font_path)
        matplotlib.rcParams["font.family"] = fm.FontProperties(fname=_font_path).get_name()
        matplotlib.rcParams["axes.unicode_minus"] = False
        break


# ── 배치 요약 CSV 로딩 ─────────────────────────────────────────────────
def load_batch_summary(csv_path):
    """
    batch_summary_*.csv의 첫 데이터 행에서 배치 단위 통계만 읽어온다.
    형식이 다르거나 읽기 실패하면 None을 반환한다(비교 목록에서 조용히 제외 -
    이 파일이 아닌 다른 CSV가 섞여 있어도 에러 없이 넘어가기 위함).
    """
    try:
        with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
            row = next(csv.DictReader(f), None)
        if row is None:
            return None
        return {
            "path": csv_path,
            "file": os.path.basename(csv_path),
            "date": row["검증 날짜"],
            "time": row["검증 시간"],
            "total": int(row["전체 처리 장수"]),
            "yield_pct": float(row["수율(%)"]),
            "defect_pct": float(row["불량률(%)"]),
            "avg_ms": float(row["평균 처리시간(ms)"]),
        }
    except Exception:
        return None


def find_batch_summaries(folder):
    """폴더 내 batch_summary_*.csv를 전부 읽어 날짜/시간순으로 정렬해 반환한다."""
    entries = []
    for path in sorted(glob.glob(os.path.join(folder, SUMMARY_CSV_GLOB))):
        entry = load_batch_summary(path)
        if entry is not None:
            entries.append(entry)
    entries.sort(key=lambda e: (e["date"], e["time"]))
    return entries


# ── 메인 윈도우 ────────────────────────────────────────────────────────
class CompareWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Wafer NPU 배치 결과 비교 도구 (사후 분석)")
        self.setMinimumSize(1100, 800)
        self.entries = []
        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(10)
        root.setContentsMargins(12, 12, 12, 12)

        # ── 결과 폴더 선택 ─────────────────────────────────────────
        folder_grp = QGroupBox("결과 폴더")
        fl = QHBoxLayout(folder_grp)
        self.folder_label = QLabel("폴더를 선택하세요 (batch_summary_*.csv가 있는 폴더)")
        self.folder_label.setStyleSheet("color:#aaa; font-size:12px;")
        fl.addWidget(self.folder_label, 1)
        btn_browse = QPushButton("📁  폴더 선택")
        btn_browse.setFixedHeight(32)
        btn_browse.clicked.connect(self._choose_folder)
        fl.addWidget(btn_browse)
        root.addWidget(folder_grp)

        # ── 배치 목록(체크박스로 다중 선택) ──────────────────────────
        list_grp = QGroupBox("배치 목록 (체크한 항목을 비교)")
        ll = QVBoxLayout(list_grp)
        self.list_widget = QListWidget()
        self.list_widget.setStyleSheet(
            "QListWidget{background:#1a1a1a;color:#ddd;border:1px solid #444;"
            "font-size:13px;}"
            "QListWidget::item{padding:8px 6px; min-height:22px;}"
            "QListWidget::item:selected{background:#2a2a2a;}"
            # 체크박스 자체를 크게 + 어두운 배경에서도 잘 보이도록 대비 강화
            "QListWidget::indicator{width:20px; height:20px;"
            "border:2px solid #bbbbbb; border-radius:4px; background:#2a2a2a;}"
            "QListWidget::indicator:hover{border:2px solid #1abc9c;}"
            "QListWidget::indicator:checked{background:#1abc9c; border:2px solid #1abc9c;}"
        )
        # 체크박스의 정확한 아이콘 영역만이 아니라, 행 전체(텍스트 포함) 어디를
        # 클릭해도 체크/해제가 토글되도록 프레스 이벤트를 가로챈다. Qt 기본
        # 동작은 인디케이터의 정확한 픽셀 영역을 클릭해야만 토글되는데, 그
        # 영역이 스타일에 따라 미묘하게 달라져 사용자가 놓치기 쉽기 때문이다.
        self.list_widget.viewport().installEventFilter(self)
        # 체크 상태가 바뀔 때마다(위 이벤트 필터로 토글되든, 인디케이터를 정확히
        # 클릭하든) 배경색을 직접 칠한다 - QSS의 `::item:checked`는 리스트
        # 아이템의 체크 상태에 반응하지 않아(Qt 스타일 엔진 한계) 코드로 처리.
        self.list_widget.itemChanged.connect(self._on_item_check_changed)
        ll.addWidget(self.list_widget)

        self.empty_label = QLabel("비교할 이전 결과가 없습니다.")
        self.empty_label.setAlignment(Qt.AlignCenter)
        self.empty_label.setStyleSheet("color:#888; font-size:13px; padding:24px;")
        self.empty_label.setVisible(False)
        ll.addWidget(self.empty_label)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self.btn_compare = QPushButton("▶  선택 항목 비교")
        self.btn_compare.setFixedHeight(36)
        self.btn_compare.setEnabled(False)
        self.btn_compare.setStyleSheet(
            "QPushButton{background:#1abc9c;color:white;font-weight:bold;"
            "border-radius:6px;padding:0 16px;}"
            "QPushButton:disabled{background:#3a3a3a;color:#666;}"
            "QPushButton:hover{background:#16a085;}"
        )
        self.btn_compare.clicked.connect(self._compare_selected)
        btn_row.addWidget(self.btn_compare)
        ll.addLayout(btn_row)
        root.addWidget(list_grp, 2)

        # ── 최근 2개 배치 증감 배너 ──────────────────────────────────
        self.delta_banner = QLabel("")
        self.delta_banner.setAlignment(Qt.AlignCenter)
        self.delta_banner.setWordWrap(True)
        self.delta_banner.setVisible(False)
        root.addWidget(self.delta_banner)

        # ── 수율/불량률 비교 표 ──────────────────────────────────────
        table_grp = QGroupBox("수율 / 불량률 비교")
        tl = QVBoxLayout(table_grp)
        self.compare_table = QTableWidget(0, 4)
        self.compare_table.setHorizontalHeaderLabels(["날짜/시간", "전체 장수", "수율(%)", "불량률(%)"])
        self.compare_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.compare_table.setSelectionMode(QAbstractItemView.NoSelection)
        self.compare_table.verticalHeader().setVisible(False)
        hdr = self.compare_table.horizontalHeader()
        hdr.setSectionResizeMode(QHeaderView.Stretch)
        self.compare_table.setStyleSheet(
            "QTableWidget{background:#1a1a1a;color:#ddd;gridline-color:#3a3a3a;"
            "border:1px solid #444;}"
            "QHeaderView::section{background:#333;color:#eee;border:none;"
            "padding:5px;font-weight:bold;}"
        )
        tl.addWidget(self.compare_table)
        root.addWidget(table_grp, 2)

        # ── 수율 추이 꺾은선 그래프 (3개 이상 선택 시) ─────────────────
        self.chart_grp = QGroupBox("수율 추이 (3개 이상 선택 시 표시)")
        cl = QVBoxLayout(self.chart_grp)
        self.figure = Figure(figsize=(7, 3.2), dpi=100)
        self.figure.patch.set_facecolor("#252525")
        self.canvas = FigureCanvas(self.figure)
        cl.addWidget(self.canvas)
        self.chart_grp.setVisible(False)
        root.addWidget(self.chart_grp, 3)

        self.setStyleSheet("""
            QMainWindow,QWidget{background:#252525;color:#ddd;}
            QGroupBox{border:1px solid #444;border-radius:6px;
                      margin-top:8px;padding-top:8px;color:#eee;}
            QGroupBox::title{subcontrol-origin:margin;left:8px;}
            QPushButton{background:#3a3a3a;color:#ddd;
                        border:1px solid #555;border-radius:4px;padding:4px 8px;}
            QPushButton:hover{background:#4a4a4a;}
        """)

    # ── 체크박스: 행 전체 클릭 토글 + 체크 시 배경색 강조 ────────────────
    def eventFilter(self, obj, event):
        if (obj is self.list_widget.viewport()
                and event.type() == QEvent.MouseButtonPress
                and event.button() == Qt.LeftButton):
            item = self.list_widget.itemAt(event.pos())
            if item is not None:
                item.setCheckState(
                    Qt.Unchecked if item.checkState() == Qt.Checked else Qt.Checked
                )
                return True  # Qt 기본 인디케이터 판정이 또 토글하지 않도록 이벤트 소비
        return super().eventFilter(obj, event)

    def _on_item_check_changed(self, item):
        if item.checkState() == Qt.Checked:
            item.setBackground(QColor("#1e824c"))
        else:
            item.setBackground(QColor("#1a1a1a"))

    # ── 폴더 선택 / 목록 채우기 ────────────────────────────────────────
    def _choose_folder(self):
        default_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
        folder = QFileDialog.getExistingDirectory(self, "결과 폴더 선택", default_dir)
        if not folder:
            return
        self.folder_label.setText(folder)
        self.entries = find_batch_summaries(folder)
        self._populate_list()
        self._reset_compare_views()

    def _populate_list(self):
        self.list_widget.clear()
        has_entries = len(self.entries) > 0
        self.list_widget.setVisible(has_entries)
        self.empty_label.setVisible(not has_entries)
        self.btn_compare.setEnabled(has_entries)
        for e in self.entries:
            text = (
                f"{e['date']} {e['time']}  |  {e['total']}장  |  "
                f"수율 {e['yield_pct']:.1f}%  |  불량률 {e['defect_pct']:.1f}%  "
                f"({e['file']})"
            )
            item = QListWidgetItem(text)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            item.setData(Qt.UserRole, e)
            self.list_widget.addItem(item)

    def _reset_compare_views(self):
        self.compare_table.setRowCount(0)
        self.delta_banner.setVisible(False)
        self.chart_grp.setVisible(False)

    # ── 비교 실행 ──────────────────────────────────────────────────────
    def _compare_selected(self):
        selected = []
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item.checkState() == Qt.Checked:
                selected.append(item.data(Qt.UserRole))

        if not selected:
            QMessageBox.information(self, "안내", "비교할 배치를 하나 이상 체크하세요.")
            return

        selected.sort(key=lambda e: (e["date"], e["time"]))
        self._fill_compare_table(selected)
        self._update_delta_banner(selected)
        self._update_trend_chart(selected)

    def _fill_compare_table(self, selected):
        self.compare_table.setRowCount(len(selected))
        for r, e in enumerate(selected):
            self.compare_table.setItem(r, 0, QTableWidgetItem(f"{e['date']} {e['time']}"))
            self.compare_table.setItem(r, 1, QTableWidgetItem(str(e["total"])))
            self.compare_table.setItem(r, 2, QTableWidgetItem(f"{e['yield_pct']:.1f}"))
            self.compare_table.setItem(r, 3, QTableWidgetItem(f"{e['defect_pct']:.1f}"))

    def _update_delta_banner(self, selected):
        if len(selected) < 2:
            self.delta_banner.setVisible(False)
            return

        prev, latest = selected[-2], selected[-1]
        delta = latest["defect_pct"] - prev["defect_pct"]
        tag = f"(최근 2개 배치: {prev['date']} {prev['time']} → {latest['date']} {latest['time']})"

        if delta > 0:
            text = f"▲ 불량률 {delta:.1f}%p 증가 {tag}"
            color = "#c0392b"
        elif delta < 0:
            text = f"▼ 불량률 {abs(delta):.1f}%p 감소 {tag}"
            color = "#1e824c"
        else:
            text = f"– 불량률 변화 없음 {tag}"
            color = "#555555"

        self.delta_banner.setText(text)
        self.delta_banner.setStyleSheet(
            f"background:{color}; color:white; font-weight:bold; font-size:13px;"
            "padding:8px; border-radius:6px;"
        )
        self.delta_banner.setVisible(True)

    def _update_trend_chart(self, selected):
        if len(selected) < 3:
            self.chart_grp.setVisible(False)
            return

        self.chart_grp.setVisible(True)
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.set_facecolor("#1a1a1a")

        labels = [f"{e['date']}\n{e['time']}" for e in selected]
        yields = [e["yield_pct"] for e in selected]
        x = range(len(selected))

        ax.plot(x, yields, marker="o", color="#1abc9c", linewidth=2)
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, fontsize=8, color="#ddd")
        ax.set_ylabel("수율(%)", color="#ddd")
        ax.set_ylim(0, 100)
        ax.tick_params(colors="#ddd")
        ax.grid(True, alpha=0.25, color="#888")
        for spine in ax.spines.values():
            spine.set_color("#555")

        self.figure.tight_layout()
        self.canvas.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = CompareWindow()
    win.show()
    sys.exit(app.exec_())
