import sys
import pickle

from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
                             QWidget, QPushButton, QListWidget, QLabel, QScrollArea,
                             QFrame, QGridLayout, QListWidgetItem, QMessageBox, QLineEdit)

from PyQt5.QtCore import Qt, QThread, pyqtSignal, QPropertyAnimation, QEasingCurve, QRect
from PyQt5.QtGui import QFont, QPalette, QColor, QPainter, QPainterPath

from utils import *
from options import args
from models import model_factory
from dataloaders import dataloader_factory


class RoundedButton(QPushButton):
    def __init__(self, text, parent=None):
        super().__init__(text, parent)

        self.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                          stop: 0 #BB86FC, stop: 1 #9965E8);
                color: #FFFFFF;
                border: none;
                border-radius: 15px;
                padding: 12px 24px;
                font-size: 14px;
                font-weight: bold;
                min-height: 20px;
            }
            QPushButton:hover {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                          stop: 0 #03DAC5, stop: 1 #00BFA5);
            }
            QPushButton:pressed {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                          stop: 0 #8A5CF6, stop: 1 #7C4DDB);
            }
            QPushButton:disabled {
                background: #2A2A2A;
                color: #666666;
            }
        """)


class RoundedFrame(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                          stop: 0 #1E1E1E, stop: 1 #121212);
                border: 2px solid #BB86FC;
                border-radius: 15px;
                margin: 5px;
                padding: 10px;
            }
        """)


class RoundedLineEdit(QLineEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("""
            QLineEdit {
                background: #1E1E1E;
                border: 2px solid #BB86FC;
                border-radius: 15px;
                padding: 10px 15px;
                font-size: 14px;
                color: #FFFFFF;
            }
            QLineEdit:focus {
                border-color: #03DAC5;
                background: #252525;
                outline: none;
            }
            QLineEdit::placeholder {
                color: #B0B0B0;
            }
        """)


class RoundedListWidget(QListWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("""
            QListWidget {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                          stop: 0 #1E1E1E, stop: 1 #121212);
                border: 2px solid #BB86FC;
                border-radius: 15px;
                padding: 10px;
                font-size: 13px;
                selection-background-color: #BB86FC;
                selection-color: #FFFFFF;
                outline: none;
                color: #FFFFFF;
            }
            QListWidget::item {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                          stop: 0 #2A2A2A, stop: 1 #1E1E1E);
                border: 1px solid #444444;
                border-radius: 8px;
                padding: 8px;
                margin: 2px;
                color: #FFFFFF;
            }
            QListWidget::item:selected {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                          stop: 0 #BB86FC, stop: 1 #9965E8);
                color: #FFFFFF;
                border-color: #03DAC5;
            }
            QListWidget::item:hover {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                          stop: 0 #03DAC5, stop: 1 #00BFA5);
                border-color: #BB86FC;
                color: #FFFFFF;
            }
        """)


class ModelThread(QThread):
    prediction_ready = pyqtSignal(list, list)
    error_occurred = pyqtSignal(str)

    def __init__(self, model, favorite_anime_ids, dataset, id_to_anime):
        super().__init__()
        self.model = model
        self.favorite_anime_ids = favorite_anime_ids
        self.dataset = dataset
        self.id_to_anime = id_to_anime

    def run(self):
        try:
            if not self.favorite_anime_ids:
                self.error_occurred.emit("Please add some favorite animes first!")
                return

            smap = self.dataset["smap"]
            inverted_smap = {v: k for k, v in smap.items()}

            # Convert anime IDs to model format
            converted_ids = []
            for anime_id in self.favorite_anime_ids:
                if anime_id in smap:
                    converted_ids.append(smap[anime_id])

            if not converted_ids:
                self.error_occurred.emit("None of the selected animes are in the model vocabulary!")
                return

            target_len = 128
            padded = converted_ids + [0] * (target_len - len(converted_ids))
            input_tensor = torch.tensor(padded, dtype=torch.long).unsqueeze(0)

            # Get predictions - Daha fazla öneri al (favori animeler çıkarılacağı için)
            # Model vocabulary boyutuna göre ayarlanabilir, burada 100 alıyoruz
            max_predictions = min(100, len(inverted_smap))

            with torch.no_grad():
                logits = self.model(input_tensor)
                last_logits = logits[:, -1, :]
                top_scores, top_indices = torch.topk(last_logits, k=max_predictions, dim=1)

            # Convert back to anime names and filter out favorites
            recommendations = []
            scores = []

            for idx, score in zip(top_indices.numpy()[0], top_scores.detach().numpy()[0]):
                if idx in inverted_smap:
                    anime_id = inverted_smap[idx]
                    if anime_id in self.favorite_anime_ids:
                        continue
                    if str(anime_id) in self.id_to_anime:
                        anime_name = self.id_to_anime[str(anime_id)]
                        recommendations.append(anime_name)
                        scores.append(float(score))

                        # En iyi 20 öneriye ulaştığımızda dur
                        if len(recommendations) >= 20:
                            break

            if len(recommendations) < 20:
                print(f"Warning: Only found {len(recommendations)} recommendations after filtering favorites")

            self.prediction_ready.emit(recommendations, scores)

        except Exception as e:
            self.error_occurred.emit(f"Error during prediction: {str(e)}")


class AnimeRecommendationGUI(QMainWindow):
    def __init__(self, checkpoint_path, dataset_path, animes_path):
        super().__init__()
        self.model = None
        self.dataset = None
        self.id_to_anime = {}
        self.favorite_animes = []
        self.all_animes = []
        self.checkpoint_path = checkpoint_path
        self.dataset_path = dataset_path
        self.animes_path = animes_path

        self.init_ui()
        self.load_model_and_data()

    def init_ui(self):
        self.setWindowTitle("Anime Recommendation System")
        self.setGeometry(200, 100, 1520, 800)

        # Set dark theme with purple and teal accents
        self.setStyleSheet("""
            QMainWindow {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                          stop: 0 #1E1E1E, stop: 1 #121212);
                color: #FFFFFF;
            }
            QLabel {
                color: #BB86FC;
                font-weight: bold;
            }
        """)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # Left panel - Anime selection
        left_panel = self.create_left_panel()
        main_layout.addWidget(left_panel, 1)

        # Middle panel - Favorite animes
        middle_panel = self.create_middle_panel()
        main_layout.addWidget(middle_panel, 1)

        # Right panel - Recommendations
        right_panel = self.create_right_panel()
        main_layout.addWidget(right_panel, 1)

    def create_left_panel(self):
        frame = RoundedFrame()
        layout = QVBoxLayout(frame)

        title = QLabel("Available Animes")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #BB86FC; margin-bottom: 10px;")
        layout.addWidget(title)

        # Search box
        self.search_box = RoundedLineEdit()
        self.search_box.setPlaceholderText("Search for anime...")
        self.search_box.textChanged.connect(self.filter_anime_list)
        layout.addWidget(self.search_box)

        self.anime_list = RoundedListWidget()
        layout.addWidget(self.anime_list)

        self.add_favorite_btn = RoundedButton("Add to Favorites")
        self.add_favorite_btn.clicked.connect(self.add_to_favorites)
        self.add_favorite_btn.setEnabled(False)
        layout.addWidget(self.add_favorite_btn)

        return frame

    def create_middle_panel(self):
        frame = RoundedFrame()
        layout = QVBoxLayout(frame)

        title = QLabel("My Favorite Animes")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #BB86FC; margin-bottom: 10px;")
        layout.addWidget(title)

        self.favorites_list = RoundedListWidget()
        layout.addWidget(self.favorites_list)

        button_layout = QVBoxLayout()

        self.get_recommendations_btn = RoundedButton("Get Recommendations")
        self.get_recommendations_btn.clicked.connect(self.get_recommendations)
        self.get_recommendations_btn.setEnabled(False)
        button_layout.addWidget(self.get_recommendations_btn)

        self.clear_favorites_btn = RoundedButton("Clear Favorites")
        self.clear_favorites_btn.clicked.connect(self.clear_favorites)
        self.clear_favorites_btn.setEnabled(False)
        button_layout.addWidget(self.clear_favorites_btn)

        layout.addLayout(button_layout)

        return frame

    def create_right_panel(self):
        frame = RoundedFrame()
        layout = QVBoxLayout(frame)

        title = QLabel("Recommendations & Scores")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #BB86FC; margin-bottom: 10px;")
        layout.addWidget(title)

        self.recommendations_list = RoundedListWidget()
        layout.addWidget(self.recommendations_list)

        self.status_label = QLabel("Load your favorite animes and click 'Get Recommendations'")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("""
            QLabel {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                          stop: 0 #2A2A2A, stop: 1 #1E1E1E);
                color: #B0B0B0;
                border: 1px solid #03DAC5;
                border-radius: 10px;
                padding: 10px;
                font-style: italic;
            }
        """)
        layout.addWidget(self.status_label)

        return frame

    def load_model_and_data(self):
        try:
            self.status_label.setText("Loading model and data...")

            # Initialize args
            args.bert_max_len = 128

            # Load dataset from the provided path
            dataset_path = Path(self.dataset_path)
            with dataset_path.open('rb') as f:
                self.dataset = pickle.load(f)

            # Load anime data from the provided path
            with open(self.animes_path, "r", encoding="utf-8") as file:
                self.id_to_anime = json.load(file)

            # Initialize model
            train_loader, val_loader, test_loader = dataloader_factory(args)
            del train_loader
            del val_loader
            del test_loader
            self.model = model_factory(args)

            # Load model weights
            self.load_checkpoint()

            # Populate anime list
            self.populate_anime_list()

            self.status_label.setText("Model loaded successfully! Select your favorite animes.")
            self.add_favorite_btn.setEnabled(True)

        except Exception as e:
            self.status_label.setText(f"Error loading model: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to load model and data:\n{str(e)}")

    def load_checkpoint(self):
        try:
            checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            print(f"Model loaded from: {self.checkpoint_path}")
        except Exception as e:
            raise Exception(f"Failed to load checkpoint from {self.checkpoint_path}: {str(e)}")

    def populate_anime_list(self):
        # Get all anime names and sort them
        self.all_animes = [(int(k), v) for k, v in self.id_to_anime.items()]
        self.all_animes.sort(key=lambda x: x[1])  # Sort by name

        for anime_id, anime_name in self.all_animes:
            item = QListWidgetItem(f"{anime_name} (ID{anime_id})")
            item.setData(Qt.UserRole, anime_id)
            self.anime_list.addItem(item)

    def filter_anime_list(self):
        search_text = self.search_box.text().lower()

        for i in range(self.anime_list.count()):
            item = self.anime_list.item(i)
            item_text = item.text().lower()
            item.setHidden(search_text not in item_text)

    def add_to_favorites(self):
        current_item = self.anime_list.currentItem()
        if current_item:
            anime_name = current_item.text()
            anime_id = current_item.data(Qt.UserRole)

            if anime_id not in self.favorite_animes:
                self.favorite_animes.append(anime_id)

                fav_item = QListWidgetItem(anime_name)
                fav_item.setData(Qt.UserRole, anime_id)
                self.favorites_list.addItem(fav_item)

                self.get_recommendations_btn.setEnabled(True)
                self.clear_favorites_btn.setEnabled(True)

                # Clear search box
                self.search_box.clear()

                self.status_label.setText(f"Added '{anime_name}' to favorites!")
            else:
                self.status_label.setText(f"'{anime_name}' is already in favorites!")

    def clear_favorites(self):
        self.favorite_animes.clear()
        self.favorites_list.clear()
        self.recommendations_list.clear()

        self.get_recommendations_btn.setEnabled(False)
        self.clear_favorites_btn.setEnabled(False)

        self.status_label.setText("Favorites cleared! Add some animes to get recommendations.")

    def get_recommendations(self):
        if not self.favorite_animes:
            self.status_label.setText("Please add some favorite animes first!")
            return

        self.status_label.setText("Getting recommendations... Please wait.")
        self.get_recommendations_btn.setEnabled(False)

        # Start model prediction in separate thread
        self.model_thread = ModelThread(self.model, self.favorite_animes, self.dataset, self.id_to_anime)
        self.model_thread.prediction_ready.connect(self.show_recommendations)
        self.model_thread.error_occurred.connect(self.show_error)
        self.model_thread.start()

    def show_recommendations(self, recommendations, scores):
        self.recommendations_list.clear()

        for i, (anime_name, score) in enumerate(zip(recommendations, scores)):
            item_text = f"#{i + 1}: {anime_name}\nScore: {score:.4f}"
            item = QListWidgetItem(item_text)
            self.recommendations_list.addItem(item)

        self.status_label.setText(f"Found {len(recommendations)} recommendations based on your favorites!")
        self.get_recommendations_btn.setEnabled(True)

    def show_error(self, error_message):
        self.status_label.setText(error_message)
        self.get_recommendations_btn.setEnabled(True)
        QMessageBox.warning(self, "Warning", error_message)




def main():


    # Validate checkpoint file
    if not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint file not found: {args.checkpoint}")
        sys.exit(1)

    if not args.checkpoint.endswith('.pth'):
        print(f"Error: Checkpoint file must be a .pth file: {args.checkpoint}")
        sys.exit(1)

    # Validate dataset file
    if not Path(args.dataset).exists():
        print(f"Error: Dataset file not found: {args.dataset}")
        sys.exit(1)

    if not args.dataset.endswith('.pkl'):
        print(f"Error: Dataset file must be a .pkl file: {args.dataset}")
        sys.exit(1)

    # Validate animes file
    if not Path(args.animes).exists():
        print(f"Error: Animes file not found: {args.animes}")
        sys.exit(1)

    if not args.animes.endswith('.json'):
        print(f"Error: Animes file must be a .json file: {args.animes}")
        sys.exit(1)

    print(f"Using checkpoint: {args.checkpoint}")
    print(f"Using dataset: {args.dataset}")
    print(f"Using animes: {args.animes}")

    app = QApplication(sys.argv)

    # Set application style
    app.setStyle('Fusion')

    # Create and show the main window with all paths
    window = AnimeRecommendationGUI(args.checkpoint, args.dataset, args.animes)
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
