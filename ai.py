import sys
import os
import re
import time
import json
import html
import queue
import pickle
import hashlib
import datetime
import threading
import subprocess
import webbrowser
from urllib.parse import quote

# Чтобы print() с кириллицей не падал в консоли Windows
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
import cv2
import face_recognition
from deepface import DeepFace
import speech_recognition as sr
import pyttsx3
import requests

import torch
import torch.nn as nn
import torch.optim as optim

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QTextEdit, QPushButton,
                             QLineEdit)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap, QFont, QPainter, QPen, QColor

# Необязательно: pip install pymorphy3 — тогда "погода в чебоксарах" превратится в "Чебоксары"
try:
    import pymorphy3
    MORPH = pymorphy3.MorphAnalyzer()
except Exception:
    MORPH = None


# ==================================================================
#                         КОНФИГУРАЦИЯ
# ==================================================================
KNOWN_FACES_DIR = "known_faces"
ENCODINGS_FILE = "encodings.pkl"
INTENT_MODEL_FILE = "intent_model.pth"
NOTES_FILE = "notes.json"
CAMERA_INDEX = 0
LANGUAGE = "ru-RU"
USE_WHISPER = False

DEFAULT_CITY = "Чебоксары"
DEFAULT_MUSIC_QUERY = "Дима Билан Невозможное возможно"

FACE_SCALE = 0.5
FACE_TOLERANCE = 0.5
EMOTION_COOLDOWN = 3
GREET_COOLDOWN = 1800
HISTORY_LIMIT = 12

USE_OLLAMA = True
OLLAMA_HOST = "http://localhost:11434"
OLLAMA_MODEL = "qwen2.5:7b"

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
UNKNOWN = "Неизвестный"

FORBIDDEN_COMMANDS = [
    "rm -rf", "format", "del /f", "shutdown", "reboot",
    "mkfs", "dd if=", "> /dev/sda", "chmod -R 777 /",
    "отформатируй диск", "удали все файлы", "выключи компьютер", "перезагрузи компьютер",
]

EXIT_WORDS = {"выход", "выключись", "пока", "свидания", "завершить"}

EMOTION_RU = {
    "happy": "радость", "sad": "грусть", "angry": "злость",
    "neutral": "нейтрально", "surprise": "удивление",
    "fear": "страх", "disgust": "отвращение",
}


def ru_plural(n, forms):
    n = abs(int(n)) % 100
    if 11 <= n <= 19:
        return forms[2]
    n %= 10
    if n == 1:
        return forms[0]
    if 2 <= n <= 4:
        return forms[1]
    return forms[2]


def format_duration(sec):
    h, rem = divmod(int(sec), 3600)
    m, s = divmod(rem, 60)
    parts = []
    if h:
        parts.append(f"{h} {ru_plural(h, ('час', 'часа', 'часов'))}")
    if m:
        parts.append(f"{m} {ru_plural(m, ('минуту', 'минуты', 'минут'))}")
    if s:
        parts.append(f"{s} {ru_plural(s, ('секунду', 'секунды', 'секунд'))}")
    return " ".join(parts)


NUM_WORDS = {
    'один': 1, 'одну': 1, 'одна': 1, 'два': 2, 'две': 2, 'три': 3, 'четыре': 4,
    'пять': 5, 'шесть': 6, 'семь': 7, 'восемь': 8, 'девять': 9, 'десять': 10,
    'одиннадцать': 11, 'двенадцать': 12, 'тринадцать': 13, 'четырнадцать': 14,
    'пятнадцать': 15, 'шестнадцать': 16, 'семнадцать': 17, 'восемнадцать': 18,
    'девятнадцать': 19, 'двадцать': 20, 'тридцать': 30, 'сорок': 40,
    'пятьдесят': 50, 'шестьдесят': 60,
}


def parse_duration(text):
    tokens = re.findall(r'\d+|[а-яё]+', text.lower())
    total, i = 0, 0
    while i < len(tokens):
        tok = tokens[i]
        if tok.isdigit():
            num = int(tok)
            i += 1
        elif tok in NUM_WORDS:
            num = NUM_WORDS[tok]
            i += 1
            if num >= 20 and i < len(tokens) and 0 < NUM_WORDS.get(tokens[i], 0) < 10:
                num += NUM_WORDS[tokens[i]]
                i += 1
        elif tok == 'полчаса':
            total += 1800
            i += 1
            continue
        elif tok == 'час':
            total += 3600
            i += 1
            continue
        elif tok.startswith('минут'):
            total += 60
            i += 1
            continue
        elif tok.startswith('секунд'):
            total += 1
            i += 1
            continue
        else:
            i += 1
            continue

        if i < len(tokens):
            unit = tokens[i]
            if unit.startswith('сек'):
                total += num
                i += 1
            elif unit.startswith('мин'):
                total += num * 60
                i += 1
            elif unit.startswith('час'):
                total += num * 3600
                i += 1
    return total


def clean_for_tts(text):
    text = re.sub(r'[\U00010000-\U0010ffff]', '', text)
    text = re.sub(r'[*_#`>~]+', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def pretty_name(filename):
    stem = os.path.splitext(filename)[0]
    name = re.sub(r'[_\-\s]*\d+$', '', stem).replace('_', ' ').strip()
    return name or stem


CLASSES = ['weather', 'time', 'date', 'music', 'chat', 'open_app',
           'timer', 'note_add', 'note_read']

TRAINING_DATA = {
    'weather': [
        'какая погода в чебоксарах', 'погода в чебоксарах',
        'какая сейчас погода', 'покажи погоду', 'температура на улице',
        'какая погода', 'что там с погодой', 'прогноз погоды',
        'какая погода в москве', 'погода в питере', 'будет ли дождь',
        'нужен ли зонт', 'сколько градусов на улице',
    ],
    'time': [
        'сколько сейчас времени', 'который час', 'сколько времени',
        'какое время', 'подскажи время', 'время', 'который сейчас час',
    ],
    'date': [
        'какая сегодня дата', 'какое сегодня число', 'какой день',
        'скажи дату', 'какая дата', 'какой сегодня день недели',
        'сегодняшняя дата',
    ],
    'music': [
        'включи песню димы билана невозможно возможно',
        'включи диму билана', 'поставь невозможное возможно',
        'включи билана', 'хочу послушать диму билана',
        'включи песню', 'поставь музыку', 'включи музыку',
        'включи трек', 'поставь что-нибудь послушать',
    ],
    'chat': [
        'привет', 'как дела', 'расскажи анекдот', 'кто ты',
        'что ты умеешь', 'давай поболтаем', 'как тебя зовут',
        'расскажи о себе', 'привет как дела', 'спасибо',
        'как настроение', 'расскажи что-нибудь интересное', 'что нового',
        'помоги мне', 'объясни мне',
    ],
    'open_app': [
        'открой блокнот', 'запусти блокнот', 'открой калькулятор',
        'запусти калькулятор', 'открой браузер', 'запусти браузер',
        'открой проводник', 'запусти проводник', 'открой паинт',
        'запусти паинт',
    ],
    'timer': [
        'поставь таймер на пять минут', 'таймер на 10 минут', 'запусти таймер',
        'поставь таймер', 'засеки тридцать секунд', 'напомни через пять минут',
        'поставь таймер на час', 'разбуди меня через десять минут',
        'таймер на 30 секунд',
    ],
    'note_add': [
        'запомни что я люблю кофе', 'запомни', 'запиши заметку',
        'сделай заметку купить хлеб', 'запиши купить молоко',
        'добавь заметку', 'запомни что завтра встреча',
    ],
    'note_read': [
        'какие у меня заметки', 'прочитай заметки', 'что ты помнишь',
        'покажи заметки', 'что я просил запомнить', 'мои заметки',
        'прочитай мои заметки',
    ],
}


def tokenize(text):
    return re.findall(r'\w+', text.lower().replace('ё', 'е'))


def build_vocab():
    vocab = {'<pad>': 0, '<unk>': 1}
    for examples in TRAINING_DATA.values():
        for text in examples:
            for w in tokenize(text):
                if w not in vocab:
                    vocab[w] = len(vocab)
    return vocab


VOCAB = build_vocab()


def training_hash():
    payload = json.dumps({'v': 2, 'c': CLASSES, 'd': TRAINING_DATA},
                         sort_keys=True, ensure_ascii=False)
    return hashlib.md5(payload.encode('utf-8')).hexdigest()


class IntentNet(nn.Module):
    def __init__(self, vocab_size, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 32, padding_idx=0)
        self.fc1 = nn.Linear(32, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        mask = (x != 0).unsqueeze(-1).float()
        pooled = (self.embedding(x) * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return self.fc2(self.dropout(self.relu(self.fc1(pooled))))


def text_to_indices(text):
    return [VOCAB.get(w, VOCAB['<unk>']) for w in tokenize(text)]


def train_intent_classifier():
    print("\n" + "=" * 60)
    print("[TRAIN] ОБУЧЕНИЕ ИНТЕНТ-КЛАССИФИКАТОРА")
    print("=" * 60)
    torch.manual_seed(42)

    X, y = [], []
    for cls_idx, cls_name in enumerate(CLASSES):
        for text in TRAINING_DATA[cls_name]:
            X.append(text_to_indices(text))
            y.append(cls_idx)

    max_len = max(len(x) for x in X)
    X_tensor = torch.tensor([x + [0] * (max_len - len(x)) for x in X],
                            dtype=torch.long, device=DEVICE)
    y_tensor = torch.tensor(y, dtype=torch.long, device=DEVICE)

    model = IntentNet(len(VOCAB), len(CLASSES)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    epochs = 250
    for epoch in range(epochs):
        model.train()
        drop = (torch.rand(X_tensor.shape, device=DEVICE) < 0.15) & (X_tensor != 0)
        X_aug = X_tensor.masked_fill(drop, VOCAB['<unk>'])

        optimizer.zero_grad()
        out = model(X_aug)
        loss = criterion(out, y_tensor)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            print(f"  Эпоха {epoch + 1:3d}/{epochs} | Loss: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        acc = (model(X_tensor).argmax(1) == y_tensor).float().mean().item() * 100
    print(f"  Итоговая точность на обучающих данных: {acc:.1f}%")

    torch.save({'model': model.state_dict(), 'vocab': dict(VOCAB), 'hash': training_hash()},
               INTENT_MODEL_FILE)
    print("[OK] Модель обучена и сохранена!\n")
    return model


def load_or_train_model():
    if os.path.exists(INTENT_MODEL_FILE):
        try:
            data = torch.load(INTENT_MODEL_FILE, map_location=DEVICE, weights_only=True)
            if data.get('hash') == training_hash():
                VOCAB.clear()
                VOCAB.update(data['vocab'])
                model = IntentNet(len(VOCAB), len(CLASSES)).to(DEVICE)
                model.load_state_dict(data['model'])
                model.eval()
                print("[OK] Загружена обученная модель интентов.")
                return model
            print("[INFO] Обучающие данные изменились — переобучаю модель.")
        except Exception as e:
            print(f"[WARN] Не удалось загрузить модель ({e}) — переобучаю.")
    return train_intent_classifier()


def predict_intent(text, model):
    tokens = tokenize(text)
    if not any(w in VOCAB for w in tokens):
        return 'chat', 0.0
    x = torch.tensor([text_to_indices(text)], dtype=torch.long, device=DEVICE)
    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
        conf, pred = torch.max(probs, 0)
    return CLASSES[pred.item()], conf.item()


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray, list)
    status_signal = pyqtSignal(str)
    update_person_signal = pyqtSignal(str, str)

    def __init__(self):
        super().__init__()
        self.running = True
        self.known_face_encodings = []
        self.known_face_names = []
        self.last_emotion = {}
        self._emotion_busy = False
        self._last_emotion_ts = 0.0

    def _faces_signature(self):
        sig = []
        for fn in sorted(os.listdir(KNOWN_FACES_DIR)):
            if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                st = os.stat(os.path.join(KNOWN_FACES_DIR, fn))
                sig.append((fn, int(st.st_mtime), st.st_size))
        return sig

    def load_known_faces(self):
        if not os.path.exists(KNOWN_FACES_DIR):
            os.makedirs(KNOWN_FACES_DIR)
            msg = f"Создана папка {KNOWN_FACES_DIR}. Положи туда фото (имя файла = имя человека)."
            print(f"[INFO] {msg}")
            self.status_signal.emit(msg)
            return

        signature = self._faces_signature()

        if os.path.exists(ENCODINGS_FILE):
            try:
                with open(ENCODINGS_FILE, "rb") as f:
                    data = pickle.load(f)
                if data.get("signature") == signature:
                    self.known_face_encodings = data["encodings"]
                    self.known_face_names = data["names"]
                    print(f"[INFO] Загружено {len(self.known_face_names)} лиц из кеша.")
                    return
                print("[INFO] Фото в папке изменились — пересканирую.")
            except Exception as e:
                print(f"[WARN] Кеш не загружен: {e}. Пересканирую папку.")

        print("[INFO] Сканирую папку с лицами...")
        for filename, _, _ in signature:
            path = os.path.join(KNOWN_FACES_DIR, filename)
            try:
                image = face_recognition.load_image_file(path)
                encodings = face_recognition.face_encodings(image)
                if encodings:
                    self.known_face_encodings.append(encodings[0])
                    name = pretty_name(filename)
                    self.known_face_names.append(name)
                    print(f"[OK] Загружено лицо: {name}")
                else:
                    print(f"[WARN] На фото {filename} лицо не найдено.")
            except Exception as e:
                print(f"[ERROR] Ошибка загрузки {filename}: {e}")

        try:
            with open(ENCODINGS_FILE, "wb") as f:
                pickle.dump({"signature": signature,
                             "encodings": self.known_face_encodings,
                             "names": self.known_face_names}, f)
            print("[INFO] Кеш лиц сохранён.")
        except Exception as e:
            print(f"[ERROR] Не удалось сохранить кеш: {e}")

    def _identify(self, encoding):
        if not self.known_face_encodings:
            return UNKNOWN
        dists = face_recognition.face_distance(self.known_face_encodings, encoding)
        best = int(np.argmin(dists))
        return self.known_face_names[best] if dists[best] <= FACE_TOLERANCE else UNKNOWN

    def _analyze_emotion(self, crop, name):
        try:
            res = DeepFace.analyze(crop, actions=['emotion'],
                                   enforce_detection=False, silent=True)
            if isinstance(res, list):
                res = res[0]
            self.last_emotion[name] = res['dominant_emotion']
        except Exception as e:
            print(f"[EMOTION] {e}")
        finally:
            self._emotion_busy = False

    def _handle_faces(self, frame, locations, names):
        primary = next(((loc, n) for loc, n in zip(locations, names) if n != UNKNOWN), None)
        if primary is None:
            self.update_person_signal.emit(names[0] if names else "", "")
            return

        loc, name = primary
        self.update_person_signal.emit(name, self.last_emotion.get(name, ""))

        now = time.time()
        if not self._emotion_busy and now - self._last_emotion_ts > EMOTION_COOLDOWN:
            k = int(round(1 / FACE_SCALE))
            top, right, bottom, left = [v * k for v in loc]
            m = int((bottom - top) * 0.2)
            h, w = frame.shape[:2]
            crop = frame[max(0, top - m):min(h, bottom + m),
                         max(0, left - m):min(w, right + m)].copy()
            if crop.size:
                self._last_emotion_ts = now
                self._emotion_busy = True
                threading.Thread(target=self._analyze_emotion,
                                 args=(crop, name), daemon=True).start()

    def run(self):
        self.load_known_faces()

        backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
        cap = cv2.VideoCapture(CAMERA_INDEX, backend)
        if not cap.isOpened():
            self.status_signal.emit("Ошибка: не удалось открыть камеру.")
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        k = int(round(1 / FACE_SCALE))
        frame_idx = 0
        overlay = []

        while self.running:
            ret, frame = cap.read()
            if not ret:
                self.status_signal.emit("Ошибка: камера перестала отдавать кадры.")
                break

            if frame_idx % 2 == 0:
                small = cv2.resize(frame, (0, 0), fx=FACE_SCALE, fy=FACE_SCALE)
                rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                locations = face_recognition.face_locations(rgb_small)
                encodings = face_recognition.face_encodings(rgb_small, locations)
                names = [self._identify(e) for e in encodings]

                overlay = []
                for (top, right, bottom, left), name in zip(locations, names):
                    label = name
                    emo = self.last_emotion.get(name)
                    if emo:
                        label += f" ({EMOTION_RU.get(emo, emo)})"
                    overlay.append((top * k, right * k, bottom * k, left * k, label))

                self._handle_faces(frame, locations, names)
            frame_idx += 1

            self.change_pixmap_signal.emit(frame, overlay)
            time.sleep(0.01)

        cap.release()

    def stop(self):
        self.running = False
        self.wait(3000)


class TTSWorker(threading.Thread):
    """На Windows SAPI5 работает только один раз на объекте,
    поэтому движок пересоздаётся для каждой фразы."""
    def __init__(self):
        super().__init__(daemon=True)
        self.q = queue.Queue()
        self.stopped = False

    @staticmethod
    def _make_engine():
        engine = pyttsx3.init()
        engine.setProperty('rate', 180)
        found = False
        for v in engine.getProperty('voices'):
            try:
                if 'ru' in str(v.languages).lower() or 'russian' in v.name.lower():
                    engine.setProperty('voice', v.id)
                    found = True
                    break
            except Exception:
                pass
        if not found:
            print("[WARN] Русский голос не найден — установи его в настройках системы.")
        return engine

    def run(self):
        while True:
            item = self.q.get()
            try:
                if item is None:
                    break
                text = item
                engine = None
                try:
                    engine = self._make_engine()
                    engine.say(text)
                    engine.runAndWait()
                except Exception as e:
                    print(f"[TTS ERROR] {e}")
                finally:
                    if engine is not None:
                        try:
                            engine.stop()
                        except Exception:
                            pass
                        del engine
            finally:
                self.q.task_done()

    def say(self, text, wait=True):
        if self.stopped or not text:
            return
        self.q.put(text)
        if wait:
            self.q.join()

    def wait_idle(self):
        if not self.stopped:
            self.q.join()

    def stop(self):
        self.stopped = True
        try:
            while True:
                self.q.get_nowait()
                self.q.task_done()
        except queue.Empty:
            pass
        self.q.put(None)


WEATHER_CODES = {
    0: 'ясно', 1: 'преимущественно ясно', 2: 'переменная облачность', 3: 'пасмурно',
    45: 'туман', 48: 'изморозь', 51: 'слабая морось', 53: 'морось', 55: 'сильная морось',
    56: 'ледяная морось', 57: 'ледяная морось', 61: 'небольшой дождь', 63: 'дождь',
    65: 'сильный дождь', 66: 'ледяной дождь', 67: 'ледяной дождь', 71: 'небольшой снег',
    73: 'снег', 75: 'сильный снег', 77: 'снежная крупа', 80: 'небольшие ливни',
    81: 'ливни', 82: 'сильные ливни', 85: 'снегопад', 86: 'сильный снегопад',
    95: 'гроза', 96: 'гроза с градом', 99: 'гроза с сильным градом',
}

APP_ALIASES = {
    "блокнот": ("блокнот", "notepad"),
    "калькулятор": ("калькулятор",),
    "проводник": ("проводник",),
    "паинт": ("паинт", "пейнт", "paint"),
}
APP_COMMANDS = {
    "блокнот":     {"win": ["notepad"],  "mac": ["open", "-a", "TextEdit"],   "linux": ["gedit"]},
    "калькулятор": {"win": ["calc"],     "mac": ["open", "-a", "Calculator"], "linux": ["gnome-calculator"]},
    "проводник":   {"win": ["explorer"], "mac": ["open", "."],                "linux": ["xdg-open", "."]},
    "паинт":       {"win": ["mspaint"],  "mac": ["open", "-a", "Preview"],    "linux": ["kolourpaint"]},
}


class AudioThread(QThread):
    transcription_signal = pyqtSignal(str, str)
    status_signal = pyqtSignal(str)
    quit_signal = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.running = True
        self.ready = False
        self.recognizer = sr.Recognizer()
        self.mic = None
        self.intent_model = None
        self.whisper_model = None

        self.history = []
        self.timers = []
        self.text_queue = queue.Queue()
        self.ctx = {"name": "", "emotion": "", "ts": 0.0}
        self.greeted = {}

        self.tts = TTSWorker()
        self.tts.start()

    def say(self, text, wait=True):
        print(f"[Ассистент] {text}")
        self.transcription_signal.emit("assistant", text)
        self.tts.say(clean_for_tts(text), wait)

    def submit_text(self, text):
        self.text_queue.put(text)

    @pyqtSlot(str, str)
    def on_person(self, name, emotion):
        if not name:
            return
        self.ctx = {"name": name, "emotion": emotion, "ts": time.time()}
        if name != UNKNOWN and self.ready and self.running:
            now = time.time()
            if now - self.greeted.get(name, 0) > GREET_COOLDOWN:
                self.greeted[name] = now
                self.say(f"Привет, {name}! Рад тебя видеть.", wait=False)

    def _init_mic(self):
        try:
            self.mic = sr.Microphone()
            with self.mic as src:
                self.status_signal.emit("Калибрую микрофон...")
                self.recognizer.adjust_for_ambient_noise(src, duration=1)
        except Exception as e:
            self.mic = None
            msg = f"Микрофон недоступен ({e}). Можно писать команды в поле ввода."
            print(f"[ERROR] {msg}")
            self.transcription_signal.emit("assistant", msg)

    def _check_ollama(self):
        try:
            r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=3)
            if r.status_code == 200:
                names = [m.get("name", "") for m in r.json().get("models", [])]
                if any(OLLAMA_MODEL in n for n in names):
                    print(f"[OK] Ollama доступна. Модель '{OLLAMA_MODEL}' найдена.")
                else:
                    msg = (f"Модель '{OLLAMA_MODEL}' не найдена в Ollama. Доступные: {names}. "
                           f"Скачай её: ollama pull {OLLAMA_MODEL}")
                    print(f"[WARN] {msg}")
                    self.transcription_signal.emit("assistant", msg)
            else:
                print(f"[WARN] Ollama ответила кодом {r.status_code}")
        except requests.exceptions.ConnectionError:
            msg = "Не удалось подключиться к Ollama. Запусти сервер: ollama serve"
            print(f"[ERROR] {msg}")
            self.transcription_signal.emit("assistant", msg)
        except Exception as e:
            print(f"[ERROR] Ошибка проверки Ollama: {e}")

    @staticmethod
    def _city_candidates(word):
        cands = []
        if MORPH:
            try:
                cands.append(MORPH.parse(word)[0].normal_form)
            except Exception:
                pass
        cands.append(word)
        for cut in (1, 2, 3):
            if len(word) - cut >= 4:
                cands.append(word[:-cut])
        return list(dict.fromkeys(cands))

    @staticmethod
    def _geocode(name):
        r = requests.get("https://geocoding-api.open-meteo.com/v1/search",
                         params={"name": name, "count": 1, "language": "ru"}, timeout=5)
        results = r.json().get("results")
        return results[0] if results else None

    def handle_weather(self, text):
        query = DEFAULT_CITY.lower()
        m = re.search(r'\bв\s+(?:городе\s+)?([а-яё\-]+)', text.lower())
        if m:
            query = m.group(1)
        try:
            place = None
            for cand in self._city_candidates(query):
                place = self._geocode(cand)
                if place:
                    break
            if not place:
                self.say(f"Не нашёл город {query}.")
                return

            r = requests.get("https://api.open-meteo.com/v1/forecast", params={
                "latitude": place["latitude"], "longitude": place["longitude"],
                "current": "temperature_2m,weather_code,wind_speed_10m",
                "wind_speed_unit": "ms", "timezone": "auto"}, timeout=5)
            cur = r.json()["current"]
            temp = round(cur["temperature_2m"])
            wind = round(cur["wind_speed_10m"])
            desc = WEATHER_CODES.get(cur["weather_code"], "без подробностей")
            sign = "минус " if temp < 0 else ""
            self.say(f"{place['name']}: {desc}, {sign}{abs(temp)} "
                     f"{ru_plural(temp, ('градус', 'градуса', 'градусов'))}, "
                     f"ветер {wind} {ru_plural(wind, ('метр', 'метра', 'метров'))} в секунду.")
        except Exception as e:
            print(f"[WEATHER] {e}")
            self.say("Не удалось получить погоду.")

    def handle_time(self):
        self.say(f"Сейчас {datetime.datetime.now().strftime('%H:%M')}")

    def handle_date(self):
        now = datetime.datetime.now()
        days = ['понедельник', 'вторник', 'среда', 'четверг',
                'пятница', 'суббота', 'воскресенье']
        months = ['января', 'февраля', 'марта', 'апреля', 'мая', 'июня',
                  'июля', 'августа', 'сентября', 'октября', 'ноября', 'декабря']
        self.say(f"Сегодня {now.day} {months[now.month - 1]}, {days[now.weekday()]}, {now.year} года.")

    def handle_music(self, text):
        query = DEFAULT_MUSIC_QUERY
        tl = text.lower()
        for marker in ["включи", "поставь", "хочу послушать"]:
            if marker in tl:
                rest = tl.split(marker, 1)[1]
                rest = re.sub(r'\b(пожалуйста|песню|песни|музыку|трек|что-нибудь|послушать)\b',
                              ' ', rest)
                rest = re.sub(r'\s+', ' ', rest).strip()
                if rest:
                    query = rest
                break
        self.say(f"Открываю поиск: {query}")
        webbrowser.open('https://www.youtube.com/results?search_query=' + quote(query))

    def handle_open_app(self, text):
        tl = text.lower()
        if "браузер" in tl:
            self.say("Открываю браузер.")
            webbrowser.open("https://www.google.com")
            return

        plat = "win" if sys.platform.startswith("win") else "mac" if sys.platform == "darwin" else "linux"
        for app, aliases in APP_ALIASES.items():
            if any(a in tl for a in aliases):
                try:
                    subprocess.Popen(APP_COMMANDS[app][plat])
                    self.say(f"Открываю {app}.")
                except Exception as e:
                    print(f"[APP] {e}")
                    self.say(f"Не получилось открыть {app}.")
                return
        self.say("Не знаю такого приложения.")

    def handle_timer(self, text):
        seconds = parse_duration(text)
        if seconds <= 0:
            self.say("На сколько поставить таймер? Скажи, например: таймер на пять минут.")
            return
        t = threading.Timer(seconds, self._timer_done, args=(seconds,))
        t.daemon = True
        t.start()
        self.timers.append(t)
        self.say(f"Таймер на {format_duration(seconds)} запущен.")

    def _timer_done(self, seconds):
        if self.running:
            self.say(f"Таймер на {format_duration(seconds)} сработал!", wait=False)

    @staticmethod
    def _load_notes():
        try:
            with open(NOTES_FILE, encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    @staticmethod
    def _save_notes(notes):
        with open(NOTES_FILE, "w", encoding="utf-8") as f:
            json.dump(notes, f, ensure_ascii=False, indent=2)

    def handle_note_add(self, text):
        skip = {'запомни', 'запиши', 'сделай', 'добавь', 'заметку', 'заметка', 'что', 'пожалуйста'}
        words = text.split()
        while words and words[0].lower() in skip:
            words.pop(0)
        content = " ".join(words).strip()
        if not content:
            self.say("Что именно запомнить?")
            return
        notes = self._load_notes()
        notes.append({"text": content, "time": datetime.datetime.now().isoformat(timespec="minutes")})
        self._save_notes(notes)
        self.say(f"Запомнил: {content}")

    def handle_note_read(self):
        notes = self._load_notes()
        if not notes:
            self.say("Заметок пока нет.")
            return
        last = notes[-5:]
        self.say(f"Всего заметок: {len(notes)}. Последние: " + "; ".join(n["text"] for n in last))

    def _system_prompt(self):
        parts = ["Ты — дружелюбный голосовой ассистент. Отвечай кратко, по-русски, "
                 "в разговорном стиле, максимум 2-3 предложения. "
                 "Не используй markdown, списки и эмодзи — твой ответ будет озвучен."]
        c = self.ctx
        if c["name"] and c["name"] != UNKNOWN and time.time() - c["ts"] < 15:
            parts.append(f"Через камеру ты видишь собеседника: {c['name']}.")
            if c["emotion"]:
                parts.append(f"Сейчас у него эмоция: {EMOTION_RU.get(c['emotion'], c['emotion'])}. "
                             "Если уместно, учитывай его настроение.")
        notes = self._load_notes()[-5:]
        if notes:
            parts.append("Заметки пользователя: " + "; ".join(n["text"] for n in notes) + ".")
        return " ".join(parts)

    def handle_chat(self, text):
        if not USE_OLLAMA:
            self.say("Модуль диалога отключён. Включи USE_OLLAMA в настройках.")
            return

        print(f"[OLLAMA] Запрос: {text}")
        self.status_signal.emit("Думаю...")
        messages = ([{"role": "system", "content": self._system_prompt()}]
                    + self.history + [{"role": "user", "content": text}])
        try:
            r = requests.post(f"{OLLAMA_HOST}/api/chat", json={
                "model": OLLAMA_MODEL,
                "messages": messages,
                "stream": False,
                "keep_alive": "30m",
                "options": {"temperature": 0.7, "num_predict": 200},
            }, timeout=120)
            r.raise_for_status()
            answer = r.json().get("message", {}).get("content", "").strip()
            if answer:
                self.history += [{"role": "user", "content": text},
                                 {"role": "assistant", "content": answer}]
                self.history = self.history[-HISTORY_LIMIT:]
                self.say(answer)
            else:
                self.say("Модель вернула пустой ответ.")
        except requests.exceptions.Timeout:
            print("[OLLAMA] Таймаут.")
            self.say("Нейросеть задумалась. Попробуй ещё раз.")
        except requests.exceptions.ConnectionError:
            print("[OLLAMA] Нет соединения.")
            self.say("Сервер Ollama не отвечает. Проверь, запущен ли он.")
        except requests.exceptions.HTTPError as e:
            print(f"[OLLAMA] {e} | {getattr(e.response, 'text', '')[:200]}")
            self.say("Ollama вернула ошибку. Проверь, скачана ли модель.")
        except Exception as e:
            print(f"[OLLAMA] {e}")
            self.say("Ошибка при обращении к нейросети.")

    def process(self, text):
        tl = text.lower()
        for forbidden in FORBIDDEN_COMMANDS:
            if forbidden in tl:
                self.say(f"Команда '{text}' заблокирована в целях безопасности.")
                return

        intent, conf = predict_intent(text, self.intent_model)
        print(f"[INTENT] {intent} ({conf * 100:.1f}%)")
        self.status_signal.emit(f"Интент: {intent} ({conf * 100:.0f}%)")

        if conf < 0.5:
            if USE_OLLAMA:
                print("[INFO] Низкая уверенность, отправляю в Ollama.")
                self.handle_chat(text)
            else:
                self.say("Не совсем понял, повтори, брат.")
            return

        handlers = {
            'weather':   lambda: self.handle_weather(text),
            'time':      self.handle_time,
            'date':      self.handle_date,
            'music':     lambda: self.handle_music(text),
            'open_app':  lambda: self.handle_open_app(text),
            'timer':     lambda: self.handle_timer(text),
            'note_add':  lambda: self.handle_note_add(text),
            'note_read': self.handle_note_read,
            'chat':      lambda: self.handle_chat(text),
        }
        handlers[intent]()

    def listen_once(self):
        with self.mic as src:
            try:
                audio = self.recognizer.listen(src, timeout=3, phrase_time_limit=8)
            except sr.WaitTimeoutError:
                return ""

        if USE_WHISPER:
            if self.whisper_model is None:
                try:
                    self.status_signal.emit("Загружаю Whisper...")
                    import whisper
                    self.whisper_model = whisper.load_model("small", device=DEVICE)
                except Exception as e:
                    print(f"[WHISPER] {e}")
                    self.whisper_model = False
            if self.whisper_model:
                try:
                    with open("_temp.wav", "wb") as f:
                        f.write(audio.get_wav_data())
                    result = self.whisper_model.transcribe(
                        "_temp.wav", language="ru", fp16=(DEVICE == 'cuda'))
                    return result["text"].strip()
                finally:
                    if os.path.exists("_temp.wav"):
                        os.remove("_temp.wav")

        try:
            return self.recognizer.recognize_google(audio, language=LANGUAGE)
        except sr.UnknownValueError:
            return ""
        except sr.RequestError as e:
            self.status_signal.emit(f"Ошибка сервиса распознавания: {e}")
            return ""

    def _next_input(self):
        try:
            return self.text_queue.get_nowait()
        except queue.Empty:
            pass
        if self.mic is None:
            try:
                return self.text_queue.get(timeout=0.5)
            except queue.Empty:
                return ""
        self.tts.wait_idle()
        return self.listen_once()

    @staticmethod
    def _is_exit(text):
        tokens = tokenize(text)
        return len(tokens) <= 3 and any(t in EXIT_WORDS for t in tokens)

    def run(self):
        self.status_signal.emit("Загружаю нейросеть интентов...")
        self.intent_model = load_or_train_model()
        if USE_OLLAMA:
            self._check_ollama()
        self._init_mic()

        self.ready = True
        self.status_signal.emit("Слушаю..." if self.mic else "Микрофона нет — пиши команды в поле ввода.")
        self.say("Система готова к работе, брат. Слушаю вас.")

        while self.running:
            try:
                text = self._next_input()
                if not text:
                    continue

                print(f"[Вы] {text}")
                self.transcription_signal.emit("user", text)

                if self._is_exit(text):
                    self.say("До связи, брат!")
                    self.running = False
                    self.quit_signal.emit()
                    break

                self.process(text)
                self.status_signal.emit("Слушаю..." if self.mic else "Жду команду...")

            except Exception as e:
                print(f"[AUDIO ERROR] {e}")
                time.sleep(0.5)

    def stop(self):
        self.running = False
        for t in self.timers:
            t.cancel()
        self.tts.stop()
        if not self.wait(3000):
            self.terminate()
            self.wait(1000)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Нейросеть-Ассистент (DeepSeek Edition + Ollama)")
        self.setGeometry(100, 100, 1200, 700)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        video_widget = QWidget()
        video_layout = QVBoxLayout(video_widget)

        self.video_label = QLabel("Инициализация камеры...")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black; color: white;")
        self.video_label.setMinimumSize(640, 480)
        video_layout.addWidget(self.video_label)

        self.person_info = QLabel("Личность: - | Эмоция: -")
        self.person_info.setFont(QFont("Arial", 12))
        self.person_info.setStyleSheet("padding: 5px; background-color: #f0f0f0;")
        video_layout.addWidget(self.person_info)

        main_layout.addWidget(video_widget, 2)

        chat_widget = QWidget()
        chat_layout = QVBoxLayout(chat_widget)

        chat_label = QLabel("Транскрибация разговора:")
        chat_label.setFont(QFont("Arial", 11, QFont.Bold))
        chat_layout.addWidget(chat_label)

        self.chat_history = QTextEdit()
        self.chat_history.setReadOnly(True)
        self.chat_history.setFont(QFont("Arial", 10))
        chat_layout.addWidget(self.chat_history)

        input_layout = QHBoxLayout()
        self.input_line = QLineEdit()
        self.input_line.setPlaceholderText("Или напиши команду и нажми Enter...")
        self.btn_send = QPushButton("Отправить")
        input_layout.addWidget(self.input_line)
        input_layout.addWidget(self.btn_send)
        chat_layout.addLayout(input_layout)

        btn_layout = QHBoxLayout()
        self.btn_clear = QPushButton("Очистить чат")
        btn_layout.addWidget(self.btn_clear)
        self.btn_stop = QPushButton("Остановить")
        self.btn_stop.clicked.connect(self.stop_all)
        btn_layout.addWidget(self.btn_stop)
        chat_layout.addLayout(btn_layout)

        main_layout.addWidget(chat_widget, 1)

        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Загрузка...")

        self.video_thread = VideoThread()
        self.audio_thread = AudioThread()

        self.video_thread.change_pixmap_signal.connect(self.update_image)
        self.video_thread.update_person_signal.connect(self.update_person_info)
        self.video_thread.update_person_signal.connect(self.audio_thread.on_person)
        self.video_thread.status_signal.connect(self.update_status)

        self.audio_thread.transcription_signal.connect(self.add_to_chat)
        self.audio_thread.status_signal.connect(self.update_status)
        self.audio_thread.quit_signal.connect(self.close)

        self.btn_clear.clicked.connect(self.clear_chat)
        self.btn_send.clicked.connect(self.send_text)
        self.input_line.returnPressed.connect(self.send_text)

        self.video_thread.start()
        self.audio_thread.start()

    def update_image(self, cv_img, faces):
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()

        painter = QPainter(img)
        painter.setPen(QPen(QColor(0, 255, 0), 2))
        painter.setFont(QFont("Arial", 12, QFont.Bold))
        for top, right, bottom, left, label in faces:
            painter.drawRect(left, top, right - left, bottom - top)
            painter.drawText(left, max(top - 6, 14), label)
        painter.end()

        self.video_label.setPixmap(
            QPixmap.fromImage(img).scaled(640, 480, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def update_person_info(self, name, emotion):
        text = f"Личность: {name or '-'}"
        text += f" | Эмоция: {EMOTION_RU.get(emotion, emotion) if emotion else '-'}"
        self.person_info.setText(text)

    def add_to_chat(self, role, text):
        safe = html.escape(text).replace("\n", "<br>")
        if role == "user":
            formatted = f"<b style='color: blue;'>Вы:</b> {safe}"
        else:
            formatted = f"<b style='color: green;'>Ассистент:</b> {safe}"
        self.chat_history.append(formatted)
        self.chat_history.verticalScrollBar().setValue(
            self.chat_history.verticalScrollBar().maximum())

    def update_status(self, message):
        self.status_bar.showMessage(message)

    def send_text(self):
        text = self.input_line.text().strip()
        if text:
            self.audio_thread.submit_text(text)
            self.input_line.clear()

    def clear_chat(self):
        self.chat_history.clear()
        self.audio_thread.history.clear()

    def stop_all(self):
        self.close()

    def closeEvent(self, event):
        self.video_thread.stop()
        self.audio_thread.stop()
        time.sleep(0.2)
        event.accept()


if __name__ == "__main__":
    print("[START] Нейросеть-ассистент загружается...")
    print(f"[INFO] Устройство: {DEVICE}")
    print(f"[INFO] Ollama: {'включена' if USE_OLLAMA else 'отключена'}")
    if USE_OLLAMA:
        print(f"       Модель: {OLLAMA_MODEL}")
        print(f"       Хост: {OLLAMA_HOST}")

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
     

