import os
import time
import argparse
import logging
import shutil
import threading
import queue
import psutil
import concurrent.futures
from pathlib import Path

import torch
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from transformers import AutoModelForImageClassification, AutoFeatureExtractor, AutoImageProcessor, ViTForImageClassification
from nudenet import NudeDetector
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms

class ResourceMonitor:
    def __init__(self):
        self.process = psutil.Process(os.getpid())
    
    def get_memory_usage(self):
        return self.process.memory_info().rss / (1024 * 1024)  # в МБ
    
    def get_cpu_usage(self):
        return self.process.cpu_percent()

class ImageCategorizer:
    def __init__(self, input_dir, output_dir, num_threads=4, log_file=None):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.num_threads = num_threads
        self.resource_monitor = ResourceMonitor()
        
        self.categories = ['cars', 'landscapes', 'interiors', 'documents', 'people', 'adult', 'others']
        for category in self.categories:
            os.makedirs(self.output_dir / category, exist_ok=True)
        
        self.log_queue = queue.Queue()
        self.files_queue = queue.Queue()
        
        self.setup_logging(log_file)
        self.load_models()
        self.lock = threading.Lock()
    
    def setup_logging(self, log_file):
        self.logger = logging.getLogger('ImageCategorizer')
        self.logger.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
    
    def load_models(self):
        self.logger.info("Загрузка моделей...")
        start_time = time.time()
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        
        self.logger.info(f"Используем устройство: {device}")
        
        # Модель для классификации автомобилей (специфичная для автомобилей)
        self.car_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
        self.car_model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224").to(device)
        self.car_model.eval()
        
        # Модель для сцен (пейзажи и интерьеры)
        self.scene_processor = AutoImageProcessor.from_pretrained("microsoft/beit-base-patch16-224-pt22k-ft22k")
        self.scene_model = AutoModelForImageClassification.from_pretrained("microsoft/beit-base-patch16-224-pt22k-ft22k").to(device)
        self.scene_model.eval()
        
        # Модель для документов
        self.document_model = tf.keras.applications.EfficientNetB0(weights='imagenet')
        
        # Модель для людей
        self.people_model = tf.keras.applications.EfficientNetB3(weights='imagenet')
        
        # Модель для контента для взрослых
        self.nude_detector = NudeDetector()
        
        # Общая модель классификации
        self.general_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).to(device)
        self.general_model.eval()
        
        # Трансформации для изображений
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        elapsed_time = time.time() - start_time
        mem_usage = self.resource_monitor.get_memory_usage()
        self.logger.info(f"Модели загружены за {elapsed_time:.2f} секунд. Использовано памяти: {mem_usage:.2f} МБ")
    
    def is_car(self, image):
        try:
            img = Image.open(image).convert('RGB')
            inputs = self.car_processor(images=img, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.car_model(**inputs)
            
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)[0]
            
            # Индексы автомобилей в ImageNet
            car_indices = [817, 511, 468, 751, 661, 817, 864, 867, 407, 436, 656, 627, 717, 734, 864, 581, 555, 569, 654, 675]
            car_prob = sum(probabilities[i].item() for i in car_indices if i < len(probabilities))
            
            # Также проверяем с общей моделью
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                output = self.general_model(img_tensor)
            
            general_probs = torch.nn.functional.softmax(output, dim=1)[0]
            general_car_prob = sum(general_probs[i].item() for i in car_indices if i < len(general_probs))
            
            final_car_prob = max(car_prob, general_car_prob)
            return final_car_prob > 0.15, final_car_prob
        except Exception as e:
            self.logger.error(f"Ошибка при проверке изображения на автомобиль: {e}")
            return False, 0
    
    def is_landscape(self, image):
        try:
            img = Image.open(image).convert('RGB')
            inputs = self.scene_processor(images=img, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.scene_model(**inputs)
            
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)[0]
            
            # Индексы пейзажей
            landscape_indices = [927, 979, 975, 976, 978, 970, 980, 981, 983, 974, 980, 976]
            
            # Индексы, которые точно не пейзажи
            not_landscape_indices = list(range(0, 400)) + list(range(600, 900))
            
            landscape_prob = sum(probabilities[i].item() for i in landscape_indices if i < len(probabilities))
            not_landscape_prob = sum(probabilities[i].item() for i in not_landscape_indices if i < len(probabilities))
            
            # Анализ яркости и цветовой гаммы для определения пейзажа
            np_img = np.array(img)
            hsv = cv2.cvtColor(np_img, cv2.COLOR_RGB2HSV)
            
            # Пейзажи обычно имеют много голубого (небо) и зеленого (трава)
            blue_green_mask = cv2.inRange(hsv, (90, 50, 50), (150, 255, 255))
            blue_green_ratio = np.count_nonzero(blue_green_mask) / (np_img.shape[0] * np_img.shape[1])
            
            # Усиливаем вероятность, если много зеленого/голубого
            landscape_prob = landscape_prob * (1 + blue_green_ratio)
            
            return landscape_prob > 0.2 and landscape_prob > not_landscape_prob, landscape_prob
        except Exception as e:
            self.logger.error(f"Ошибка при проверке изображения на пейзаж: {e}")
            return False, 0
    
    def is_interior(self, image):
        try:
            img = Image.open(image).convert('RGB')
            inputs = self.scene_processor(images=img, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.scene_model(**inputs)
            
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)[0]
            
            # Индексы интерьеров и связанных объектов
            interior_indices = [522, 527, 565, 564, 525, 529, 532, 533, 534, 573, 575, 765, 788, 789, 791, 793]
            interior_prob = sum(probabilities[i].item() for i in interior_indices if i < len(probabilities))
            
            # Дополнительная проверка на признаки интерьера
            np_img = np.array(img)
            
            # Определяем прямые линии (характерные для интерьеров)
            gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            
            # Используем вероятностный алгоритм Хафа для поиска линий
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=50, maxLineGap=10)
            
            has_straight_lines = False
            if lines is not None and len(lines) > 10:
                has_straight_lines = True
            
            # Усиливаем вероятность при наличии прямых линий
            if has_straight_lines:
                interior_prob *= 1.5
                
            return interior_prob > 0.15, interior_prob
        except Exception as e:
            self.logger.error(f"Ошибка при проверке изображения на интерьер: {e}")
            return False, 0
    
    def is_document(self, image):
        try:
            img = cv2.imread(str(image))
            if img is None:
                return False, 0
                
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            processed_img = cv2.resize(rgb_img, (224, 224))
            processed_img = tf.keras.applications.efficientnet.preprocess_input(processed_img[np.newaxis, ...])
            
            predictions = self.document_model.predict(processed_img, verbose=0)
            
            # Индексы документов и связанных объектов
            document_indices = [402, 497, 508, 721, 722]
            document_prob = sum(predictions[0][i] for i in document_indices if i < len(predictions[0]))
            
            # Дополнительная проверка на документ через анализ линий и текста
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Проверка на горизонтальные линии
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
            
            has_horizontal_lines = False
            if lines is not None:
                horizontal_count = 0
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    if abs(y2 - y1) < 10:  # горизонтальная линия
                        horizontal_count += 1
                has_horizontal_lines = horizontal_count > 3
            
            # Проверка на однородные области (часто бывают в документах)
            ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            has_rectangular_areas = False
            if contours:
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = float(w)/h
                    # Проверка на прямоугольную форму с определенным соотношением сторон
                    if 1.0 <= aspect_ratio <= 4.0 and w*h > (img.shape[0]*img.shape[1])/20:
                        has_rectangular_areas = True
                        break
            
            # Определение белого фона
            has_white_background = np.mean(gray) > 180
            
            document_score = document_prob
            if has_horizontal_lines:
                document_score *= 1.5
            if has_rectangular_areas:
                document_score *= 1.3
            if has_white_background:
                document_score *= 1.2
                
            return document_score > 0.15, document_score
        except Exception as e:
            self.logger.error(f"Ошибка при проверке изображения на документ: {e}")
            return False, 0
    
    def is_person(self, image):
        try:
            img = cv2.imread(str(image))
            if img is None:
                return False, 0
                
            # Используем EfficientNet для определения людей
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            processed_img = cv2.resize(rgb_img, (224, 224))
            processed_img = tf.keras.applications.efficientnet.preprocess_input(processed_img[np.newaxis, ...])
            
            predictions = self.people_model.predict(processed_img, verbose=0)
            
            # Индексы людей и частей тела
            person_indices = list(range(394, 410)) + [804, 850]
            person_prob = sum(predictions[0][i] for i in person_indices if i < len(predictions[0]))
            
            # Проверка с помощью каскадов Хаара для обнаружения лиц
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            has_faces = len(faces) > 0
            
            # Проверка с помощью NudeNet для обнаружения людей
            nude_result = self.nude_detector.detect(str(image))
            has_person_parts = any(class_name in ['FACE_F', 'FACE_M', 'MALE_BREAST', 'FEMALE_BREAST', 
                                                  'MALE_GENITALIA', 'FEMALE_GENITALIA', 'BUTTOCKS', 
                                                  'FEET', 'BELLY', 'ARMPITS'] 
                                  for detection in nude_result for class_name in [detection['class']])
            
            person_score = person_prob
            if has_faces:
                person_score = max(person_score, 0.5)
            if has_person_parts:
                person_score = max(person_score, 0.7)
                
            return person_score > 0.15, person_score
        except Exception as e:
            self.logger.error(f"Ошибка при проверке изображения на человека: {e}")
            return False, 0
    
    def is_adult(self, image):
        try:
            # Используем NudeDetector для определения контента для взрослых
            nude_detection = self.nude_detector.detect(str(image))
            
            # Смотрим наличие обнаженных частей тела
            has_explicit_parts = any(class_name in ['MALE_GENITALIA', 'FEMALE_GENITALIA', 'FEMALE_BREAST_EXPOSED', 'BUTTOCKS_EXPOSED']
                                 for detection in nude_detection 
                                 for class_name in [detection['class']])
            
            # Определяем количество и вероятность обнаженных частей тела
            explicit_scores = [detection['score'] 
                           for detection in nude_detection 
                           if detection['class'] in ['MALE_GENITALIA', 'FEMALE_GENITALIA', 
                                                     'FEMALE_BREAST_EXPOSED', 'BUTTOCKS_EXPOSED']]
            
            # Также проверяем общие части тела для оценки общей ситуации
            body_scores = [detection['score'] 
                       for detection in nude_detection]
                       
            # Общее количество обнаруженных частей тела                
            body_parts_count = len(body_scores)
            
            explicit_prob = sum(explicit_scores) if explicit_scores else 0
            
            # NudeNet возвращает в detection только найденные части,
            # поэтому проверяем наличие нескольких частей тела
            is_likely_nude = len(nude_detection) >= 3 and explicit_prob > 0.3
            
            # Если есть явные части и общий показатель выше порога
            final_adult_prob = explicit_prob
            if has_explicit_parts:
                final_adult_prob = max(final_adult_prob, 0.6)
            if is_likely_nude:
                final_adult_prob = max(final_adult_prob, 0.7)
                
            # Проверка на особые случаи
            # Если очень много деталей тела обнаружено
            if body_parts_count > 5 and any(s > 0.7 for s in body_scores):
                final_adult_prob = max(final_adult_prob, 0.5)
            
            return final_adult_prob > 0.3, final_adult_prob
        except Exception as e:
            self.logger.error(f"Ошибка при проверке изображения на эротику: {e}")
            return False, 0
    
    def classify_image(self, image_path):
        try:
            start_time = time.time()
            self.logger.info(f"Обработка файла: {image_path}")
            
            # Проводим все проверки
            is_car, car_prob = self.is_car(image_path)
            is_landscape, landscape_prob = self.is_landscape(image_path)
            is_interior, interior_prob = self.is_interior(image_path)
            is_document, document_prob = self.is_document(image_path)
            is_person, person_prob = self.is_person(image_path)
            is_adult, adult_prob = self.is_adult(image_path)
            
            # Определяем категорию с наибольшей вероятностью
            categories = {
                'cars': car_prob if is_car else 0,
                'landscapes': landscape_prob if is_landscape else 0,
                'interiors': interior_prob if is_interior else 0,
                'documents': document_prob if is_document else 0,
                'people': person_prob if is_person else 0,
                'adult': adult_prob if is_adult else 0
            }
            
            # Особый случай: если человек и эротика, то приоритет эротике
            if is_person and is_adult:
                if adult_prob > 0.35:
                    categories['people'] = 0
            
            # Выбираем категорию с наибольшей вероятностью
            max_category = max(categories.items(), key=lambda x: x[1])
            
            if max_category[1] > 0:
                category = max_category[0]
            else:
                category = 'others'
            
            with self.lock:
                dest_path = self.output_dir / category / image_path.name
                shutil.copy2(image_path, dest_path)
            
            elapsed_time = time.time() - start_time
            mem_usage = self.resource_monitor.get_memory_usage()
            cpu_usage = self.resource_monitor.get_cpu_usage()
            
            probs_str = ", ".join([f"{cat}: {prob:.2f}" for cat, prob in categories.items() if prob > 0])
            
            log_message = f"Файл {image_path.name} классифицирован как {category}. "
            log_message += f"Вероятности: {probs_str}. "
            log_message += f"Время: {elapsed_time:.2f}с, Память: {mem_usage:.2f}МБ, CPU: {cpu_usage:.2f}%"
            
            return log_message
            
        except Exception as e:
            return f"Ошибка при обработке {image_path}: {e}"
    
    def process_file(self):
        while True:
            try:
                file_path = self.files_queue.get(block=False)
                log_message = self.classify_image(file_path)
                self.log_queue.put((file_path, log_message))
            except queue.Empty:
                break
            except Exception as e:
                self.log_queue.put((file_path, f"Ошибка при обработке {file_path}: {e}"))
            finally:
                self.files_queue.task_done()
    
    def log_worker(self):
        while not self.log_queue.empty():
            try:
                _, message = self.log_queue.get(block=False)
                self.logger.info(message)
                self.log_queue.task_done()
            except queue.Empty:
                break
            except Exception as e:
                self.logger.error(f"Ошибка в логгере: {e}")
                self.log_queue.task_done()
    
    def run(self):
        self.logger.info(f"Начинаем обработку изображений из {self.input_dir}")
        start_time = time.time()
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
        for file in self.input_dir.iterdir():
            if file.is_file() and file.suffix.lower() in image_extensions:
                self.files_queue.put(file)
        
        total_files = self.files_queue.qsize()
        self.logger.info(f"Найдено {total_files} файлов для обработки")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            for _ in range(self.num_threads):
                executor.submit(self.process_file)
        
        while not self.files_queue.empty() or not self.log_queue.empty():
            self.log_worker()
            time.sleep(0.1)
        
        total_time = time.time() - start_time
        self.logger.info(f"Обработка завершена. Всего обработано {total_files} файлов за {total_time:.2f} секунд.")
        self.logger.info(f"Среднее время на файл: {total_time/total_files:.2f} секунд.")

def main():
    parser = argparse.ArgumentParser(description='Классификатор изображений по категориям')
    parser.add_argument('--input', '-i', type=str, required=True, help='Папка с исходными изображениями')
    parser.add_argument('--output', '-o', type=str, required=True, help='Папка для сохранения классифицированных изображений')
    parser.add_argument('--threads', '-t', type=int, default=4, help='Количество потоков')
    parser.add_argument('--log', '-l', type=str, default=None, help='Файл для логов')
    
    args = parser.parse_args()
    
    categorizer = ImageCategorizer(
        input_dir=args.input,
        output_dir=args.output,
        num_threads=args.threads,
        log_file=args.log
    )
    
    categorizer.run()

if __name__ == "__main__":
    main()