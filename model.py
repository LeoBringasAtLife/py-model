from io import BytesIO
from pathlib import Path
import random

import joblib
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neural_network import MLPClassifier

IMG_SIZE = 28
DATA_DIR = Path('data')
TRAIN_DIR = DATA_DIR / 'train'
TEST_DIR = DATA_DIR / 'test'
MODEL_PATH = Path('digit_model.joblib')


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    font_candidates = [
        'C:/Windows/Fonts/arial.ttf',
        'C:/Windows/Fonts/calibri.ttf',
        'C:/Windows/Fonts/tahoma.ttf',
    ]

    for candidate in font_candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def generate_digit_image(digit: int, rng: random.Random) -> Image.Image:
    image = Image.new('L', (IMG_SIZE, IMG_SIZE), color=0)
    draw = ImageDraw.Draw(image)
    font = _load_font(rng.randint(16, 24))

    text = str(digit)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    x = (IMG_SIZE - text_w) // 2 + rng.randint(-3, 3)
    y = (IMG_SIZE - text_h) // 2 + rng.randint(-3, 3)
    draw.text((x, y), text, fill=255, font=font)

    angle = rng.uniform(-20, 20)
    image = image.rotate(angle, fillcolor=0)

    if rng.random() < 0.35:
        image = image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.3, 1.0)))

    arr = np.array(image, dtype=np.float32)
    noise = np.random.normal(loc=0.0, scale=rng.uniform(4.0, 14.0), size=arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)

    return Image.fromarray(arr, mode='L')


def generate_dataset(train_per_digit: int = 120, test_per_digit: int = 30) -> None:
    for split_dir, count in ((TRAIN_DIR, train_per_digit), (TEST_DIR, test_per_digit)):
        for digit in range(10):
            digit_dir = split_dir / str(digit)
            digit_dir.mkdir(parents=True, exist_ok=True)

            for old_file in digit_dir.glob('*.png'):
                old_file.unlink()

            rng = random.Random(42 + digit + count)
            for idx in range(count):
                image = generate_digit_image(digit, rng)
                image.save(digit_dir / f'{digit}_{idx:04d}.png')


def load_images(split_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    x_list: list[np.ndarray] = []
    y_list: list[int] = []

    for digit_dir in sorted(split_dir.iterdir(), key=lambda p: p.name):
        if not digit_dir.is_dir() or not digit_dir.name.isdigit():
            continue

        label = int(digit_dir.name)
        for image_path in sorted(digit_dir.glob('*.png')):
            image = Image.open(image_path).convert('L').resize((IMG_SIZE, IMG_SIZE))
            arr = np.array(image, dtype=np.float32) / 255.0
            x_list.append(arr.flatten())
            y_list.append(label)

    if not x_list:
        raise RuntimeError(f'No images were found in: {split_dir}')

    x = np.array(x_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    return x, y


def train_model(
    train_per_digit: int = 120,
    test_per_digit: int = 30,
    max_iter: int = 400,
) -> tuple[MLPClassifier, dict[str, float]]:
    print('Generating images...')
    generate_dataset(train_per_digit=train_per_digit, test_per_digit=test_per_digit)

    print('Loading images...')
    x_train, y_train = load_images(TRAIN_DIR)
    x_test, y_test = load_images(TEST_DIR)

    print(f'Train: {len(y_train)} images | Test: {len(y_test)} images')

    model = MLPClassifier(
        hidden_layer_sizes=(128,),
        activation='relu',
        solver='adam',
        max_iter=max_iter,
        random_state=42,
    )

    print('Training model...')
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    y_proba = model.predict_proba(x_test)

    metrics = {
        'loss': float(log_loss(y_test, y_proba)),
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'train_images': float(len(y_train)),
        'test_images': float(len(y_test)),
    }
    return model, metrics


def save_model(model: MLPClassifier, model_path: Path = MODEL_PATH) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)


def load_saved_model(model_path: Path = MODEL_PATH) -> MLPClassifier | None:
    if model_path.exists():
        return joblib.load(model_path)
    return None


def preprocess_uploaded_image(file_bytes: bytes) -> np.ndarray:
    image = Image.open(BytesIO(file_bytes)).convert('L').resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(image, dtype=np.float32) / 255.0

    # Auto-invert if image background is white.
    if float(arr.mean()) > 0.5:
        arr = 1.0 - arr

    return arr.flatten().reshape(1, -1)


def main() -> None:
    model, metrics = train_model(train_per_digit=120, test_per_digit=30, max_iter=400)
    save_model(model)
    print(f"Loss: {metrics['loss']:.4f}, Accuracy: {metrics['accuracy']:.4f}")
    print(f'Saved model: {MODEL_PATH}')


if __name__ == '__main__':
    main()
