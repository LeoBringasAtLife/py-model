from pathlib import Path
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neural_network import MLPClassifier

IMG_SIZE = 28
DATA_DIR = Path('data')
TRAIN_DIR = DATA_DIR / 'train'
TEST_DIR = DATA_DIR / 'test'


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


def main() -> None:
    print('Generating images...')
    generate_dataset(train_per_digit=120, test_per_digit=30)

    print('Loading images...')
    x_train, y_train = load_images(TRAIN_DIR)
    x_test, y_test = load_images(TEST_DIR)

    print(f'Train: {len(y_train)} images | Test: {len(y_test)} images')

    model = MLPClassifier(
        hidden_layer_sizes=(128,),
        activation='relu',
        solver='adam',
        max_iter=400,
        random_state=42,
    )

    print('Training model...')
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    y_proba = model.predict_proba(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    loss = log_loss(y_test, y_proba)

    print(f'Loss: {loss:.4f}, Accuracy: {accuracy:.4f}')


if __name__ == '__main__':
    main()
