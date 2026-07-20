# BREPS: Руководство по установке (Исправленное)

Оригинальный `README.md` упускает из виду некоторые важные нюансы, связанные со сборкой и отсутствующими зависимостями (в особенности, пакетом `pydiffvg`, необходимым для дифференцируемого рендеринга и работы с SAM-моделями на этапе оптимизации bounding box).

Ниже описаны все шаги, которые мы проделали для того, чтобы окружение работало полностью и без ошибок `ModuleNotFoundError`.

## 1. Базовая установка

Для начала установите PyTorch и основные системные зависимости (через Conda/Pip).
> **Важно:** верните версию PyTorch 1.13.1 (как советуют авторы), чтобы избежать конфликтов (например, ошибки "CUDA driver is too old" при установке новейшей версии PyTorch).

```bash
# 1.1 Устанавливаем PyTorch
pip install torch==1.13.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

# 1.2 Ставим cmake и scikit-image
conda install -y scikit-image
conda install -y -c anaconda cmake
```

## 2. Установка `pydiffvg`

Для расчета градиентов по BBox при помощи SAM-моделей необходима библиотека дифференцируемого векторного рендеринга `pydiffvg`. Она не ставится по умолчанию из pip-репозитория и может вызывать ошибки при стандартном pip-клонировании, поэтому ставим из исходников:

```bash
# Клонируем в любое временное место (здесь для примера в саму папку BREPS/diffvg)
git clone https://github.com/BachiLi/diffvg.git
cd diffvg
git submodule update --init --recursive
python setup.py install
cd ..

# Pydiffvg требует две зависимости, которые не всегда подтягиваются автоматически
pip install svgpathtools cssutils
```

## 3. Исправление `pyproject.toml`

При выполнении команды `pip install -e .` происходит ошибка сборки _"Multiple top-level packages discovered"_. `setuptools` пугается множества папок (assets, scripts, heatmaps) и отказывается собирать решение. 
К тому же оригинальный `.toml` не включает подпапки датасетов.

Для исправления этого в самый конец файла `pyproject.toml` мы добавили:
```toml
[tool.setuptools.packages.find]
where = ["."]
include = ["isegm*"]
```

## 4. Очистка кода от "приватных" модулей

Некоторые датасеты и скрипты, на которые есть ссылки в коде (в частности `UserStudyNewDataset`, `InteractionDataset`, `WBCDataset`, `LvisDataset` и модуль `ScribblePredictor`), попросту отсутствуют в публичном релизе репозитория. При импорте они вызывают `ModuleNotFoundError`.

Чтобы скрипты не ломались сразу при проверке импортов, мы закомментировали их:

1. В `isegm/data/datasets/__init__.py`: убраны ссылки на `.userstudy_new`, `.interactionset`, `.wbc`, `.lvis`.
2. В `isegm/inference/utils.py`: убраны ссылки на аналогичные классы в блоке импортов.
3. В `isegm/inference/predictors/__init__.py`: закомментирована строка `from .scribble_predictor import ScribblePredictor`.

## 5. Завершение установки пакета

И наконец, доустанавливаем недостающие библиотеки логирования и вычислений, запускаемые пакетом, и устанавливаем саму библиотеку `isegm` локально:

```bash
pip install loguru numba
pip install -e .
```

Проверить, что всё работает, можно запустив:
```bash
python3 scripts/evaluate_boxes_model_sam.py --help
```
*(оно должно выдать подсказку по опциям скрипта с `Exit code: 0`)*
