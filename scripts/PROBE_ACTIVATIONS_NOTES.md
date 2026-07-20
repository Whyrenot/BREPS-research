# probe_decoder_activations.py — заметки

Скрипт измеряет, **в каком слое SAM** крошечный сдвиг box-промпта (critical shift)
превращается в катастрофическое расхождение маски. Внутри пары изображение одно и
то же → image embedding идентичен → всё расхождение рождается в `prompt_encoder` и
усиливается слоями `mask_decoder`.

## 1. Имена слоёв (совпадают с кодом авторов)

Эталон — официальный репозиторий Meta AI
[facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything)
(Kirillov et al., *Segment Anything*, ICCV 2023), файлы
`segment_anything/modeling/{prompt_encoder,mask_decoder,transformer}.py`.

Листовые модули именуются через `named_modules()`, т.е. **дословно** авторскими
именами атрибутов: `mask_decoder.transformer.layers.0.self_attn.q_proj`,
`cross_attn_token_to_image`, `output_upscaling.3`, `iou_prediction_head.layers.2`
и т.д.

Два хука стоят на корневых модулях, возвращающих кортеж; элементы кортежа
подписаны **именами переменных из `return` авторского `forward()`**:

| Ключ на графике | Код авторов | Что это |
|---|---|---|
| `prompt_encoder[sparse_embeddings]` | `PromptEncoder.forward → return sparse_embeddings, dense_embeddings` | эмбеддинги углов бокса, **зависят от бокса** |
| `prompt_encoder[dense_embeddings]`  | там же | `no_mask_embed`, **не зависит от бокса** (sanity-check: расхождение ≡ 0) |
| `mask_decoder[masks]`    | `MaskDecoder.forward → return masks, iou_pred` | логиты маски (финальный выход) |
| `mask_decoder[iou_pred]` | там же | предсказанный IoU-скор (финальный выход) |

Старые обозначения `out0`/`out1`, `prompt enc`, `[sparse]`/`[dense]` больше не
используются. Маппинг задан в `_semantic_out_names()`.

## 2. Только используемые слои, строго в порядке forward

Хукается ровно то, чей выход **реально используется** при одиночном box-предикте
(`multimask_output=False`); порядок точек на оси = порядок срабатывания хуков =
порядок исполнения forward.

Исключены как неиспользуемые/дубликаты (список `_UNUSED_RE` + фильтр контейнеров
в `register_hooks()`):

- **контейнерные модули** (`self_attn`, `mlp`, `output_upscaling` как Sequential,
  `transformer`, блоки `layers.N`, …) — их выход это тот же тензор, что у
  последнего ребёнка (например, `self_attn` ≡ `self_attn.out_proj`), чистые
  дубликаты на графике;
- **`prompt_encoder.pe_layer`** — его `forward` срабатывает только из
  `get_dense_pe()` (позиционное кодирование картинки, аргумент `image_pe`
  декодера); от бокса не зависит, расхождение ≡ 0. Для боксов авторы зовут
  `pe_layer.forward_with_coords()`, что хук не видит;
- **`output_hypernetworks_mlps.1–3`** — при `multimask_output=False` авторский
  `MaskDecoder.forward` берёт `mask_slice = slice(0, 1)`: эти MLP отрабатывают,
  но их выход отбрасывается. Остаётся только `output_hypernetworks_mlps.0`.

Дети `prompt_encoder` (`point_embeddings`, `no_mask_embed`, `not_a_point_embed`,
`mask_downscaling`) в box-пути не вызывают `forward` (доступ через `.weight`),
поэтому в захват не попадают автоматически; промпт-уровень виден через корневой
выход `prompt_encoder[sparse_embeddings/dense_embeddings]`.

Ожидаемые нули **используемых** слоёв (не убирать, уметь объяснить): в слое 0
`cross_attn_token_to_image.k_proj/v_proj` и `cross_attn_image_to_token.q_proj`
равны 0, т.к. ветка `keys` (image embedding + dense) до первой image→token
attention от бокса не зависит. Ловушка интерпретации: в
`cross_attn_image_to_token` авторы вызывают attention со **свопнутыми**
аргументами (`q=k, k=q`), так что `q_proj` этого блока обрабатывает image-сторону,
а `k_proj` — токены.

## 3. Метрики

**Расхождение пары** (на разности активаций `A_best − A_other`), на слой:
- `raw_l2`  = ‖Δ‖₂
- `rel_l2`  = ‖Δ‖₂ / ‖A_best‖₂           (относительная L2, сравнима между слоями)
- `rms_l2`  = ‖Δ‖₂ / √numel              (RMS на элемент)
- `l1_rel`  = ‖Δ‖₁ / ‖A_best‖₁           (относительная L1)

Типы пар: `critical` = best vs bad, `control` = best vs контрольный бокс (сдвиг той
же величины, что best→bad, но в случайном направлении; включается флагом `--control`).

**Std активаций** (разброс самих значений `A`, не разности), на слой, по сценариям:
`normal` = best_box, `critical` = bad_box, `control` = контрольный бокс.

## 4. Выходные файлы

Std больше **не отдельный график**: на графиках L2 и L1 каждая кривая — это
среднее по парам, а вокруг неё закрашенный «бегущий» диапазон mean ± std
(std того же расхождения по парам, на каждый слой; нижняя граница обрезается
нулём, т.к. расхождение неотрицательно).

| Аргумент | Содержимое |
|----------|-----------|
| `--out_csv`      | таблица расхождений: строка на `(layer, pair_type)`, колонки raw/rel/rms_l2 + l1_rel (mean/std) |
| `--out_plot`     | raw L2 + relative L2, каждая линия с полосой mean ± std |
| `--out_l1_plot`  | relative L1 с полосой mean ± std |
| `--out_std_csv`  | широкая таблица std самих активаций по сценариям: строка на слой, `std_{normal,critical,control}_mean`, `_cases_std`, отношения `std_*_over_normal` |
| `--per_case_csv` | (опц.) per-(pair, layer) raw/rel/rms_l2 + l1_rel |

Флаги `--leaf_only` и `--out_std_plot` удалены: leaf-режим с исключением
неиспользуемых слоёв теперь единственное поведение, std встроен в основные
графики полосой.

## 5. Запуск на сервере

```bash
CUDA_VISIBLE_DEVICES=3 python scripts/probe_decoder_activations.py \
    --critical_shifts critical_shifts.json \
    --images_dir /path/to/FOR_TEST/images \
    --checkpoint_path /path/to/sam_vit_b_01ec64.pth \
    --model_name SAM --model_type vit_b \
    --limit 100 --control \
    --out_csv      outputs_smoothed/decoder_activation_l2.csv \
    --out_plot     visualizations/decoder_activation_l2.png \
    --out_l1_plot  visualizations/decoder_activation_l1.png \
    --out_std_csv  outputs_smoothed/decoder_activation_std.csv
```

`--control` нужен, чтобы появились линия control на панелях расхождений и
сценарий control на панели std; без него считаются только `normal` + `critical`.

Важно: старые результаты (`results/decoder_activation_*.csv/png`) сгенерированы
прошлыми версиями скрипта (там ещё есть `pe_layer`, дубликаты-контейнеры,
`[sparse]/[dense]`, неиспользуемые гипер-MLP) — после запуска новой версии их
нужно заменить.
