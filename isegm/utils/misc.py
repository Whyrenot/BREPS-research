import torch
import numpy as np
import pickle
from isegm.inference import utils
from isegm.utils.vis import draw_probmap, draw_with_blend_and_boxes, draw_with_blend_and_clicks
from pathlib import Path
from loguru import logger

def get_dims_with_exclusion(dim, exclude=None):
    dims = list(range(dim))
    if exclude is not None:
        dims.remove(exclude)

    return dims



def save_checkpoint(net, checkpoints_path, epoch=None, prefix='', verbose=True, multi_gpu=False):
    if epoch is None:
        checkpoint_name = 'last_checkpoint.pth'
    else:
        checkpoint_name = f'{epoch:03d}.pth'

    if prefix:
        checkpoint_name = f'{prefix}_{checkpoint_name}'

    if not checkpoints_path.exists():
        checkpoints_path.mkdir(parents=True)

    checkpoint_path = checkpoints_path / checkpoint_name
    if verbose:
        logger.info(f'Save checkpoint to {str(checkpoint_path)}')

    net = net.module if multi_gpu else net
    torch.save({'state_dict': net.state_dict(),
                'config': net._config}, str(checkpoint_path))


def get_bbox_from_mask(mask):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax


def expand_bbox(bbox, expand_ratio, min_crop_size=None):
    rmin, rmax, cmin, cmax = bbox
    rcenter = 0.5 * (rmin + rmax)
    ccenter = 0.5 * (cmin + cmax)
    height = expand_ratio * (rmax - rmin + 1)
    width = expand_ratio * (cmax - cmin + 1)
    if min_crop_size is not None:
        height = max(height, min_crop_size)
        width = max(width, min_crop_size)

    rmin = int(round(rcenter - 0.5 * height))
    rmax = int(round(rcenter + 0.5 * height))
    cmin = int(round(ccenter - 0.5 * width))
    cmax = int(round(ccenter + 0.5 * width))

    return rmin, rmax, cmin, cmax


def clamp_bbox(bbox, rmin, rmax, cmin, cmax):
    return (max(rmin, bbox[0]), min(rmax, bbox[1]),
            max(cmin, bbox[2]), min(cmax, bbox[3]))


def get_bbox_iou(b1, b2):
    h_iou = get_segments_iou(b1[:2], b2[:2])
    w_iou = get_segments_iou(b1[2:4], b2[2:4])
    return h_iou * w_iou


def get_segments_iou(s1, s2):
    a, b = s1
    c, d = s2
    intersection = max(0, min(b, d) - max(a, c) + 1)
    union = max(1e-6, max(b, d) - min(a, c) + 1)
    return intersection / union


def get_labels_with_sizes(x):
    obj_sizes = np.bincount(x.flatten())
    labels = np.nonzero(obj_sizes)[0].tolist()
    labels = [x for x in labels if x != 0]
    return labels, obj_sizes[labels].tolist()

def get_predictor_and_zoomin_params(args, dataset_name):
    predictor_params = {}

    if args.clicks_limit is not None:
        if args.clicks_limit == -1:
            args.clicks_limit = args.n_clicks
        predictor_params['net_clicks_limit'] = args.clicks_limit

    if args.eval_mode == 'cvpr':
        zoom_in_params = {
            # We force resize on first interaction since base model not able to handle high resolution TETRIS images
            'skip_clicks': -1 if dataset_name == 'TETRIS' else 0,
            'target_size': 600 if dataset_name == 'DAVIS' else 400
        }
    elif args.eval_mode.startswith('fixed'):
        crop_size = int(args.eval_mode[5:])
        zoom_in_params = {
            'skip_clicks': -1,
            'target_size': (crop_size, crop_size)
        }
    else:
        raise NotImplementedError

    return predictor_params, zoom_in_params


def get_checkpoints_list_and_logs_path(args, cfg):
    logs_prefix = ''
    if args.exp_path:
        rel_exp_path = args.exp_path
        checkpoint_prefix = ''
        if ':' in rel_exp_path:
            rel_exp_path, checkpoint_prefix = rel_exp_path.split(':')

        exp_path_prefix = cfg.EXPS_PATH / rel_exp_path
        candidates = list(exp_path_prefix.parent.glob(exp_path_prefix.stem + '*'))
        assert len(candidates) == 1, "Invalid experiment path."
        exp_path = candidates[0]
        checkpoints_list = sorted((exp_path / 'checkpoints').glob(checkpoint_prefix + '*.pt*'), reverse=True)
        assert len(checkpoints_list) > 0, "Couldn't find any checkpoints."

        if checkpoint_prefix:
            if len(checkpoints_list) == 1:
                logs_prefix = checkpoints_list[0].stem
            else:
                logs_prefix = f'all_{checkpoint_prefix}'
        else:
            logs_prefix = 'all_checkpoints'

        logs_path = args.logs_path / exp_path.relative_to(cfg.EXPS_PATH)
    else:
        checkpoints_list = [Path(utils.find_checkpoint(cfg.INTERACTIVE_MODELS_PATH, args.checkpoint))]
        logs_path = args.logs_path / 'others' / checkpoints_list[0].stem

    return checkpoints_list, Path(str(logs_path) + "_" + ("MIN" if args.optim_min else "MAX")), logs_prefix


def save_results(args, row_name, dataset_name, logs_path, logs_prefix, dataset_results,
                 save_ious=False, print_header=True, single_model_eval=False):
    all_ious, elapsed_time = dataset_results
    ious_path = logs_path / 'ious' / (logs_prefix if logs_prefix else '')
    # with open(ious_path / f'{dataset_name}_fullious.pkl', 'wb') as fp:
    #     pickle.dump({"all_ious":all_ious}, fp)

    print(np.array(all_ious, dtype=object).shape)
    all_ious = np.array(all_ious, dtype=object)[:, 1]

    mean_spc, mean_spi = utils.get_time_metrics(all_ious, elapsed_time)

    iou_thrs = np.arange(0.8, min(0.95, args.target_iou) + 0.001, 0.05).tolist()
    noc_list, over_max_list = utils.compute_noc_metric(all_ious, iou_thrs=iou_thrs, max_clicks=args.n_clicks)

    row_name = 'last' if row_name == 'last_checkpoint' else row_name
    model_name = str(logs_path.relative_to(args.logs_path)) + ':' + logs_prefix if logs_prefix else logs_path.stem
    header, table_row = utils.get_results_table(noc_list, over_max_list, row_name, dataset_name,
                                                mean_spc, elapsed_time, args.n_clicks,
                                                model_name=model_name)

    if args.print_ious:
        min_num_clicks = min(len(x) for x in all_ious)
        mean_ious = np.array([np.array(x[:min_num_clicks]) for x in all_ious]).mean(axis=0)[:, 0]
        miou_str = ' '.join([f'mIoU@{click_id}={mean_ious[click_id - 1]:.2%};'
                             for click_id in list(range(1, 100)) if click_id <= min_num_clicks])
        table_row += '; ' + miou_str
    else:
        target_iou_int = int(args.target_iou * 100)
        if target_iou_int not in [80, 85, 90]:
            noc_list, over_max_list = utils.compute_noc_metric(all_ious, iou_thrs=[args.target_iou],
                                                               max_clicks=args.n_clicks)
            table_row += f' NoC@{args.target_iou:.1%} = {noc_list[0]:.2f};'
            table_row += f' >={args.n_clicks}@{args.target_iou:.1%} = {over_max_list[0]}'

    if print_header:
        print(header)
    print(table_row)

    if save_ious:
        ious_path = logs_path / 'ious' / (logs_prefix if logs_prefix else '')
        ious_path.mkdir(parents=True, exist_ok=True)
        with open(ious_path / f'{dataset_name}_{args.eval_mode}_{args.mode}_{args.n_clicks}.pkl', 'wb') as fp:
            pickle.dump(all_ious, fp)

    name_prefix = ''
    if logs_prefix:
        name_prefix = logs_prefix + '_'
        if not single_model_eval:
            name_prefix += f'{dataset_name}_'

    log_path = logs_path / f'{name_prefix}{args.eval_mode}_{args.mode}_{args.n_clicks}.txt'
    if log_path.exists():
        with open(log_path, 'a') as f:
            f.write(table_row + '\n')
    else:
        with open(log_path, 'w') as f:
            if print_header:
                f.write(header + '\n')
            f.write(table_row + '\n')


def save_iou_analysis_data(args, dataset_name, logs_path, logs_prefix, dataset_results, model_name=None):
    all_ious, _ = dataset_results

    name_prefix = ''
    if logs_prefix:
        name_prefix = logs_prefix + '_'
    name_prefix += dataset_name + '_'
    if model_name is None:
        model_name = str(logs_path.relative_to(args.logs_path)) + ':' + logs_prefix if logs_prefix else logs_path.stem

    pkl_path = logs_path / f'plots/{name_prefix}{args.eval_mode}_{args.mode}_{args.n_clicks}.pickle'
    pkl_path.parent.mkdir(parents=True, exist_ok=True)
    with pkl_path.open('wb') as f:
        pickle.dump({
            'dataset_name': dataset_name,
            'model_name': f'{model_name}_{args.mode}',
            'all_ious': all_ious,

        }, f)


def get_prediction_vis_callback(logs_path, dataset_name, prob_thresh, modality='click'):
    import cv2
    save_path = logs_path / 'predictions_vis' / dataset_name
    save_path.mkdir(parents=True, exist_ok=True)
    match modality:
        case "click":
            draw_func = draw_with_blend_and_clicks
        case "bbox":
            draw_func = draw_with_blend_and_boxes
        case _:
            raise NotImplementedError(f"Modality {modality} is not supported")

    def callback(image, gt_mask, pred_probs, sample_id, click_indx, clicks_list):
        sample_path = save_path / f'{sample_id}_{click_indx}.jpg'
        prob_map = draw_probmap(pred_probs)
        image_with_mask = draw_func(image, pred_probs > prob_thresh, prompt_list=clicks_list)
        cv2.imwrite(str(sample_path), np.concatenate((image_with_mask, prob_map), axis=1)[:, :, ::-1])

    return callback
