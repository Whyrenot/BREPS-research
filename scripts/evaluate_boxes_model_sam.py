import sys
import argparse
from pathlib import Path
import torch
sys.path.insert(0, '.')
from isegm.inference import utils
from isegm.utils.exp import load_config_file
from isegm.inference.predictors import get_predictor
from isegm.inference.evaluation import evaluate_dataset
from isegm.inference.utils import setup_deterministic
from isegm.model.is_sam_model import ISModelSAM
from isegm.utils.misc import get_checkpoints_list_and_logs_path, save_results, save_iou_analysis_data, get_prediction_vis_callback

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--deterministic', action='store_true', default=False, help='set deterministic evaluation')

    parser.add_argument('mode', choices=['NoBRS'], help='')

    group_checkpoints = parser.add_mutually_exclusive_group(required=True)
    group_checkpoints.add_argument('--checkpoint', type=str, default='',
                                   help='The path to the checkpoint. '
                                        'This can be a relative path (relative to cfg.INTERACTIVE_MODELS_PATH) '
                                        'or an absolute path. The file extension can be omitted.')

    group_checkpoints.add_argument('--exp-path', type=str, default='',
                                   help='The relative path to the experiment with checkpoints.'
                                        '(relative to cfg.EXPS_PATH)')

    parser.add_argument('--datasets', type=str, default='GrabCut,Berkeley,DAVIS,COCO_MVal,TETRIS',
                        help='List of datasets on which the model should be tested. '
                             'Datasets are separated by a comma')

    parser.add_argument('--modality', type=str, default='click', choices=['click', 'bbox'],
                        help="Modality to test on. Options are 'click' or 'bbox'")
    group_device = parser.add_mutually_exclusive_group()
    group_device.add_argument('--gpus', type=str, default='0',
                              help='ID of used GPU.')
    group_device.add_argument('--cpu', action='store_true', default=False,
                              help='Use only CPU for inference.')

    group_iou_thresh = parser.add_mutually_exclusive_group()
    group_iou_thresh.add_argument('--target-iou', type=float, default=0.90,
                                  help='Target IoU threshold for the NoC metric. (min possible value = 0.8)')
    group_iou_thresh.add_argument('--iou-analysis', action='store_true', default=False,
                                  help='Plot mIoU(number of clicks) with target_iou=1.0.')

    parser.add_argument('--thresh', type=float, required=False, default=0.49,
                        help='The segmentation mask is obtained from the probability outputs using this threshold.')
    parser.add_argument('--lambda_mult', type=float, required=False, default=0.1,
                        help='reg_value * lambda in loss')
    parser.add_argument('--save-ious', action='store_true', default=False)
    parser.add_argument('--print-ious', action='store_true', default=False)
    parser.add_argument('--vis-preds', action='store_true', default=False)
    parser.add_argument('--model-name', type=str, default=None,
                        help='The model name that is used for making plots.')
    parser.add_argument('--config-path', type=str, default='./config.yml',
                        help='The path to the config file.')
    parser.add_argument('--logs-path', type=str, default='',
                        help='The path to the evaluation logs. Default path: cfg.EXPS_PATH/evaluation_logs.')
    parser.add_argument('--optim_min', action='store_true', default=False, help='Minimization of iou during optimization')
    parser.add_argument('--vis_optim', action='store_true', default=False, help='Save Visualizations of optimization steps')
    parser.add_argument('--n_opt_steps', type=int, default=11, help='Number of steps to optimize')
    parser.add_argument('--lr_mult', type=float, default=1, help='LR for optim multiplyer')
    parser.add_argument('--n_workers', type=int, default=1, help='Number of parallel workers on inference')
    parser.add_argument('--n_samples', type=int, default=0, help='Slice only N samples (for debug only)')
    parser.add_argument('--n-clicks', type=int, default=20, help='Maximum number of clicks for the NoC metric.')
    parser.add_argument('--min-n-clicks', type=int, default=1, help='Minimum number of clicks for the evaluation.')
    parser.add_argument('--clicks-limit', type=int, default=20)
    parser.add_argument('--eval-mode', type=str, default='cv2', help='Evaluation mode (e.g. cv2)')
    args = parser.parse_args()
    if args.cpu:
        args.device = torch.device('cpu')
    else:
        args.device = torch.device(f"cuda:{args.gpus.split(',')[0]}")

    if (args.iou_analysis or args.print_ious) and args.min_n_clicks <= 1:
        args.target_iou = 1.01
    else:
        args.target_iou = max(0.8, args.target_iou)

    cfg = load_config_file(args.config_path, return_edict=True)
    cfg.EXPS_PATH = Path(cfg.EXPS_PATH)

    if args.logs_path == '':
        args.logs_path = cfg.EXPS_PATH / 'evaluation_logs'
    else:
        args.logs_path = Path(args.logs_path)

    return args, cfg


def main():
    args, cfg = parse_args()

    if args.deterministic:
        print("SAM use non-deterministic operators (e.g. cumsum)!")
        setup_deterministic(seed=0, is_full=False)

    checkpoints_list, logs_path, logs_prefix = get_checkpoints_list_and_logs_path(args, cfg)
    logs_path.mkdir(parents=True, exist_ok=True)

    single_model_eval = len(checkpoints_list) == 1
    assert not args.iou_analysis if not single_model_eval else True, \
        "Can't perform IoU analysis for multiple checkpoints"
    print_header = single_model_eval
    for dataset_name in args.datasets.split(','):
        dataset = utils.get_dataset(dataset_name, cfg, args)

        for checkpoint_path in checkpoints_list:
            model = ISModelSAM(device=args.device, model_path=checkpoint_path)

            predictor_params, zoomin_params = get_predictor_and_zoomin_params(args, dataset_name)
            predictor = get_predictor(model, args.mode, args.device,prompt_modality= args.modality,
                                      prob_thresh=args.thresh,
                                      predictor_params=predictor_params,
                                      zoom_in_params=zoomin_params, with_flip=False, model_name='sam')
            vis_callback = get_prediction_vis_callback(logs_path, dataset_name, args.thresh, args.modality) if args.vis_preds else None
            dataset_results = evaluate_dataset(dataset, predictor, pred_thr=args.thresh,
                                               max_iou_thr=args.target_iou,
                                               min_clicks=args.min_n_clicks,
                                               max_clicks=args.n_clicks,
                                               callback=vis_callback, args=args, support_deterministic=False)

            row_name = args.mode if single_model_eval else checkpoint_path.stem
            if args.iou_analysis:
                save_iou_analysis_data(args, dataset_name, logs_path,
                                       logs_prefix, dataset_results,
                                       model_name=args.model_name)

            save_results(args, row_name, dataset_name, logs_path, logs_prefix, dataset_results,
                         save_ious=single_model_eval and args.save_ious,
                         single_model_eval=single_model_eval,
                         print_header=print_header)

            print_header = False

            del predictor, model
            torch.cuda.empty_cache()

def get_predictor_and_zoomin_params(args, dataset_name):
    predictor_params = {}

    if args.clicks_limit is not None:
        if args.clicks_limit == -1:
            args.clicks_limit = args.n_clicks
        predictor_params['net_clicks_limit'] = args.clicks_limit

    # SAM does not use Zoomin
    zoom_in_params = None

    return predictor_params, zoom_in_params


if __name__ == '__main__':
    main()
