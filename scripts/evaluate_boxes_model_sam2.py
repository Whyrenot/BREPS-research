import sys
sys.path.insert(0, '.')
from isegm.inference import utils
from isegm.inference.predictors import get_predictor
from isegm.inference.evaluation import evaluate_dataset
from isegm.inference.utils import setup_deterministic
from isegm.model.is_sam2_model import ISModelSAM2
from isegm.utils.misc import get_checkpoints_list_and_logs_path, save_results, save_iou_analysis_data, get_prediction_vis_callback
from evaluate_boxes_model_sam import get_predictor_and_zoomin_params, parse_args



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
            model = ISModelSAM2(device=args.device, model_path=checkpoint_path)

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





if __name__ == '__main__':
    main()
