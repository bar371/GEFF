from ReIDModules.CAL.tools.eval_metrics import evaluate, evaluate_with_clothes
# from Scripts.inference import args
TRACK_MIN_IMGS = 10


def evalute_wrapper(dataset, distmat,  q_pids, g_pids, q_camids, g_camids, q_clothes_ids, g_clothes_ids, extra_msg):
    print("Computing CMC and mAP ", extra_msg)
    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)
    print("Results ---------------------------------------------------")
    print('top1:{:.1%} top5:{:.1%} top10:{:.1%} top20:{:.1%} mAP:{:.1%}'.format(cmc[0], cmc[4], cmc[9], cmc[19], mAP))
    print("-----------------------------------------------------------")
    if dataset in ['ltcc', 'ccvid', 'vcclothes']:
        print('Evaluating clothes changing with mode:SC')
        print("Results ---------------------------------------------------")
        cmc, mAP = evaluate_with_clothes(distmat, q_pids, g_pids, q_camids, g_camids, q_clothes_ids, g_clothes_ids,
                                         mode='SC')
        print(
            'top1:{:.1%} top5:{:.1%} top10:{:.1%} top20:{:.1%} mAP:{:.1%}'.format(cmc[0], cmc[4], cmc[9], cmc[19], mAP))
        print("-----------------------------------------------------------")

        print('Evaluating clothes changing with mode:CC')
        print("Results ---------------------------------------------------")
        cmc, mAP = evaluate_with_clothes(distmat, q_pids, g_pids, q_camids, g_camids, q_clothes_ids, g_clothes_ids,
                                         mode='CC')
        print(
            'top1:{:.1%} top5:{:.1%} top10:{:.1%} top20:{:.1%} mAP:{:.1%}'.format(cmc[0], cmc[4], cmc[9], cmc[19], mAP))
        print("-----------------------------------------------------------")


def evaluate_performance_ccvid(tracks_results, alpha):
    correct_predictions = 0
    incorrect_predictions = 0
    for track_prediction in tracks_results:
        track_final_scores = track_prediction.get('final_scores')
        track_true_id = track_prediction.get('track_id').split('_')[1]
        predicted_id = max(track_final_scores, key=track_final_scores.get)
        if int(track_true_id) == predicted_id:
            correct_predictions += 1
        else:
            incorrect_predictions += 1

    print(f'Running with alpha: {alpha}\n'
          f'Predicted correctly: {correct_predictions}\n'
          f'Predicted incorrectly: {incorrect_predictions}\n'
          f'Accuracy: {correct_predictions / (correct_predictions + incorrect_predictions)}\n')


def evaluate_performance_42street(tracks_results, alpha, tracklets_num_imgs):
    correct_predictions_per_track = 0
    incorrect_predictions_per_track = 0
    unknown_id_per_track = 0
    correct_predictions_per_image = 0
    incorrect_predictions_per_image = 0
    unknown_id_per_image = 0
    for track_prediction in tracks_results:
        track_num_imgs = len(tracklets_num_imgs[track_prediction.get('track_id')])
        if track_num_imgs < TRACK_MIN_IMGS:
            continue
        track_true_id = track_prediction.get('track_id').split('_')[3]
        track_final_scores = track_prediction.get('final_scores')
        predicted_id = max(track_final_scores, key=track_final_scores.get)
        if int(track_true_id) == 12:
            unknown_id_per_track += 1
            unknown_id_per_image += track_num_imgs
            continue
        if int(track_true_id) == predicted_id:
            correct_predictions_per_track += 1
            correct_predictions_per_image += track_num_imgs
        else:
            incorrect_predictions_per_track += 1
            incorrect_predictions_per_image += track_num_imgs

    print(f'Running with alpha: {alpha}\n'
          f'Total tracks: {correct_predictions_per_track + incorrect_predictions_per_track}\n'
          f'Result for track-based evaluation:\n'
          f'Predicted correctly: {correct_predictions_per_track}\n'
          f'Predicted incorrectly: {incorrect_predictions_per_track}\n'
          f'Unknown IDs: {unknown_id_per_track}\n'
          f'Accuracy: {correct_predictions_per_track / (correct_predictions_per_track + incorrect_predictions_per_track)}\n')

    print(f'Result for image-based evaluation:\n'
          f'Total tracks: {correct_predictions_per_image + incorrect_predictions_per_image}\n'
          f'Predicted correctly: {correct_predictions_per_image}\n'
          f'Predicted incorrectly: {incorrect_predictions_per_image}\n'
          f'Unknown IDs: {unknown_id_per_image}\n'
          f'Accuracy: {correct_predictions_per_image / (correct_predictions_per_image + incorrect_predictions_per_image)}\n')
