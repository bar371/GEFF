from ReIDModules.CAL.tools.eval_metrics import evaluate, evaluate_with_clothes
# from Scripts.inference import args


def evalute_wrapper(dataset, distmat,  q_pids, g_pids, q_camids, g_camids, q_clothes_ids, g_clothes_ids, extra_msg):
    print("Computing CMC and mAP ", extra_msg)
    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)
    print("Results ---------------------------------------------------")
    print('top1:{:.1%} top5:{:.1%} top10:{:.1%} top20:{:.1%} mAP:{:.1%}'.format(cmc[0], cmc[4], cmc[9], cmc[19], mAP))
    print("-----------------------------------------------------------")
    if dataset in ['ltcc', 'ccvid']:
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
