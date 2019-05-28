# cv2_saliency_prediction
Computer Vision 2 Project at Uni Hamburg

# How to run
## train
python src/second_try.py -o train -dd ~/cv2_saliency_prediction/cv2_data -md ~/cv2_saliency_prediction/model -ld ~/cv2_saliency_prediction/logs-pd ~/cv2_saliency_prediction/preds -cpp ~/cv2_saliency_prediction/checkpoints/cp.ckpt

## predict
python src/second_try.py -o predict -dd ~/cv2_saliency_prediction/cv2_data -md ~/cv2_saliency_prediction/model -ld ~/cv2_saliency_prediction/logs-pd ~/cv2_saliency_prediction/preds -cpp ~/cv2_saliency_prediction/checkpoints/cp.ckpt

