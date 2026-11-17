# SIFT

python main.py --mode train --data $TRAIN --method sift --vocab_size 300 --bow saved\sift\bow_balanced.pkl --model saved\sift\model_balanced.pkl --classes saved\classes.pkl

python main.py --mode test --data $TEST --bow saved\sift\bow.pkl --model saved\sift\model.pkl --classes saved\classes.pkl

python main.py --mode predict --image $PRED_IMG --bow saved\sift\bow.pkl --model saved\sift\model.pkl --classes saved\classes.pkl --plot




python main.py --mode train --data $TRAIN --method orb --vocab_size 300 --bow saved\orb\bow.pkl --model saved\orb\model.pkl --classes saved\classes.pkl

python main.py --mode test --data $TEST --bow saved\orb\bow.pkl --model saved\orb\model.pkl --classes saved\classes.pkl

python main.py --mode predict --image $PRED_IMG --bow saved\orb\bow.pkl --model saved\orb\model.pkl --classes saved\classes.pkl --plot




python main.py --mode train --data $TRAIN --method resnet18 --vocab_size 200 --bow saved\resnet18\bow.pkl --model saved\resnet18\model.pkl --classes saved\classes.pkl

python main.py --mode test --data $TEST --bow saved\resnet18\bow.pkl --model saved\resnet18\model.pkl --classes saved\classes.pkl

python main.py --mode predict --image $PRED_IMG --bow saved\resnet18\bow.pkl --model saved\resnet18\model.pkl --classes saved\classes.pkl --plot


[test] Метрики:
[Classifier] Accuracy: 0.4545
              precision    recall  f1-score   support

           0       0.17      0.34      0.23        64
           1       0.82      0.64      0.72       540
           2       0.64      0.06      0.10       326
           3       0.47      0.58      0.52       280
           4       0.22      0.13      0.16       198
           5       0.26      0.69      0.38       224
           6       0.64      0.66      0.65       358
           7       0.43      0.59      0.50       322
           8       0.38      0.46      0.42       204
           9       0.26      0.43      0.33        88
          10       0.82      0.51      0.63       286
          11       0.62      0.28      0.39       114
          12       0.53      0.40      0.46       386
          13       0.23      0.77      0.35       104
          14       0.24      0.70      0.35       114
          15       0.76      0.16      0.26       206
          16       0.30      0.20      0.24       176
          17       0.42      0.46      0.44        92
          18       0.37      0.30      0.33       100
          19       0.48      0.31      0.38        90
          20       0.74      0.64      0.68        44
          21       0.69      0.60      0.64        90
          22       0.80      0.42      0.55        38
          23       0.67      0.35      0.46        34
          24       0.91      0.43      0.59        46
          25       0.77      0.71      0.74        56
          26       0.00      0.00      0.00        28
          27       0.90      0.33      0.49        54
          28       1.00      0.09      0.16        46

    accuracy                           0.45      4708
   macro avg       0.54      0.42      0.42      4708
weighted avg       0.55      0.45      0.45      4708



[test] Метрики:
[Classifier] Accuracy: 0.1376
              precision    recall  f1-score   support

           0       0.04      0.19      0.07        64
           1       0.21      0.24      0.22       540
           2       0.15      0.01      0.02       326
           3       0.09      0.14      0.11       280
           4       0.05      0.05      0.05       198
           5       0.10      0.21      0.14       224
           6       0.17      0.18      0.17       358
           7       0.15      0.03      0.05       322
           8       0.27      0.31      0.29       204
           9       0.02      0.07      0.02        88
          10       0.20      0.31      0.24       286
          11       0.33      0.12      0.18       114
          12       0.13      0.10      0.11       386
          13       0.07      0.10      0.08       104
          14       0.14      0.35      0.20       114
          15       0.20      0.11      0.14       206
          16       0.20      0.02      0.04       176
          17       0.12      0.04      0.06        92
          18       0.04      0.02      0.03       100
          19       0.00      0.00      0.00        90
          20       0.35      0.36      0.36        44
          21       0.00      0.00      0.00        90
          22       0.10      0.05      0.07        38
          23       0.00      0.00      0.00        34
          24       0.24      0.17      0.20        46
          25       0.40      0.21      0.28        56
          26       0.00      0.00      0.00        28
          27       0.00      0.00      0.00        54
          28       0.00      0.00      0.00        46

    accuracy                           0.14      4708
   macro avg       0.13      0.12      0.11      4708
weighted avg       0.15      0.14      0.13      4708
