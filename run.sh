#triviaQA
#python main.py --gpu 0 --trainFile data/triviaQA/wikipedia-train.csv --devFile data/triviaQA/wikipedia-dev.csv --tag wiki &
#python main.py --gpu 1 --trainFile data/triviaQA/wikipedia-train.csv --devFile data/triviaQA/wikipedia-dev.csv --queryFilter true --tag wiki &
#python main.py --gpu 2 --trainFile data/triviaQA/wikipedia-train.csv --devFile data/triviaQA/wikipedia-dev.csv --testMaxLen 400 --tag wiki &
#python main.py --gpu 3 --trainFile data/triviaQA/wikipedia-train.csv --devFile data/triviaQA/wikipedia-dev.csv --queryFilter true --testMaxLen 400 --tag wiki &

#python main.py --gpu 0 --senSel false --trainFile data/triviaQA/wikipedia-train.csv --devFile data/triviaQA/wikipedia-dev.csv --tag wiki &
#python main.py --gpu 1 --senSel false --trainFile data/triviaQA/wikipedia-train.csv --devFile data/triviaQA/wikipedia-dev.csv --queryFilter true --tag wiki &
#python main.py --gpu 2 --senSel false --trainFile data/triviaQA/wikipedia-train.csv --devFile data/triviaQA/wikipedia-dev.csv --testMaxLen 400 --tag wiki &
#python main.py --gpu 3 --senSel false --trainFile data/triviaQA/wikipedia-train.csv --devFile data/triviaQA/wikipedia-dev.csv --queryFilter true --testMaxLen 400 --tag wiki &

#python main.py --sameLSTM false --gpu 0 --senSel false --trainFile data/triviaQA/wikipedia-train.csv --devFile data/triviaQA/wikipedia-dev.csv --tag wiki &
#python main.py --sameLSTM false --gpu 1 --senSel false --trainFile data/triviaQA/wikipedia-train.csv --devFile data/triviaQA/wikipedia-dev.csv --queryFilter true --tag wiki &
#python main.py --sameLSTM false --gpu 2 --senSel false --trainFile data/triviaQA/wikipedia-train.csv --devFile data/triviaQA/wikipedia-dev.csv --testMaxLen 400 --tag wiki &
#python main.py --sameLSTM false --gpu 3 --senSel false --trainFile data/triviaQA/wikipedia-train.csv --devFile data/triviaQA/wikipedia-dev.csv --queryFilter true --testMaxLen 400 --tag wiki &

#python main.py --sameLSTM false --gpu 0 --trainFile data/triviaQA/wikipedia-train.csv --devFile data/triviaQA/wikipedia-dev.csv --tag wiki &
#python main.py --sameLSTM false --gpu 1 --trainFile data/triviaQA/wikipedia-train.csv --devFile data/triviaQA/wikipedia-dev.csv --queryFilter true --tag wiki &
#python main.py --sameLSTM false --gpu 2 --trainFile data/triviaQA/wikipedia-train.csv --devFile data/triviaQA/wikipedia-dev.csv --testMaxLen 400 --tag wiki &
#python main.py --sameLSTM false --gpu 3 --trainFile data/triviaQA/wikipedia-train.csv --devFile data/triviaQA/wikipedia-dev.csv --queryFilter true --testMaxLen 400 --tag wiki &

#SQuAD
#python main.py --epoch 20 --batch 32 --gpu 1 --tag squad_v3_senTokenize_false
#python main.py --gpu 0 --batch 16 --tag newsQA --trainFile data/newsQA/train.csv --devFile data/newsQA/dev.csv
#python main.py --gpu 0 --batch 16 --tag triviaQA --trainFile data/triviaQA/wikipedia-train.csv --devFile data/triviaQA/wikipedia-dev.csv
#python main.py --gpu 0 --batch 16 --mode test --devFile data/triviaQA/wikipedia-dev.csv  --loadModelDir 1109143226_maxlen_400_testmaxlen_1000_queryFilter_false_sensel_false_epoch_10_b_16_sameLSTM_true_wiq_true_embedFix_true_triviaQA_withMask_preprocess  --loadModelFile b_16_e_300_h_300_56999

python main.py --gpu 0 --mode test --loadModelDir 1029133928_maxlen_400_testmaxlen_1000_queryFilter_false_sensel_false_epoch_20_b_32_sameLSTM_true_wiq_true_embedFix_true_squad --loadModelFile b_32_e_300_h_300_42999

#python main.py --gpu 1 --tag word_dropout
#python main.py --gpu 1 --mode debug --loadModelDir 1029133928_maxlen_400_testmaxlen_1000_queryFilter_false_sensel_false_epoch_20_b_32_sameLSTM_true_wiq_true_embedFix_true_squad  --loadModelFile b_32_e_300_h_300_30999
#python main.py --gpu 0 --mode debug --loadModelDir 1027162359_maxlen_400_testmaxlen_1000_queryFilter_false_sensel_false_epoch_20_b_64_sameLSTM_true_embedFix_true_squad_no_masking
#python main.py --gpu 0 --mode debug --loadModelDir 1027162359_maxlen_400_testmaxlen_1000_queryFilter_false_sensel_false_epoch_20_b_64_sameLSTM_true_embedFix_true_squad_no_masking 	--loadModelFile b_64_e_300_h_300_0
#python main.py --gpu 1 --mode debug --loadModelDir 1024174825_maxlen_400_testmaxlen_1000_queryFilter_false_sensel_true_e_300_h_300_lr_0.001_b_16_sameLSTM_true_embedFix_true_init_rand_param_rand --loadModelFile b_16_e_300_h_300_1999
#python main.py --gpu 0 --mode debug --loadModelDir 1026112534_maxlen_400_testmaxlen_1000_queryFilter_false_sensel_true_e_300_h_300_lr_0.001_b_16_sameLSTM_true_embedFix_true_word_dropout --loadModelFile b_16_e_300_h_300_7999
#python main.py --gpu 0 --mode debug --devFile data/triviaQA/wikipedia-dev.csv --loadModelDir 1022101522_maxlen_400_testmaxlen_1000_queryFilter_false_sensel_false_e_300_h_300_lr_0.001_b_16_sameLSTM_true_embedFix_true_wiki_train

#python main.py --gpu 1 --mode debug --loadModelDir 1027070907_maxlen_400_testmaxlen_1000_queryFilter_false_sensel_true_epoch_20_b_64_sameLSTM_true_embedFix_true_squad_loss_update2
#python main.py --gpu 0 --mode debug --loadModelDir 1026112534_maxlen_400_testmaxlen_1000_queryFilter_false_sensel_true_e_300_h_300_lr_0.001_b_16_sameLSTM_true_embedFix_true_word_dropout 
#python main.py --gpu 0 --mode debug --loadModelDir 1026135819_maxlen_400_testmaxlen_1000_queryFilter_false_sensel_true_e_300_h_300_lr_0.001_b_64_sameLSTM_true_embedFix_true_SQuAD_dropout --loadModelFile b_64_e_300_h_300_1999
#python main.py --gpu 0 --mode debug --loadModelDir 1026135819_maxlen_400_testmaxlen_1000_queryFilter_false_sensel_true_e_300_h_300_lr_0.001_b_64_sameLSTM_true_embedFix_true_SQuAD_dropout --loadModelFile b_64_e_300_h_300_11999

#python main.py --gpu 0 --mode debug --loadModelDir 1026135819_maxlen_400_testmaxlen_1000_queryFilter_false_sensel_true_e_300_h_300_lr_0.001_b_64_sameLSTM_true_embedFix_true_SQuAD_dropout


#python main.py --gpu 0 --trainFile data/newsQA/train.csv --devFile data/newsQA/dev.csv --tag newsQA_dropout&
#python main.py --epoch 50 --gpu 1 --tag SQuAD_dropout

#newsQA
#python main.py --gpu 0 --tag news &
#python main.py --gpu 1 --queryFilter true --tag news &
#python main.py --gpu 2 --testMaxLen 400 --tag news &
#python main.py --gpu 3 --queryFilter true --testMaxLen 400 --tag news &

#python main.py --senSel false --gpu 0 --tag news &
#python main.py --senSel false --gpu 1 --queryFilter true --tag news &
#python main.py --senSel false --gpu 2 --testMaxLen 400 --tag news &
#python main.py --senSel false --gpu 3 --queryFilter true --testMaxLen 400 --tag news &

#python3 main.py --sameLSTM false --gpu 0 --tag squad &
#python3 main.py --sameLSTM false --gpu 1 --queryFilter true --tag squad &
#python3 main.py --sameLSTM false --gpu 0 --testMaxLen 400 --tag squad &
#python3 main.py --sameLSTM false --gpu 1 --queryFilter true --testMaxLen 400 --tag squad &

#python3 main.py --sameLSTM false --senSel false --gpu 0 --tag squad &
#python3 main.py --sameLSTM false --senSel false --gpu 1 --queryFilter true --tag squad &
#python3 main.py --sameLSTM false --senSel false --gpu 2 --testMaxLen 400 --tag squad &
#python3 main.py --sameLSTM false --senSel false --gpu 3 --queryFilter true --testMaxLen 400 --tag squad &



#python3 main.py --gpu 0 --senSel false --trainFile data/triviaQA/wikipedia-train.csv --devFile data/triviaQA/verified-wikipedia-dev.csv --tag verified-wiki &
#python3 main.py --gpu 1 --senSel false --trainFile data/triviaQA/web-train.csv --devFile data/triviaQA/verified-web-dev.csv --tag verified-web &

#python3 main.py --gpu 0 --senSel false --devFile data/triviaQA/verified-web-dev.csv --tag verified-web --mode test --loadModelDir 1011102929_maxlen_400_testmaxlen_1000_queryFilter_false_sensel_false_e_300_h_300_lr_0.001_b_16_sameLSTM_true_embedFix_true_verified-web
#python3 main.py --gpu 0 --senSel false --devFile data/triviaQA/verified-wikipedia-dev.csv --tag verified-wiki --mode test --loadModelDir 1010173948_maxlen_400_testmaxlen_1000_queryFilter_false_sensel_false_e_300_h_300_lr_0.001_b_16_sameLSTM_true_embedFix_true_verified-wiki

#python3 main.py --gpu 0 --senSel false --trainFile data/triviaQA/web-train.csv --devFile data/triviaQA/verified-web-dev.csv --tag web_train
#python3 main.py --gpu 1 --senSel false --trainFile data/triviaQA/wikipedia-train.csv --devFile data/triviaQA/verified-wikipedia-dev.csv --tag wiki_train

#python3 main.py --gpu 0 --senSel false --mode test --devFile data/triviaQA/verified-wikipedia-dev.csv --mode test --loadModelDir  --loadModelFile 
#python3 main.py --gpu 1 --senSel false --mode test --devFile data/triviaQA/wikipedia-dev.csv --mode test --loadModelDir 1018122135_maxlen_400_testmaxlen_1000_queryFilter_false_sensel_false_e_300_h_300_lr_0.001_b_16_sameLSTM_true_embedFix_true_wiki_train --loadModelFile b_16_e_300_h_300_17999
#python3 main.py --gpu 0 --senSel false --devFile data/triviaQA/verified-web-dev.csv --mode test --loadModelDir 1018101524_maxlen_400_testmaxlen_1000_queryFilter_false_sensel_false_e_300_h_300_lr_0.001_b_16_sameLSTM_true_embedFix_true_web_train --loadModelFile b_16_e_300_h_300_11999



#python3 main.py --gpu 1 --senSel false --trainFile data/triviaQA/web-train.csv --devFile data/triviaQA/web-dev.csv --queryFilter true --tag web &
#python main.py --sameLSTM false --gpu 2 --senSel false --trainFile data/triviaQA/wikipedia-train.csv --devFile data/triviaQA/wikipedia-dev.csv --testMaxLen 400 --tag wiki &
#python main.py --sameLSTM false --gpu 3 --senSel false --trainFile data/triviaQA/wikipedia-train.csv --devFile data/triviaQA/wikipedia-dev.csv --queryFilter true --testMaxLen 400 --tag wiki &

#python main.py --sameLSTM false --gpu 0 --trainFile data/triviaQA/wikipedia-train.csv --devFile data/triviaQA/wikipedia-dev.csv --tag wiki &
#python main.py --sameLSTM false --gpu 1 --trainFile data/triviaQA/wikipedia-train.csv --devFile data/triviaQA/wikipedia-dev.csv --queryFilter true --tag wiki &
#python main.py --sameLSTM false --gpu 2 --trainFile data/triviaQA/wikipedia-train.csv --devFile data/triviaQA/wikipedia-dev.csv --testMaxLen 400 --tag wiki &
#python main.py --sameLSTM false --gpu 3 --trainFile data/triviaQA/wikipedia-train.csv --devFile data/triviaQA/wikipedia-dev.csv --queryFilter true --testMaxLen 400 --tag wiki &

