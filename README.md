> train your own dataset by   <br>
>> "python main.py --raw_data_prefix $GEMINI_DATA_IN1/ --data_name gowalla --out_prefix $GEMINI_DATA_OUT/ --min_loc_freq 10 --min_user_freq 20 --n_epoch 35 --unvisit_loc_t True --n_nearest 2000 --train_n_neg 15 --eval_n_neg 100 --train_batch_size 35 --eval_batch_size 35 --k_t 10 --k_d 15 --dimension 128 --exp_factor 1"
 <br>
> $GEMINI_DATA_IN1/ means the path of data  <br>
$GEMINI_DATA_OUT/ means the path of the output dir <br>

