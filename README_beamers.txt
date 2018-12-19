##### README TO RUN DECODING VARIATIONS

source /data2/the_beamers/virtualenv/bin/activate

## ORIGINAL BEAM SEARCH
python3 /data2/the_beamers/OpenNMT-reno/translate.py \
-model /data2/the_beamers/models/opennmt_sample_model.pt \
-src /data2/the_beamers/data/test/test_input.txt \
-output /data2/the_beamers/data/output_reno/test_output_10beam.txt \
-beam_size 10 \
-n_best 10 \
-max_length 50 \
-block_ngram_repeat 1 \
-replace_unk \
-gpu 0 \
-batch_size 1

## BEAM SEARCH LIMITING FOR NUMBER OF CANDIDATES PER HYPOTHESIS
## Implemented 12/19
python3 /data2/the_beamers/OpenNMT-reno/translate.py \
-model /data2/the_beamers/models/opennmt_sample_model.pt \
-src /data2/the_beamers/data/test/test_input.txt \
-output /data2/the_beamers/data/output_reno/test_output_10beam_3percands.txt \
-beam_size 10 \
-n_best 10 \
-max_length 50 \
-block_ngram_repeat 1 \
-replace_unk \
-gpu 0 \
-batch_size 1 \
-k_per_cand 3

