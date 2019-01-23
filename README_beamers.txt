##### README TO RUN DECODING VARIATIONS

source /data2/the_beamers/venv_pytorch/bin/activate

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

## BEAM SEARCH PENALIZING FOR OVERLAPPING TOKENS
## Implemented 12/19
python3 /data2/the_beamers/OpenNMT-reno/translate.py \
-model /data2/the_beamers/models/opennmt_sample_model.pt \
-src /data2/the_beamers/data/test/test_input.txt \
-output /data2/the_beamers/data/output_reno/test_output_10beam_1.0hamming.txt \
-beam_size 10 \
-n_best 10 \
-max_length 50 \
-block_ngram_repeat 1 \
-replace_unk \
-gpu 0 \
-batch_size 1 \
-hamming_penalty 1.0

## BEAM SEARCH PENALIZING OVERLAPPING TOKENS AND
## LIMITING FOR NUMBER OF CANDIDATES PER HYPOTHESIS
## Implemented 12/19
python3 /data2/the_beamers/OpenNMT-reno/translate.py \
-model /data2/the_beamers/models/opennmt_sample_model.pt \
-src /data2/the_beamers/data/test/test_input.txt \
-output /data2/the_beamers/data/output_reno/test_output_10beam_3percands_1.0hamming.txt \
-beam_size 10 \
-n_best 10 \
-max_length 50 \
-block_ngram_repeat 1 \
-replace_unk \
-gpu 0 \
-batch_size 1 \
-k_per_cand 3 \
-hamming_penalty 1.0

paste /data2/the_beamers/data/output_reno/test_output_10beam.txt /data2/the_beamers/data/output_reno/test_output_10beam_3percands.txt | less

## ITERATIVE BEAM SEARCH
python3 /data2/the_beamers/OpenNMT-reno/translate.py \
-model /data2/the_beamers/models/opennmt_sample_model.pt \
-src /data2/the_beamers/data/test/test_input.txt \
-output /data2/the_beamers/data/output_reno/test_output_5beam_10iter.txt \
-beam_size 5 \
-n_best 1 \
-max_length 50 \
-block_ngram_repeat 1 \
-replace_unk \
-gpu 0 \
-batch_size 1 \
-beam_iters 10

