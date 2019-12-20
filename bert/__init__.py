python -m language.question_answering.bert_joint.prepare_nq_data   --logtostderr   --input_jsonl data/simplified-nq-test.jsonl   --output_tfrecord out/nq-train.tfrecords-00000-of-00001   --max_seq_length=512   --include_unknowns=0.02   --vocab_file=bert-joint-baseline/vocab-nq.txt


/data1/achaptykov/model/googlebert

PATH=/data1/achaptykov/model/googlebert/pyth36:$PATH ./gsutil version -l | grep python
