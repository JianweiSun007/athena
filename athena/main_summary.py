# coding=utf-8
# Copyright (C) 2019 ATHENA AUTHORS; Xiangang Li; Dongwei Jiang; Xiaoning Lei
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# Only support tensorflow 2.0
# pylint: disable=invalid-name, no-member, wildcard-import, unused-wildcard-import, redefined-outer-name
""" a sample implementation of LAS for HKUST """
import sys
import json
import tensorflow as tf
from absl import logging
from athena import *
import tensorboard

SUPPORTED_DATASET_BUILDER = {
    "speech_recognition_dataset": SpeechRecognitionDatasetBuilder,
    "speech_dataset": SpeechDatasetBuilder,
    "language_dataset": LanguageDatasetBuilder,
}

SUPPORTED_MODEL = {
    "deep_speech": DeepSpeechModel,
    "speech_transformer": SpeechTransformer,
    "speech_transformer2": SpeechTransformer2,
    "speech_transformer3": SpeechTransformer3,
    "mtl_transformer_ctc": MtlTransformerCtc,
    "mpc": MaskedPredictCoding,
    "rnnlm": RNNLM,
    "translate_transformer": NeuralTranslateTransformer,
}

SUPPORTED_OPTIMIZER = {
    "warmup_adam": WarmUpAdam,
    "expdecay_adam": ExponentialDecayAdam
}

DEFAULT_CONFIGS = {
    "batch_size": 32,
    "num_epochs": 20,
    "sorta_epoch": 1,
    "ckpt": None,
    "summary_dir": None,
    "solver_gpu": [0],
    "solver_config": None,
    "model": "speech_transformer",
    "num_classes": None,
    "model_config": None,
    "pretrained_model": None,
    "optimizer": "warmup_adam",
    "optimizer_config": None,
    "num_data_threads": 1,
    "dataset_builder": "speech_recognition_dataset",
    "trainset_config": None,
    "devset_config": None,
    "testset_config": None,
    "decode_config": None,
}

def parse_config(config):
    """ parse config """
    p = register_and_parse_hparams(DEFAULT_CONFIGS, config, cls="main")
    logging.info("hparams: {}".format(p))
    return p
@tf.function
def build_model_from_jsonfile(p,dataset_builder):
    """ creates model using configurations in json, load from checkpoint
    if previous models exist in checkpoint dir
    """
    model = SUPPORTED_MODEL[p.model](
        data_descriptions=dataset_builder,
        config=p.model_config,
    )

    return model


def model_summary(jsonfile,p):
    """ entry point for model training, implements train loop

	:param jsonfile: json file to read configuration from
	:param Solver: an abstract class that implements high-level logic of train, evaluate, decode, etc
	:param rank_size: total number of workers, 1 if using single gpu
	:param rank: rank of current worker, 0 if using single gpu
	"""
    summary_writer = tf.summary.create_file_writer(p.summary_dir + 'log')
    tf.summary.trace_on(graph=True, profiler=False)
    dataset_builder = SUPPORTED_DATASET_BUILDER[p.dataset_builder](p.trainset_config)
    model = build_model_from_jsonfile(p, dataset_builder)
    with summary_writer.as_default():
        tf.summary.trace_export(name="model_trace", step=0, profiler_outdir=p.summary_dir + 'log')





if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    if len(sys.argv) < 2:
        logging.warning('Usage: python {} config_json_file'.format(sys.argv[0]))
        sys.exit()
    tf.random.set_seed(1)

    json_file = sys.argv[1]
    config = None
    with open(json_file) as f:
        config = json.load(f)
    p = parse_config(config)
    BaseSolver.initialize_devices(p.solver_gpu)
    model_summary(json_file,p)
