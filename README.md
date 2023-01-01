# Music_VAE_Colab

* gen500은 500 step동안 학습한 model이 생성한 폴더입니다. gen2742는 2742 step동안 학습을 진행했습니다. gen2742_2는 똑같은 model로 한 번 더 생성한 결과입니다.

https://github.com/magenta/magenta/blob/77ed668af96edea7c993d38973b9da342bd31e82/magenta/models/music_vae/base_model.py
https://github.com/magenta/magenta/blob/77ed668af96edea7c993d38973b9da342bd31e82/magenta/models/music_vae/lstm_models.py
https://github.com/magenta/magenta/blob/77ed668af96edea7c993d38973b9da342bd31e82/magenta/contrib/rnn.py
https://github.com/magenta/magenta/blob/77ed668af96edea7c993d38973b9da342bd31e82/magenta/models/music_vae/lstm_utils.py
에서 모델을 자세히 볼 수 있습니다. 아래 내용 중, 혹시 틀린 부분이 있을 경우, 알려주시면 감사하겠습니다! 핵심만 쓰기 위해, 많은 중간과정을 생략하고 있습니다.

## ENCODER
ENCODER는 논문과 같이 latent distribution parameters(μ and σ)를 만드는데 필요한, BiLSTM의 final state vectors를 얻습니다.

## HIERARCHICAL DECODER
DECODER는 논문과 같이 HIERARCHICAL DECODER를 사용합니다. 입력 sequence(drum midi)를 4마디마다 subsequences로 분할합니다. subsequences를 unidirectional LSTM에 넣어, 각각의 embedding vectors c(Conductuor)를 듭니다. 이를 이용하여, 생성을 진행합니다. 이러한 방법으로 긴 sequence도 학습 및 생성을 진행할 수 있습니다.

## Multi-Stream Modeling
우리의 목표는, 4마디에 해당하는 drum 샘플을 뽑아내는 것입니다. 따라서 Multi-Stream Modeling는 사실상 진행하지 않습니다. 다만 구현된 모델에 맞게 학습을 위해, drum distributions만 설정하게 사용했습니다. Multi-Stream Modeling은 다양한 signal로 연주하기 위해서, independent한 3 separate distributions(drum, bass, and melody)을 만듭니다. 이는 별도의 DECODER를 사용하여, 생성을 진행합니다.

## ETC
이외에 사항들은 VAE 생성 방식과 유사합니다.

## Some code

```
    Encodes input sequences into a precursors for latent code `z`.
    Args:
       sequence: Batch of sequences to encode.
       sequence_length: Length of sequences in input batch.
    Returns:
       outputs: Raw outputs to parameterize the prior distribution in
          MusicVae.encode, sized `[batch_size, N]`.
    ....
    
    last_h_fw = states_fw[-1][-1].h
    last_h_bw = states_bw[-1][-1].h

    return tf.concat([last_h_fw, last_h_bw], 1)
```

```
    """Initializer for HierarchicalLstmDecoder.
    Hierarchicaly decodes a sequence across time.
    Each sequence is padded per-segment. For example, a sequence with
    three segments [1, 2, 3], [4, 5], [6, 7, 8 ,9] and a `max_seq_len` of 12
    is represented as `sequence = [1, 2, 3, 0, 4, 5, 0, 0, 6, 7, 8, 9]` with
    `sequence_length = [3, 2, 4]`.
    `z` initializes the first level LSTM to produce embeddings used to
    initialize the states of LSTMs at subsequent levels. The lowest-level
    embeddings are then passed to the given `core_decoder` to generate the
    final outputs.
    This decoder has 3 modes for what is used as the inputs to the LSTMs
    (excluding those in the core decoder):
      Autoregressive: (default) The inputs to the level `l` decoder are the
        final states of the level `l+1` decoder.
      Non-autoregressive: (`disable_autoregression=True`) The inputs to the
        hierarchical decoders are 0's.
      Re-encoder: (`hierarchical_encoder` provided) The inputs to the level `l`
        decoder are re-encoded outputs of level `l+1`, using the given encoder's
        matching level.
    Args:
      core_decoder: The BaseDecoder implementation to use at the output level.
      level_lengths: A list of the number of outputs of each level of the
        hierarchy. The final level is the (padded) maximum length. The product
        of the lengths must equal `hparams.max_seq_len`.
      disable_autoregression: Whether to disable the autoregression within the
        hierarchy. May also be a collection of levels on which to disable.
      hierarchical_encoder: (Optional) A HierarchicalLstmEncoder instance to use
        for re-encoding the decoder outputs at each level for use as inputs to
        the next level up in the hierarchy, instead of the final decoder state.
        The encoder level output lengths (except for the final single-output
        level) should be the reverse of `level_output_lengths`.
    Raises:
      ValueError: If `hierarchical_encoder` is given but has incompatible level
        lengths.
    """
```

## Reference
- A Hierarchical Latent Vector Model for Learning Long-Term Structure in Music https://arxiv.org/pdf/1803.05428.pdf
- magenta/magenta https://github.com/magenta/magenta
- Groove MIDI Dataset https://magenta.tensorflow.org/datasets/groove
- magenta-demos Colab https://colab.research.google.com/github/magenta/magenta-demos/blob/master/colab-notebooks/MusicVAE.ipynb
- MusicVAE - Training stops at epoch 0 with no output or explanation. #1549  https://github.com/magenta/magenta/issues/1549
- tf.keras.utils.get_file https://www.tensorflow.org/api_docs/python/tf/keras/utils/get_file
- zipfile — ZIP 아카이브 작업 https://docs.python.org/ko/3/library/zipfile.html