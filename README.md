# Music_VAE_Colab
초보 시절 작성한 것으로, 설명은 대부분 틀립니다. 

## Some code
https://github.com/magenta/magenta/blob/77ed668af96edea7c993d38973b9da342bd31e82/magenta/models/music_vae/base_model.py
https://github.com/magenta/magenta/blob/77ed668af96edea7c993d38973b9da342bd31e82/magenta/models/music_vae/lstm_models.py
https://github.com/magenta/magenta/blob/77ed668af96edea7c993d38973b9da342bd31e82/magenta/contrib/rnn.py
https://github.com/magenta/magenta/blob/77ed668af96edea7c993d38973b9da342bd31e82/magenta/models/music_vae/lstm_utils.py
에서 모델을 자세히 볼 수 있습니다. 
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

## Result
- [gen500](https://github.com/gyunggyung/Music_VAE_Colab/tree/main/gen500)은 500 step동안 학습한 model로, 생성한 결과를 저장한 폴더입니다. 
- [gen2742](https://github.com/gyunggyung/Music_VAE_Colab/tree/main/gen2742)는 2742 step동안 학습을 진행했습니다. 
- [gen2742_2](https://github.com/gyunggyung/Music_VAE_Colab/tree/main/gen2742)는 똑같은 model로 한 번 더 생성한 결과입니다.

## Reference
- A Hierarchical Latent Vector Model for Learning Long-Term Structure in Music https://arxiv.org/pdf/1803.05428.pdf
- magenta/magenta https://github.com/magenta/magenta
- Groove MIDI Dataset https://magenta.tensorflow.org/datasets/groove
- magenta-demos Colab https://colab.research.google.com/github/magenta/magenta-demos/blob/master/colab-notebooks/MusicVAE.ipynb
- MusicVAE - Training stops at epoch 0 with no output or explanation. #1549  https://github.com/magenta/magenta/issues/1549
- tf.keras.utils.get_file https://www.tensorflow.org/api_docs/python/tf/keras/utils/get_file
- zipfile — ZIP 아카이브 작업 https://docs.python.org/ko/3/library/zipfile.html
