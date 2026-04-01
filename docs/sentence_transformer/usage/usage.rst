
Usage
=====

Characteristics of Sentence Transformer (a.k.a bi-encoder) models:

1. Calculates a **fixed-size vector representation (embedding)** given **texts, images, audio, video, or combinations thereof** (depending on the model).
2. Embedding calculation is often **efficient**, embedding similarity calculation is **very fast**.
3. Applicable for a **wide range of tasks**, such as semantic textual similarity, semantic search, clustering, classification, paraphrase mining, and more.
4. Often used as a **first step in a two-step retrieval process**, where a Cross-Encoder (a.k.a. reranker) model is used to re-rank the top-k results from the bi-encoder.

Once you have `installed <../../installation.html>`_ Sentence Transformers, you can easily use Sentence Transformer models:

.. sidebar:: Documentation

   1. :class:`SentenceTransformer <sentence_transformers.sentence_transformer.model.SentenceTransformer>`
   2. :meth:`SentenceTransformer.encode <sentence_transformers.sentence_transformer.model.SentenceTransformer.encode>`
   3. :meth:`SentenceTransformer.similarity <sentence_transformers.sentence_transformer.model.SentenceTransformer.similarity>`

::

   from sentence_transformers import SentenceTransformer

   # 1. Load a pretrained Sentence Transformer model
   model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

   # The sentences to encode
   sentences = [
       "The weather is lovely today.",
       "It's so sunny outside!",
       "He drove to the stadium.",
   ]

   # 2. Calculate embeddings by calling model.encode()
   embeddings = model.encode(sentences)
   print(embeddings.shape)
   # [3, 384]

   # 3. Calculate the embedding similarities
   similarities = model.similarity(embeddings, embeddings)
   print(similarities)
   # tensor([[1.0000, 0.6660, 0.1046],
   #         [0.6660, 1.0000, 0.1411],
   #         [0.1046, 0.1411, 1.0000]])

Some Sentence Transformer models support inputs beyond text, such as images, audio, or video. You can check which modalities a model supports using the :attr:`~sentence_transformers.sentence_transformer.model.SentenceTransformer.modalities` property and the :meth:`~sentence_transformers.sentence_transformer.model.SentenceTransformer.supports` method. The :meth:`~sentence_transformers.sentence_transformer.model.SentenceTransformer.encode` method accepts different input formats depending on the modality:

- **Text**: strings.
- **Image**: PIL images, file paths, URLs, or numpy/torch arrays.
- **Audio**: file paths, numpy/torch arrays, dicts with ``"array"`` and ``"sampling_rate"`` keys, or ``torchcodec.AudioDecoder`` instances.
- **Video**: file paths, numpy/torch arrays, dicts with ``"array"`` and ``"video_metadata"`` keys, or ``torchcodec.VideoDecoder`` instances.
- **Multimodal dicts**: a dict mapping modality names to values, e.g. ``{"text": ..., "audio": ...}``. The keys must be ``"text"``, ``"image"``, ``"audio"``, or ``"video"``.
- **Chat messages**: a list of dicts with ``"role"`` and ``"content"`` keys for multimodal models that use an uncommon chat template to combine text and non-text inputs.

The following example loads a multimodal model and computes similarities between text and image embeddings:

.. sidebar:: Modality Support

   .. code-block:: python

      from sentence_transformers import SentenceTransformer
   
      model = SentenceTransformer("tomaarsen/Qwen3-VL-Embedding-2B")
   
      # List all supported modalities
      print(model.modalities)
      # ['text', 'image', 'video', 'message']
   
      # Check for a specific modality
      print(model.supports("image"))
      # True
      print(model.supports("audio"))
      # False

.. code-block:: python

   from sentence_transformers import SentenceTransformer

   # 1. Load a model that supports both text and images
   model = SentenceTransformer("tomaarsen/Qwen3-VL-Embedding-2B")

   # 2. Encode images from URLs
   img_embeddings = model.encode([
       "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg",
       "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg",
   ])

   # 3. Encode text queries (one matching + one hard negative per image)
   text_embeddings = model.encode([
       "A green car parked in front of a yellow building",
       "A red car driving on a highway",
       "A bee on a pink flower",
       "A wasp on a wooden table",
   ])

   # 4. Compute cross-modal similarities
   similarities = model.similarity(text_embeddings, img_embeddings)
   print(similarities)
   # tensor([[0.5115, 0.1078],
   #         [0.1999, 0.1108],
   #         [0.1255, 0.6749],
   #         [0.1283, 0.2704]])

.. toctree::
   :maxdepth: 1
   :caption: Tasks and Advanced Usage

   ../../../examples/sentence_transformer/applications/computing-embeddings/README
   semantic_textual_similarity
   ../../../examples/sentence_transformer/applications/semantic-search/README
   ../../../examples/sentence_transformer/applications/retrieve_rerank/README
   ../../../examples/sentence_transformer/applications/clustering/README
   ../../../examples/sentence_transformer/applications/paraphrase-mining/README
   ../../../examples/sentence_transformer/applications/parallel-sentence-mining/README
   ../../../examples/sentence_transformer/applications/image-search/README
   ../../../examples/sentence_transformer/applications/embedding-quantization/README
   custom_models
   mteb_evaluation
   efficiency

