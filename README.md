# THANOS-Tree-Augmented-Hierarchical-Attention-Network-with-Operator-Reduce-Parsing-
THANOS is an advancement in HAN (Hierarchical Attention Network) architecture. 
Here we use Tree LSTM to obtain the embeddings for each sentence. It has a hierarchical struture which captures the hierarchical structure of a document.
Two level of attention is applied both at sentence and word level. Natural language exhibits syntactic properties that would naturally combine words to phrases. 
Hence, we experimented by using Tree LSTMs instead of normal LSTMs. Batching is difficult with Tree LSTMs therefore we used SPINN algorithm (Stack Augmented Parser-Interpreter Neural Network).
SPINN model helps in batching with Tree LSTM and provides computational Speed-up. THANOS performnace was comparable with HAN algorithm.

![alt text](https://github.com/surbhardwaj/THANOS-Tree-Augmented-Hierarchical-Attention-Network-with-Operator-Reduce-Parsing-/)
