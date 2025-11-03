# Eksplorasi-Transformer-Minggu-10

Freddy Harahap - 122140018

Dwi Arthur Revangga - 122140144

## 📋 Deskripsi Proyek

Implementasi ini merupakan eksplorasi mendalam terhadap arsitektur Transformer yang diperkenalkan dalam paper "Attention is All You Need" (Vaswani et al., 2017). Model ini dilatih untuk menerjemahkan kalimat dari bahasa Inggris ke bahasa Rusia menggunakan dataset English-Russian dictionary.

## 🎯 Fitur Utama

- **Arsitektur Transformer Lengkap**: Encoder-Decoder dengan multi-head attention
- **Positional Encoding**: Sinusoidal encoding untuk informasi posisi
- **Custom Vocabulary Builder**: Sistem tokenisasi dan vocabulary management
- **Training dengan Monitoring Per-Batch**: Tracking TrainLoss, TrainAcc, ValLoss, dan ValAcc setiap batch
- **Greedy Decoding**: Implementasi inference untuk translation
- **Visualisasi Training**: Plot loss dan accuracy per batch

## 🏗️ Arsitektur Model

### Komponen Utama

1. **Token Embeddings**
   - Source (English) embedding: 128 dimensi
   - Target (Russian) embedding: 128 dimensi
   - Scaled dengan √(embedding_size)

2. **Positional Encoding**
   - Sinusoidal encoding (sin/cos functions)
   - Formula: PE(pos, 2i) = sin(pos / 10000^(2i/d_model))

3. **Transformer Core**
   - Encoder layers: 1
   - Decoder layers: 1
   - Attention heads: 4
   - Feed-forward hidden dim: 256
   - Dropout: 0.1

4. **Output Generator**
   - Linear layer mapping ke vocabulary size

### Hyperparameters

```python
BATCH_SIZE = 100
NUM_EPOCHS = 1
LEARNING_RATE = 1e-4
EMBEDDING_SIZE = 128
NUM_HEADS = 4
FFN_HIDDEN_DIM = 256
NUM_ENCODER_LAYERS = 1
NUM_DECODER_LAYERS = 1
DROPOUT = 0.1
MAX_LENGTH = 100
MIN_FREQUENCY = 2
```

## 📊 Dataset

**Source**: English-Russian Dictionary from Kaggle
- Dataset: `hijest/englishrussian-dictionary-for-machine-translate`
- Format: Tab-separated values (TSV)
- Split ratio:
  - Training: 80%
  - Validation: 10%
  - Test: 10%

## 🔧 Preprocessing Pipeline

### 1. Text Cleaning
- Normalisasi karakter khusus (no-break space → space)
- Konversi ke lowercase
- Penghapusan punctuation dan angka
- Penghapusan multiple whitespace
- Trimming spasi di awal dan akhir

### 2. Tokenization
- Normalisasi tingkat lanjut
- Pemisahan punctuation dengan spasi
- Splitting berdasarkan whitespace

### 3. Vocabulary Building
- **Special Tokens**: `<pad>`, `<sos>`, `<eos>`, `<unk>`
- Minimum frequency filtering (min_freq = 2)
- Separate vocabularies untuk source dan target

### 4. Data Collation
- Dynamic padding per batch
- Target input: tambahkan `<sos>` di awal
- Target output: tambahkan `<eos>` di akhir
- Padding mask generation
- Transpose ke format (seq_len, batch)

## 🚀 Training Process

### Alur Training Per Batch

1. **Data Loading**: Load batch data dengan masks
2. **Forward Pass**: 
   - Embedding → Positional Encoding → Transformer → Generator
   - Generate logits untuk prediksi
3. **Loss Computation**: CrossEntropyLoss (ignore padding)
4. **Backward Pass**:
   - Backpropagation
   - Gradient clipping (max_norm = 1.0)
   - Optimizer step (Adam)
5. **Metrics Calculation**:
   - Training loss dan accuracy dari batch
   - Validation loss dan accuracy dari seluruh val set
6. **Display**: Print metrics per batch

### Output Format
```
Batch 1/450 | TrainLoss: 5.2341 | TrainAcc: 0.1234 | ValLoss: 5.1234 | ValAcc: 0.1456
Batch 2/450 | TrainLoss: 5.0123 | TrainAcc: 0.1567 | ValLoss: 4.9876 | ValAcc: 0.1678
```

## 🎯 Inference - Translation

### Greedy Decoding Algorithm

1. **Preprocessing**: Tokenisasi dan konversi ke ID
2. **Encoding**: Encode source sentence sekali (efficient)
3. **Autoregressive Generation**:
   - Start dengan `<sos>` token
   - Loop untuk setiap posisi:
     - Generate causal mask
     - Decode dengan current sequence
     - Get logits dari generator
     - Select token dengan probability tertinggi (greedy)
     - Append ke sequence
     - Stop jika generate `<eos>` atau max_len
4. **Post-processing**: Remove special tokens, join menjadi string

### Karakteristik
- ✅ **Cepat**: Deterministic, tidak perlu explore multiple paths
- ✅ **Sederhana**: Easy to implement dan debug
- ❌ **Suboptimal**: Tidak guarantee best translation secara global
- ❌ **No backtracking**: Tidak bisa memperbaiki pilihan sebelumnya

### Alternative Strategies
- **Beam Search**: Maintain top-k candidates untuk hasil lebih baik
- **Sampling**: Random sampling untuk variasi output
- **Temperature Scaling**: Control randomness dalam generation

## 📈 Monitoring dan Visualisasi

### Metrics yang Ditrack
- **Training Loss**: Loss dari batch saat ini
- **Training Accuracy**: Token-level accuracy pada batch
- **Validation Loss**: Average loss pada validation set
- **Validation Accuracy**: Token-level accuracy pada validation set

### Visualisasi
Fungsi `plot_training_logs()` membuat 2 plot:
1. Training & Validation Loss per Batch
2. Training & Validation Accuracy per Batch

## 🛠️ Teknologi yang Digunakan

- **PyTorch**: Deep learning framework
- **NumPy**: Numerical computations
- **Pandas**: Data manipulation
- **Matplotlib**: Visualization
- **KaggleHub**: Dataset download

## 🎓 Konsep Penting

### 1. Attention Mechanism
Transformer menggunakan **self-attention** untuk menangkap dependencies antar token dalam sequence, tanpa memerlukan recurrent connections.

### 2. Positional Encoding
Karena Transformer tidak memiliki urutan bawaan, positional encoding menambahkan informasi posisi ke embeddings menggunakan fungsi sinusoidal.

### 3. Causal Masking
Decoder menggunakan **subsequent mask** untuk mencegah attention melihat token di masa depan (autoregressive property).

### 4. Teacher Forcing
Selama training, decoder menerima ground truth target sebagai input (bukan prediksinya sendiri) untuk mempercepat konvergensi.

### 5. Padding Mask
Boolean mask untuk mengabaikan padding tokens dalam attention computation dan loss calculation.

## 🔍 Analisis dan Improvement

### Kelebihan Implementasi
- ✅ Monitoring detail per-batch
- ✅ Clean code dengan dokumentasi lengkap
- ✅ Modular dan mudah di-extend
- ✅ Efficient data processing

### Area untuk Improvement
- 🔄 Implementasi Beam Search untuk inference lebih baik
- 🔄 Learning rate scheduling (warmup + decay)
- 🔄 Label smoothing untuk regularisasi
- 🔄 Multi-epoch training dengan early stopping
- 🔄 Checkpoint saving dan loading
- 🔄 BLEU score evaluation
- 🔄 Attention visualization
- 🔄 Subword tokenization (BPE/WordPiece)

## 👨‍💻 Penggunaan

### Prerequisites
```bash
pip install torch pandas numpy matplotlib kagglehub
```

### Running the Notebook
1. Buka notebook di Jupyter/Colab
2. Run semua cell secara berurutan
3. Model akan training dan menampilkan metrics per batch
4. Setelah training, akan ditampilkan contoh translation

### Contoh Output Translation
```
English: Hello, how are you?
Russian: привет как дела

English: I love programming and coffee.
Russian: я люблю программирование и кофе
```
