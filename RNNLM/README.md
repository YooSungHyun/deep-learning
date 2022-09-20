# Language Modeling using RNN and Transformer

본 예제는 Pytorch의 Example을 개량하여 작성되었습니다.</br>
https://github.com/pytorch/examples/tree/main/word_language_model
https://arxiv.org/abs/2203.03583

LSTM-LM만 Test 완료되었습니다.</br>
Data는 vocab, txt파일이 필요한데, 해당 파일은 직접 만드셔야 합니다. (example에서도 data소스는 제공되지 않습니다.)</br>
하드코딩 및 shell script는 경로에 맞게 잘 수정해야 할 겁니다. (소스 벌수가 많지 않으니 한번 열어보심도 좋을듯요)</br>

기본적인 구조
1. 데이터는 배치별로 가로가 아닌 세로로 배치됩니다. (그게 batch 효율적이라고 하네요)</br>
   가나다라 는,</br>
   가</br>
   나</br>
   다</br>
   라</br>
   로 배치됩니다.
2. BOS는 사용하지 않습니다.</br>
   STT 모델의 Shallow Fusion Language Model은 어떤 순서에 단어가 나왔을때 해당 단어가 이전 단어를 고려해서 있을만한 확률을 따집니다.</br>
   즉, <EOS>블라블라<EOS>블라블라<EOS>와 같은 형태로 구성해서 학습시켜도 전혀 문제될게 없습니다. (단어의 최초시작은 결국 어떤 문장의 끝 이후임을 의미하므로)
3. Window(stride)는 1입니다. (n-gram만 생각해봐도 window는 1씩 움직입니다.)</br>
   Feature: 나는 밥을</br>
   Label: 밥을 먹었다
4. bptt는 CNN에서의 커널과 비슷하며, n-gram에서의 n을 의미합니다. (몇개의 시퀀스를 고려하겠는가?)</br>
   3번 예제에서 bptt는 2입니다.
5. EOS를 사용하므로 pad는 필요없습니다. EOS를 예측하는 것으로 문장의 끝임을 의미하게됩니다.

예제에서 달라진 점은 논문의 내용을 수용하기 위한 optimizer, scheduler가 추가되었으며,</br>
이미 전처리된 데이터를 data로 활용하기 위한 dictionary 부분과 tokenize 쪽이 수정되었습니다.

```bash
optional arguments:
  -h, --help            show this help message and exit
  --data DATA           location of the data corpus
  --model MODEL         type of network (RNN_TANH, RNN_RELU, LSTM, GRU, Transformer)
  --emsize EMSIZE       size of word embeddings
  --nhid NHID           number of hidden units per layer
  --nlayers NLAYERS     number of layers
  --lr LR               initial learning rate
  --clip CLIP           gradient clipping
  --epochs EPOCHS       upper epoch limit
  --batch_size N        batch size
  --bptt BPTT           sequence length
  --dropout DROPOUT     dropout applied to layers (0 = no dropout)
  --tied                tie the word embedding and softmax weights
  --seed SEED           random seed
  --cuda                use CUDA
  --log-interval N      report interval
  --save SAVE           path to save the final model
  --onnx-export ONNX_EXPORT
                        path to export the final model in onnx format
  --nhead NHEAD         the number of heads in the encoder/decoder of the transformer model
  --dry-run             verify the code and the model
  --vocab_path          Tokenize를 위한 Vocab의 Path
```
