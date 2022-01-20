### data_preprocess.ipynb


data_preprocess.ipynb를 돌리기 위한 예제파일은 용량이 너무 커서 하단 url을 참고하세요 <br />
[wiki_preprocess_sample_normalize_light_no_enter.txt](https://www.dropbox.com/s/fk1gap9pddbkmdd/wiki_preprocess_sample_normalize_light_no_enter.txt?dl=0) <br />
<br />
data_preprocess.ipynb를 돌리기 위해서는, **konlpy와 mecab, mecab-ko, mecab-ko-dict**이 필요합니다.<br />
**bpe만 실습해보기 위함이라면, 필요 없고**, 형태소를 처리하지 않고 띄어쓰기가 살아있도록 morphs_sents를 구성하면 됩니다.<br />

눈썰미가 좋으시다면, 이미 파악하셨을지 모르겠지만, 해당 소스는 morph 처리를 하고 bpe를 하는 것이, 큰 장점으로 가져가진 못합니다. <br />
<br/>
### create_pretraining_data.ipynb
[Google-BERT](https://github.com/google-research/bert/blob/master/create_pretraining_data.py)에 기반합니다.<br />
제가 봤을때 대비 수정사항이 있나 소스가 조금 달라진 부분이 있을 수 있습니다.<br />
입력으로 들어갈 vocab.list는 각자 생성해주신걸 사용해주시고, input은 Jupyter기에 소스 내에 전부 하드코딩 해야합니다. <br />
<br />
### 출처 <br />
상기의 명시된 소스는, https://tacademy.skplanet.com/live/player/onlineLectureDetail.action?seq=164 강의에서 사용되는 소스를 변형 및 추가하여 작성되었습니다.
<br />
<br />
<br />
### time_schedule_bert <br />
[BERT로 시계열 데이터 분류 Task는 할 수 있을까? (1 - 근무 시간표 예측?)](https://shyu0522.tistory.com/87) <br />
에 대한 소스입니다. <br />
BERT의 모델을 실제로 건들여놨기 때문에, 파일명이 HuggingFace와 동일하다고 해서 무지성으로 사용하시면 안될 수 있습니다. <br />
Data는 개인적인 내용을 담고있어, 제외해놓았으므로, 소스 중간의 주석을 확인하시어, 각자만의 데이터로 한번 구성해보시기 바랍니다. <br />
salt_bert 폴더가 있어야 정상 동작합니다. <br />
실제로 작성한 소스에서 일부를 지워가며 작업하다보니, 중간에 오류나는 포인트가 있을 지 모르겠습니다. 오류가 난다면 issue에 올려주세요. <br />
vocab.list는 직접 만드셔야합니다! <br />
있는 vocab.list로, create_pretraining_data를 진행하고, run_pretraining을 진행 : **create_data_and_pretrain.ipynb** <br />
fine-tuning을 진행합니다 : **fine-tuning.ipynb** <br />
huggingface_from_pretraining의 BertForTimeSeriesClassification 부분이 사실상 핵심이므로, 이 부분만 참고하시는게 더 도움이 되실 분들도 있을지 모르겠습니다. <br />
