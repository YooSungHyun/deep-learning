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

### 출처 <br />
해당 소스는, https://tacademy.skplanet.com/live/player/onlineLectureDetail.action?seq=164 강의에서 사용되는 소스를 변형 및 추가하여 작성되었습니다.
