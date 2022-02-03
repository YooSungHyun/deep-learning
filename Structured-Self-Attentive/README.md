논문 <br />
https://arxiv.org/pdf/1703.03130.pdf <br /> <br />

참고 (GIT) <br />
논문의 메인 구현으로 GNU v3 라이센스에 기반합니다. <br />
https://github.com/ExplorerFreda/Structured-Self-Attentive-Sentence-Embedding <br />
비주얼라이징의 구현으로 MIT 라이센스에 기반합니다. <br />
https://github.com/kaushalshetty/Structured-Self-Attention <br />
 <br />
 <br />
제가 작성한 소스의 상위 라이센스는 GNU v3 라이센스에 기반하므로, 해당 레파지토리의 소스들은 GNU v3 라이센스에 의거한다고 생각하시면 됩니다. <br />
 <br />
데이터의 형식은 전체 데이터가 text, label로 구성된 csv이면 됩니다. (sep=',', encoding='utf-8') <br />
 <br />
가설 검증 및 실험 실무 적용을 위한 단계 중으로, 소스는 하드코딩이 많고, 사용하기 약간 번거로울 수 있습니다. <br />
 <br />
(파라미터는 적절히 변경해가면서 사용해보세요) <br />
data set 준비 -> tokenizer-ysh.py 실행 -> train.py(train.sh) 실행 -> 모델생성 -> predict.py로 동일 파라미터 넣어서 사용.
