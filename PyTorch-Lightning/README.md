multi_network_tbptt_test <br />
Torch Lightning에서는 tbptt를 지원합니다. 다만 구조가 잘 이해되지 않았습니다. 문의해도 답변도 잘 안달리고요 <br />
평범한 형태의 hidden_state가 풀로 연결되는 seq2seq이나 평범한 lstm형태라면 개발자 가이드만 봐도 쉽게 사용할 수 있습니다. <br />
다만 Joint되는 경우, 역전파해야하는 hidden_state가 2종류가 생기면서 Lightning을 어떻게 써야할지 모호합니다. <br />
그래서 직접 테스트 해본 소스코드를 공개합니다. (적절히 레이어 call 형식을 바꿔가면서 테스트 해볼 수 있습니다.) <br />
실험결과는 해당 discussion을 확인하세요 (https://github.com/Lightning-AI/lightning/discussions/15643) <br />
<br />
참고로 Torch는 직접 CUDA를 고려하여 설치하셔야합니다!