./utils <br />
custom_loss_2_true.py : time-series 예측을 위한 mase 커스텀 소스

***

## custom_loss_2_true.py

사용법 : seasonality가 parameter로 존재하지 않으며, y_true에 값을 2개 넘겨서 사용함 <br />
(batch_size, time-step, [기간평균오차를 계산할 대상, 예측에 대한 실제값]) <br />
해당 아티클 참조 <br />
[https://shyu0522.tistory.com/13] <br />
소스에도 주석 잘 달아놔서 아티클 번갈아가며 확인하면, 사용하는데에는 지장이 없을것으로 보임.
