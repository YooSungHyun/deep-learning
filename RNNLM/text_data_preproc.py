import re
from jamo import h2j, j2hcj
from collections import deque
from unicodedata import normalize


def trailing(sentence):
    skip_tc_token = normalize("NFC", "*")
    sentence = normalize("NFKD", sentence).upper()
    split_sentence = ""
    for idx, token in enumerate(sentence):
        if idx == 0:
            split_sentence = token
            continue
        if "|".find(sentence) > -1:
            # 띄어쓰기 vocab때문에 "|"가 제대로 전처리 되지 않았음. 여기서 강제로 처리함
            continue
        if re.search(r"[\u1100-\u115E]", token):
            # 초성인경우
            split_sentence = split_sentence + token
        elif re.search(r"[\u1161-\u11A7]", token):
            # 중성(모음)인 경우
            if idx + 1 < len(sentence) and re.search(r"[\u11A8-\u11FF]", sentence[idx + 1]):
                # 텍스트가 붙을 여지가 있고, 뒷자리가 종성이라면 그냥 중성은 붙힌다. (다음 포문에서 종성은 알아서 붙음)
                split_sentence = split_sentence + token
            else:
                # 이외의 경우 모음 다음에 나타날 글자가 없으며, 있더라도 받침이 아니다.
                split_sentence = split_sentence + token + skip_tc_token
            split_sentence = normalize("NFC", split_sentence)
        elif re.search(r"[\u11A8-\u11FF]", token):
            # 종성(받침)일때는 텍스트를 합치지 않음 (완성형으로 붙힘)
            split_sentence = normalize("NFC", split_sentence) + j2hcj(h2j(token))
        elif re.search(r"[A-Z]|[0-9]|'|\s|#|%|°", token):
            # 영어나 숫자, 등장 가능한 특수문자인 경우 완성형으로 붙힘
            split_sentence = split_sentence + normalize("NFC", token)
        else:
            raise Exception("예외 케이스 발생" + sentence)
    split_sentence = normalize("NFC", split_sentence)

    return split_sentence


with open(".txt", "r") as f:
    lines = f.readlines()
lines = deque(lines)
eval_len = 31000
test_len = 31000
train_len = len(lines) - (eval_len + test_len)
with open(".txt", "w") as f:
    while True:
        if len(lines) <= (eval_len + test_len):
            break
        result_line = trailing(lines.popleft())
        f.write(result_line)

with open(".txt", "w") as f:
    while True:
        if len(lines) <= test_len:
            break
        result_line = trailing(lines.popleft())
        f.write(result_line)

with open(".txt", "w") as f:
    while True:
        if len(lines) == 0:
            break
        result_line = trailing(lines.popleft())
        f.write(result_line)
