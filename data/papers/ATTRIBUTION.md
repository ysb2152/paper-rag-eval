# 평가 fixture 출처 표기

이 폴더의 논문 JSON 10편은 **QASPER** 데이터셋에서 발췌한 평가 fixture다.
회귀 테스트와 검색·답변 하네스가 재현 가능하도록 소량을 저장소에 포함한다.

- **데이터셋**: QASPER (Allen Institute for AI)
- **라이선스**: Creative Commons Attribution 4.0 (CC BY 4.0) — https://creativecommons.org/licenses/by/4.0/
- **원 논문**: Pradeep Dasigi, Kyle Lo, Iz Beltagy, Arman Cohan, Noah A. Smith, Matt Gardner.
  "A Dataset of Information-Seeking Questions and Answers Anchored in Research Papers." NAACL 2021.
  arXiv:2105.03011 — https://arxiv.org/abs/2105.03011
- **배포처**: https://huggingface.co/datasets/allenai/qasper

QASPER는 CC-BY 라이선스로 배포된 arXiv 논문만 골라 만든 데이터셋이다.
각 파일 이름은 원 arXiv 논문 ID다:
1705.09665, 1805.02400, 1810.04528, 1811.00942, 1907.05664,
1908.06606, 1909.00694, 1910.14497, 1912.02481, 2003.07723.

**변경 사항**: 원본을 그대로 담지 않고 `scripts/extract_papers.py`로 파싱해
이 프로젝트의 Paper/Paragraph/Question 스키마로 재구성했다.
