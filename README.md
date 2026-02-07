# TypeServiceWeb 🤖

**Transformer 모델의 내부 동작을 시각적으로 이해하는 교육용 웹 애플리케이션**

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📋 목차
- [소개](#-소개)
- [주요 기능](#-주요-기능)
- [지원 모델](#-지원-모델)
- [설치 방법](#-설치-방법)
- [사용 방법](#-사용-방법)
- [시각화 예시](#-시각화-예시)
- [프로젝트 구조](#-프로젝트-구조)
- [기술 스택](#-기술-스택)

---

## 🎯 소개

TypeServiceWeb은 GPT-2, T5 등 Transformer 기반 언어 모델의 내부 메커니즘을 **실시간으로 시각화**하는 교육용 도구입니다. 
복잡한 AI 모델의 동작 원리를 직관적으로 이해할 수 있도록 설계되었습니다.

### 왜 만들었나요?
- 🎓 **교육**: Transformer 아키텍처를 배우는 학생들을 위한 실습 도구
- 🔬 **연구**: 모델의 Attention 패턴과 Layer별 특성 분석
- 🧪 **실험**: 다양한 모델 비교 및 토큰화 방식 학습

---

## ✨ 주요 기능

### 1️⃣ 텍스트 생성 및 토큰 분석
- **실시간 텍스트 생성**: Temperature 조절로 창의성 제어
- **토큰 시각화**: 입력/생성 토큰을 색상으로 구분
- **토큰 테이블**: Token ID, 길이, 출처 등 상세 정보

### 2️⃣ Attention Visualization
- **히트맵**: 각 토큰이 다른 토큰에 주목하는 정도 시각화
- **Layer/Head 선택**: 12개 레이어 × 12개 헤드 = 144가지 패턴 탐색
- **인터랙티브**: Plotly 기반으로 확대/축소 가능

### 3️⃣ Layer Evolution
- **Hidden State 추적**: 입력부터 출력까지 레이어별 변화 그래프
- **특징 분석**: 초기/중간/후기 레이어의 역할 이해
- **Norm 시각화**: 각 레이어의 활성화 강도 측정

### 4️⃣ Neuron Pattern Analysis
- **희소 활성화**: 전체 뉴런 중 일부만 활성화되는 패턴 확인
- **Top-K 분석**: 가장 활성화된 뉴런 식별
- **Sparse Coding**: 효율적인 표현 학습 메커니즘

### 5️⃣ T5 Translation Visualization
- **Encoder-Decoder 구조**: 양방향 이해
- **3가지 Attention 메커니즘**:
  - Encoder Self-Attention (입력 문장 분석)
  - Decoder Self-Attention (Masked, 순차 생성)
  - Cross-Attention (입력-출력 정렬)

---

## 🤖 지원 모델

### GPT 계열
- `distilgpt2` (82M params, 6 layers) - 빠른 실험용
- `gpt2` (124M params, 12 layers) - 표준 모델

### T5 계열
- `t5-small` (60M params, 6 encoder + 6 decoder layers)
- Encoder-Decoder 아키텍처 학습용

---

## 🚀 설치 방법

### 1. 저장소 클론
```bash
git clone https://github.com/YOUR_USERNAME/TypeServiceWeb.git
cd TypeServiceWeb
```

### 2. 가상환경 생성 (권장)
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. 의존성 설치
```bash
pip install -r requirements.txt
```

### 4. 실행
```bash
streamlit run app.py
```

브라우저에서 `http://localhost:8501` 자동 열림

---

## 📖 사용 방법

### 기본 워크플로우

#### 1단계: 모델 선택
```
사이드바 → Model Settings
- Category: GPT-2 / T5
- Model: distilgpt2 / gpt2 / t5-small
```

#### 2단계: 생성 설정
```
- 생성할 토큰 수: 1~50
- Temperature: 0.0 (결정적) ~ 2.0 (창의적)
```

#### 3단계: 텍스트 입력
```
예시:
"The future of artificial intelligence"
"Translate English to German: Hello, how are you?"
```

#### 4단계: 분석 시작
```
"분석 시작 🚀" 버튼 클릭
→ 4가지 시각화 자동 생성
```

---

## 📊 시각화 예시

### Attention Heatmap
```
Query Token (행) → Key Token (열)
밝을수록 강한 Attention

패턴 예시:
- 대각선: 자기 자신에 집중
- 마지막 열 밝음: 문장 끝 토큰에 집중
```

### Layer Evolution
```
Y축: Average Norm (활성화 강도)
X축: Layer 번호 (0=입력, 마지막=출력)

해석:
- 초기: 낮은 norm → 단순 패턴
- 중간: 높은 norm → 복잡한 특징
- 후기: 안정화 → 예측 정제
```

---

## 📁 프로젝트 구조

```
TypeServiceWeb/
├── app.py                      # 메인 Streamlit 앱
├── requirements.txt            # 패키지 의존성
├── README.md                   # 프로젝트 문서
│
└── utils/
    ├── __init__.py
    ├── model_loader.py         # 모델 로드 및 생성
    ├── visualization.py        # 시각화 함수
    └── model_info.py           # 모델 메타데이터
```

---

## 🛠 기술 스택

### Core
- **Streamlit** 1.28+: 웹 인터페이스
- **Transformers** 4.35+: 모델 로드 및 추론
- **PyTorch** 2.0+: 딥러닝 프레임워크

### Visualization
- **Plotly** 5.17+: 인터랙티브 그래프
- **Matplotlib** 3.8+: 정적 시각화

### Analysis
- **NumPy** 1.24+: 수치 계산
- **Pandas** 2.0+: 데이터 테이블

---

## 🎓 교육 활용 사례

### 대학 강의
```python
# 수업 시나리오: Attention Mechanism 이해
1. "The cat sat on the mat" 입력
2. Layer 0 vs Layer 11 Attention 비교
3. "the", "cat", "mat" 간의 관계 분석
→ 얕은 층: 위치/문법, 깊은 층: 의미 관계
```

### 연구 실험
```python
# 실험: Temperature가 생성에 미치는 영향
Temperature 0.0: "is a → very important"
Temperature 1.5: "is a → challenging topic"
→ Attention 패턴 변화 관찰
```

---

## 🔮 향후 계획

- [ ] BERT/RoBERTa Masked LM 지원
- [ ] Attention Flow 애니메이션
- [ ] 커스텀 모델 업로드
- [ ] 다국어 UI (English/한국어)
- [ ] PDF 보고서 내보내기

---

## 🤝 기여하기

이슈 제보와 Pull Request를 환영합니다!

---

## 📝 라이선스

MIT License - 자유롭게 사용, 수정, 배포 가능합니다.

---

## 🙏 감사의 말

- **Hugging Face**: Transformers 라이브러리
- **Streamlit**: 빠른 웹 앱 개발
- **DNLAB**: 프로젝트 지원

---

<div align="center">
Made with ❤️ by DNLAB - Deep Neural Network Laboratory
</div>