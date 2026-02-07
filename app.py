import streamlit as st
import sys
from pathlib import Path

# ecco 폴더를 Python 경로에 추가
sys.path.insert(0, str(Path(__file__).parent / "ecco"))

from utils.model_loader import load_model, get_available_models, get_model_info, safe_generate
from utils.visualization import (
    display_tokens, 
    plot_token_probabilities,
    plot_attention_heatmap,
    plot_layer_predictions
)

# 페이지 설정
st.set_page_config(
    page_title="Transformer Visualizer - DNLAB",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 커스텀 CSS - 모던하고 심플한 디자인
st.markdown("""
<style>
    .main-header {
        color: #1f77b4;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .section-header {
        color: #2c3e50;
        font-size: 1.5rem;
        font-weight: 600;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 0.5rem;
    }
    
    /* Feature 카드 - 심플하고 모던하게 */
    .feature-box {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        padding: 1.5rem;
        margin: 0.5rem 0;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
    }
    .feature-box:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        transform: translateY(-2px);
    }
    .feature-box h4 {
        color: #1f77b4;
        font-size: 1.2rem;
        margin: 0 0 0.8rem 0;
        font-weight: 600;
    }
    .feature-box p {
        color: #555;
        font-size: 0.95rem;
        margin: 0;
        line-height: 1.6;
    }
    .info-box strong {
        font-weight: 600;
        display: block;
        margin-bottom: 0.3rem;
    }
    
    /* 경고 박스 */
    .warning-box {
        background: #bde3fe;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
    .warning-box strong {
        color: #1f77b4;
    }
    /* 버튼 스타일 */
    div.stButton > button {
        background-color: #1f77b4 !important;
        color: white !important;
        border: 1px solid #1f77b4 !important;
    }
    div.stButton > button:hover {
        background-color: #155a8a !important; /* 호버 시 약간 어둡게 */
        border-color: #155a8a !important;
        color: white !important;
    }
    div.stButton > button:active {
        background-color: #0e3d5e !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# 사이드바 - 설정
with st.sidebar:
    # DNLAB 로고
    st.markdown("""
    <div style='text-align: center; padding: 1rem 0;'>
        <div style='font-size: 2.5rem; font-weight: 700; color: #1f77b4;'>DNLAB</div>
        <div style='font-size: 0.9rem; color: #666; margin-top: 0.5rem;'>Data Network Lab</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # 모델 선택
    st.markdown("<h3 style='color: #1f77b4;'>Model Settings</h3>", unsafe_allow_html=True)
    
    available_models = get_available_models()
    model_category = st.selectbox("Category", list(available_models.keys()), key="model_cat")
    model_name = st.selectbox("Model", available_models[model_category], key="model_name")
    
    # 모델 정보
    with st.expander("Model Info"):
        info = get_model_info(model_name)
        st.caption(f"**Parameters:** {info['params']}")
        st.caption(f"**Layers:** {info['layers']}")
        st.caption(f"**Hidden Size:** {info['hidden_size']}")
    
    # 캐시 클리어 버튼
    if st.button("🔄 모델 재로드", help="캐시를 클리어하고 모델을 다시 로드합니다"):
        st.cache_resource.clear()
        st.success("✅ 캐시 클리어 완료! 페이지를 새로고침하세요.")
        st.rerun()
    
    st.markdown("---")
    
    # 생성 옵션
    st.markdown("<h3 style='color: #1f77b4;'>Generation Settings</h3>", unsafe_allow_html=True)
    
    max_tokens = st.slider("생성할 토큰 수", 1, 50, 10, key="max_tokens",
        help="입력 뒤에 생성할 토큰의 개수")
    temperature = st.slider("Temperature", 0.0, 2.0, 1.0, 0.1, key="temperature",
        help="높을수록 다양하고 창의적 | 낮을수록 일관되고 보수적")
    
    # Temperature 설명 박스 (구분선 제거)
    st.markdown("""
    <div style='background: #1f77b4; padding: 0.8rem; border-radius: 5px; font-size: 0.85rem; color: #ffffff; margin-top: 0.5rem;'>
        <strong>Temperature</strong><br>
        0.0: 가장 확률 높은 토큰만 선택<br>
        1.0: 적절한 다양성<br>
        2.0: 매우 창의적/예측불가
    </div>
    """, unsafe_allow_html=True)
    
    # 시각화 옵션
    st.markdown("---")
    st.markdown("<h3 style='color: #1f77b4;'>Visualization</h3>", unsafe_allow_html=True)
    
    show_tokens = st.checkbox("Token Analysis", value=True)
    show_attention = st.checkbox("Attention Heatmap", value=True)
    show_neurons = st.checkbox("Neuron Patterns", value=False)
    show_evolution = st.checkbox("Layer Evolution", value=False)
    
    if show_attention:
        st.markdown("**Attention Settings**")
        selected_layer = st.slider("Layer", 0, 11, 0, key="layer",
            help="어느 레이어의 attention을 볼지 선택 (0=초기, 11=마지막)")
        selected_head = st.slider("Head", 0, 11, 0, key="head",
            help="Multi-head 중 어느 head를 볼지 선택")
        
        st.markdown("""
        <div style='background: #1f77b4; padding: 0.8rem; border-radius: 5px; font-size: 0.85rem; color: #ffffff; margin-top: 0.5rem;'>
            <strong>Layer & Head 가이드</strong><br><br>
            <strong>Layer 선택:</strong><br>
            • Layer 0-2: 위치/구조 (균등한 분포)<br>
            • Layer 3-5: 문법/의미 (핵심 단어 집중)<br>
            • Layer 6+: 고급 의미 (목적어 집중)<br>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #888; font-size: 0.85rem;'>
        Built with Ecco & Streamlit<br>
        <span style='color: #1f77b4;'>© 2026 DNLAB</span>
    </div>
    """, unsafe_allow_html=True)

# 메인 페이지
st.markdown("<h1 class='main-header'>Transformer Visualizer</h1>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>트랜스포머 모델의 내부 동작을 실시간으로 시각화하고 분석합니다.</div>", unsafe_allow_html=True)

# Core Features - 모던한 디자인
st.markdown("<h2 class='section-header'>Core Features</h2>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class='feature-box'>
        <h4>1. Token Analysis</h4>
        <p>입력/출력 텍스트의 토큰화 비교<br>BPE/WordPiece 방식</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class='feature-box'>
        <h4>3. Neuron Pattern Analysis</h4>
        <p>FFNN 뉴런 활성화 패턴<br>Hidden States 시각화</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class='feature-box'>
        <h4>2. Attention Visualization</h4>
        <p>Multi-Head Attention 가중치<br>토큰 간 의존성 분석</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class='feature-box'>
        <h4>4. Layer Evolution</h4>
        <p>레이어별 표현 변화 추적<br>Prediction 변화</p>
    </div>
    """, unsafe_allow_html=True)

# 통합 분석 인터페이스
st.markdown("<h2 class='section-header'>Analyze Text</h2>", unsafe_allow_html=True)

# 입력 영역
col1, col2 = st.columns([2, 1])

with col1:
    example_prompts = {
        "직접 입력": "",
        "간단한 예시": "The cat sat on the",
        "복잡한 예시": "The bank can refuse to lend money",
    }
    
    selected_example = st.selectbox("예제", list(example_prompts.keys()))
    input_text = st.text_area(
        "분석할 텍스트를 입력하세요:",
        value=example_prompts[selected_example],
        height=100,
        placeholder="예: The Eiffel Tower is located in Paris."
    )

with col2:
    st.markdown("""
    <div style='background-color: #f8f9fa; padding: 1.2rem; border-radius: 8px; border: 1px solid #e0e0e0; height: 100%;'>
        <strong style='color: #1f77b4;'>이해하기</strong><br><br>
        <strong>입력→출력:</strong> 모델이 입력을 보고 다음 단어를 예측합니다<br><br>
    </div>
    """, unsafe_allow_html=True)

# 한국어 처리 안내
with st.expander("🇰🇷 한국어 입력 시 주의사항 (클릭하여 확인)"):
    st.markdown("""
    **현재 문제:** GPT-2/BERT는 영어 중심 모델입니다.
    한국어는 byte 단위로 분해되어 읽을 수 없는 문자로 표시됩니다.
    
    **예시:**
    - "안녕하세요" → `ìķĪ`, `ë`, `Ĥ`, `ķ`, `í`, `ķĺ`, `...` (10개 이상의 토큰)
    - "Hello" → `Hello` (1개 토큰)
    """)

# 분석 버튼
if st.button("분석 시작", type="primary", use_container_width=True):
    if not input_text:
        st.warning("텍스트를 입력해주세요!")
    else:
        with st.spinner(f"{model_name} 로딩 중..."):
            lm = load_model(model_name)
        
        if lm is None:
            st.error("모델 로드 실패")
        else:
            # T5 번역 모델 처리
            if model_name.startswith('t5'):
                try:
                    st.success("번역 준비 완료!")
                    st.markdown("---")
                    
                    # 번역 작업 선택
                    st.markdown("<h2 class='section-header'>Translation (번역)</h2>", unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        task = st.selectbox("번역 방향", [
                            "translate English to German",
                            "translate English to French",
                            "translate English to Romanian",
                        ])
                    
                    with col2:
                        st.info("""
                        **T5 모델**
                        - Encoder-Decoder 구조
                        - "Attention is All You Need" 논문과 동일
                        - 다양한 언어 번역 가능
                        """)
                    
                    # 번역 실행
                    with st.spinner("번역 중..."):
                        input_text_with_task = f"{task}: {input_text}"
                        input_ids = lm.tokenizer.encode(input_text_with_task, return_tensors="pt")
                        
                        outputs = lm.model.generate(
                            input_ids,
                            max_length=128,
                            num_beams=4,
                            early_stopping=True
                        )
                        
                        translated_text = lm.tokenizer.decode(outputs[0], skip_special_tokens=True)
                        
                        # 토큰 정보
                        input_tokens = lm.tokenizer.convert_ids_to_tokens(input_ids[0])
                        output_tokens = lm.tokenizer.convert_ids_to_tokens(outputs[0])
                    
                    # 결과 표시
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**입력 (Source)**")
                        st.code(input_text, language=None)
                        st.caption(f"토큰 수: {len(input_tokens)}")
                        
                        with st.expander("입력 토큰"):
                            display_tokens(input_tokens, "Input Tokens")
                    
                    with col2:
                        st.markdown("**출력 (Translation)**")
                        st.code(translated_text, language=None)
                        st.caption(f"토큰 수: {len(output_tokens)}")
                        
                        with st.expander("출력 토큰"):
                            display_tokens(output_tokens, "Output Tokens")
                    
                    # Encoder-Decoder 구조 설명
                    st.markdown("---")
                    st.markdown("<h3>Encoder-Decoder Architecture</h3>", unsafe_allow_html=True)
                    
                    st.markdown("""
                    ```
                    입력 (영어)
                        ↓
                    [ENCODER] ← Self-Attention (양방향)
                        ↓
                    Context Vector (의미 압축)
                        ↓
                    [DECODER] ← Masked Self-Attention
                        ↓      ← Cross-Attention (Encoder 참조)
                    출력 (독일어)
                    ```
                    
                    **핵심:**
                    - Encoder: 입력 문장을 이해 (양방향 Attention)
                    - Decoder: 출력 문장을 생성 (순차적)
                    - Cross-Attention: Decoder가 Encoder의 정보를 참조
                    
                    → "Attention is All You Need" 논문의 Original Transformer!
                    """)
                    
                    # T5 분석 기능 추가
                    st.markdown("---")
                    st.markdown("<h2 class='section-header'>T5 Model Analysis</h2>", unsafe_allow_html=True)
                    
                    # 순수 입력만으로 재처리 (task 프롬프트 제외)
                    pure_input_ids = lm.tokenizer.encode(input_text, return_tensors="pt")
                    pure_input_tokens = lm.tokenizer.convert_ids_to_tokens(pure_input_ids[0])
                    
                    # Encoder Attention
                    if show_attention:
                        st.markdown("### Encoder Self-Attention")
                        st.caption("입력 문장 내 토큰 간 관계")
                        
                        try:
                            # Encoder outputs (순수 입력만)
                            encoder_outputs = lm.model.encoder(
                                input_ids=pure_input_ids,
                                output_attentions=True,
                                return_dict=True
                            )
                            
                            if encoder_outputs.attentions:
                                encoder_attentions = encoder_outputs.attentions
                                max_layer = len(encoder_attentions) - 1
                                max_head = encoder_attentions[0].shape[1] - 1
                                
                                layer = min(selected_layer, max_layer)
                                head = min(selected_head, max_head)
                                
                                attention_weights = encoder_attentions[layer][0, head].detach().cpu().numpy()
                                plot_attention_heatmap(attention_weights, pure_input_tokens, layer, head)
                                
                                st.info(f"""
                                📍 **Encoder Layer {layer}/{max_layer}, Head {head}/{max_head}**
                                
                                T5-small 모델 구조:
                                - 총 **{max_layer + 1}개 레이어** (0부터 시작)
                                - 각 레이어마다 **{max_head + 1}개 Head**
                                - 현재 Layer {layer}, Head {head} 시각화 중
                                """)
                            else:
                                st.warning("Encoder Attention을 가져올 수 없습니다.")
                        except Exception as e:
                            st.error(f"Encoder Attention 오류: {str(e)}")
                            import traceback
                            st.code(traceback.format_exc())
                        
                        st.markdown("---")
                        
                        # Decoder Self-Attention
                        st.markdown("### Decoder Self-Attention")
                        st.caption("출력 문장 내 토큰 간 관계 (순차적)")
                        
                        try:
                            # Full model with attention
                            full_outputs = lm.model(
                                input_ids=input_ids,
                                decoder_input_ids=outputs[:, :-1],
                                output_attentions=True,
                                return_dict=True
                            )
                            
                            if full_outputs.decoder_attentions:
                                decoder_attentions = full_outputs.decoder_attentions
                                max_layer = len(decoder_attentions) - 1
                                max_head = decoder_attentions[0].shape[1] - 1
                                
                                layer = min(selected_layer, max_layer)
                                head = min(selected_head, max_head)
                                
                                attention_weights = decoder_attentions[layer][0, head].detach().cpu().numpy()
                                plot_attention_heatmap(attention_weights, output_tokens[:-1], layer, head)
                                
                                st.info(f"📍 Decoder Layer {layer}/{max_layer}, Head {head}/{max_head}")
                        except Exception as e:
                            st.error(f"Decoder Attention 오류: {str(e)}")
                        
                        st.markdown("---")
                        
                        # Cross-Attention
                        st.markdown("### Cross-Attention (Decoder → Encoder)")
                        st.caption("출력 토큰이 입력 토큰을 참조하는 관계")
                        
                        try:
                            import numpy as np
                            
                            if full_outputs.cross_attentions:
                                cross_attentions = full_outputs.cross_attentions
                                max_layer = len(cross_attentions) - 1
                                max_head = cross_attentions[0].shape[1] - 1
                                
                                layer = min(selected_layer, max_layer)
                                head = min(selected_head, max_head)
                                
                                # Cross attention: [decoder_len, encoder_len]
                                cross_attn = cross_attentions[layer][0, head].detach().cpu().numpy()
                                
                                import plotly.graph_objects as go
                                fig = go.Figure(data=go.Heatmap(
                                    z=cross_attn,
                                    x=input_tokens,
                                    y=output_tokens[:-1] if len(output_tokens) > 1 else output_tokens,
                                    colorscale='Reds',
                                    text=np.round(cross_attn, 2),
                                    texttemplate='%{text}',
                                    textfont={"size": 8},
                                ))
                                
                                fig.update_layout(
                                    title=f'Cross-Attention - Layer {layer}, Head {head}',
                                    xaxis_title='Input (Encoder)',
                                    yaxis_title='Output (Decoder)',
                                    height=700,
                                    autosize=True
                                )
                                
                                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True, 'responsive': True})
                                st.info(f"📍 Cross-Attention Layer {layer}/{max_layer}, Head {head}/{max_head}")
                                
                                st.markdown("""
                                **Cross-Attention 해석:**
                                - 각 출력 토큰이 어느 입력 토큰을 주목하는지 표시
                                - 밝은 색 = 강한 연결 (출력이 입력을 많이 참조)
                                - 예: "Katze" → "cat" (강한 연결)
                                """)
                            else:
                                st.warning("Cross-Attention 정보를 가져올 수 없습니다.")
                        except Exception as e:
                            st.error(f"Cross-Attention 오류: {str(e)}")
                            import traceback
                            st.code(traceback.format_exc())
                    
                    # Encoder/Decoder Hidden States
                    if show_neurons or show_evolution:
                        st.markdown("---")
                        st.markdown("### Encoder vs Decoder Hidden States")
                        
                        try:
                            # Get hidden states
                            encoder_outputs = lm.model.encoder(
                                input_ids=input_ids,
                                output_hidden_states=True,
                                return_dict=True
                            )
                            
                            full_outputs = lm.model(
                                input_ids=input_ids,
                                decoder_input_ids=outputs[:, :-1],
                                output_hidden_states=True,
                                return_dict=True
                            )
                            
                            encoder_hidden = encoder_outputs.hidden_states
                            decoder_hidden = full_outputs.decoder_hidden_states
                            
                            # Layer norms
                            import plotly.graph_objects as go
                            
                            encoder_norms = [h[0].norm(dim=-1).mean().item() for h in encoder_hidden]
                            decoder_norms = [h[0].norm(dim=-1).mean().item() for h in decoder_hidden]
                            
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=list(range(len(encoder_norms))),
                                y=encoder_norms,
                                mode='lines+markers',
                                name='Encoder',
                                line=dict(color='blue', width=3)
                            ))
                            fig.add_trace(go.Scatter(
                                x=list(range(len(decoder_norms))),
                                y=decoder_norms,
                                mode='lines+markers',
                                name='Decoder',
                                line=dict(color='red', width=3)
                            ))
                            
                            fig.update_layout(
                                title="Encoder vs Decoder: Hidden State Magnitude",
                                xaxis_title="Layer",
                                yaxis_title="Average Norm",
                                height=400
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            st.markdown("""
                            **해석:**
                            - **Encoder (파란색)**: 입력 문장 이해
                            - **Decoder (빨간색)**: 출력 문장 생성
                            - 초기 레이어: 단순 패턴
                            - 후기 레이어: 복잡한 의미
                            """)
                            
                        except Exception as e:
                            st.error(f"Hidden States 분석 오류: {str(e)}")
                    
                except Exception as e:
                    st.error(f"번역 오류: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
            
            # GPT 텍스트 생성
            else:
                # 텍스트 생성
                with st.spinner("텍스트 생성 중..."):
                    result = safe_generate(
                        lm,
                        input_text,
                        max_tokens=max_tokens,
                        temperature=temperature
                    )
                    
                    generated_text = result['generation']
                    all_tokens = result['tokens']
                
                # 입력 토큰 계산
                input_ids = lm.tokenizer.encode(input_text, add_special_tokens=False)
                n_input = len(input_ids)
                n_generated = len(all_tokens) - n_input
                
                st.success("분석 완료!")
                st.markdown("---")
                
                # 생성 결과 표시
                st.markdown("<h2 class='section-header'>생성 결과</h2>", unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**입력 (Input)**")
                    st.code(input_text, language=None)
                    st.metric("입력 토큰 수", n_input)
                
                with col2:
                    st.markdown("**전체 출력 (Output)**")
                    st.code(generated_text, language=None)
                    st.metric("총 토큰 수", len(all_tokens), f"+{n_generated} 생성됨")
                
                # 생성된 부분만 표시
                if n_generated > 0:
                    generated_only = generated_text[len(input_text):].strip()
                    st.markdown("**새로 생성된 부분**")
                    st.success(generated_only if generated_only else "[생성 없음]")
                
                # 1. Token Analysis
                if show_tokens:
                    st.markdown("---")
                    st.markdown("<h2 class='section-header'>Token Analysis</h2>", unsafe_allow_html=True)
                    
                    # 3개 탭: 전체/입력/생성
                    tab1, tab2, tab3 = st.tabs(["전체 토큰", "입력 토큰", "생성 토큰"])
                    
                    with tab1:
                        st.caption("입력 + 생성된 모든 토큰")
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            display_tokens(all_tokens, "All Tokens")
                        
                        with col2:
                            st.metric("Total", len(all_tokens))
                            st.metric("Input", n_input)
                            st.metric("Generated", n_generated)
                    
                    with tab2:
                        st.caption("입력 텍스트의 토큰만")
                        input_tokens = all_tokens[:n_input]
                        
                        col1, col2 = st.columns([2, 1])
                        with col1:
                            display_tokens(input_tokens, "Input Tokens")
                        with col2:
                            st.metric("Tokens", n_input)
                            st.metric("Characters", len(input_text))
                            if n_input > 0:
                                st.metric("Avg Length", f"{len(input_text)/n_input:.2f}")
                    
                    with tab3:
                        st.caption("새로 생성된 토큰만")
                        if n_generated > 0:
                            generated_tokens = all_tokens[n_input:]
                            
                            col1, col2 = st.columns([2, 1])
                            with col1:
                                display_tokens(generated_tokens, "Generated Tokens")
                            with col2:
                                st.metric("Tokens", n_generated)
                                generated_chars = len(generated_text) - len(input_text)
                                st.metric("Characters", generated_chars)
                                if n_generated > 0:
                                    st.metric("Avg Length", f"{generated_chars/n_generated:.2f}")
                        else:
                            st.info("생성된 토큰이 없습니다.")
                    
                    # 토큰 길이 분포
                    st.markdown("---")
                    st.subheader("Token Length Distribution")
                    
                    import plotly.graph_objects as go
                    token_lengths = [len(token.replace('Ġ', '').replace('##', '')) for token in all_tokens]
                    
                    fig = go.Figure(data=[
                        go.Bar(
                            x=list(range(len(all_tokens))),
                            y=token_lengths,
                            marker=dict(
                                color=token_lengths,
                                colorscale='Blues',
                                showscale=True,
                                colorbar=dict(title="Length")
                            ),
                            text=[f"{t}<br>{l}" for t, l in zip(all_tokens, token_lengths)],
                            hovertemplate='Token: %{text}<extra></extra>'
                        )
                    ])
                    
                    # 입력/생성 경계선
                    if n_generated > 0:
                        fig.add_vline(
                            x=n_input - 0.5,
                            line_dash="dash",
                            line_color="red",
                            annotation_text="생성 시작",
                            annotation_position="top"
                        )
                    
                    fig.update_layout(
                        title="Token Length by Position",
                        xaxis_title="Token Position",
                        yaxis_title="Character Length",
                        height=300,
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 토큰 상세 테이블
                    with st.expander("Token Details Table"):
                        import pandas as pd
                        
                        # Token IDs 가져오기
                        token_ids = lm.tokenizer.encode(generated_text, add_special_tokens=False)
                        
                        token_data = []
                        for i, (token, token_id) in enumerate(zip(all_tokens, token_ids)):
                            display_token = token.replace('Ġ', '␣').replace('##', '#')
                            source = "Input" if i < n_input else "Generated"
                            
                            token_data.append({
                                "Position": i,
                                "Token": display_token,
                                "Token ID": token_id,
                                "Length": len(token.replace('Ġ', '').replace('##', '')),
                                "Source": source
                            })
                        df = pd.DataFrame(token_data)
                        st.dataframe(df, use_container_width=True, height=300)
                        
                        st.caption(f"💡 Vocabulary size: {len(lm.tokenizer):,} tokens")
                
                # 2. Attention Visualization
                if show_attention:
                    st.markdown("---")
                    st.markdown("<h2 class='section-header'>Attention Visualization</h2>", unsafe_allow_html=True)
                    
                    try:
                        inputs_dict = lm.tokenizer(generated_text, return_tensors="pt")
                        outputs = lm.model(**inputs_dict, output_attentions=True)
                        
                        # Attention이 None인지 확인
                        if outputs.attentions is None or len(outputs.attentions) == 0:
                            st.warning("⚠️ 이 모델은 Attention 출력을 지원하지 않습니다.")
                        else:
                            attentions = outputs.attentions
                            
                            # Layer와 Head 범위 확인
                            max_layer = len(attentions) - 1
                            if selected_layer > max_layer:
                                layer = max_layer
                            else:
                                layer = selected_layer
                            
                            attention_tensor = attentions[layer]
                            
                            # attention_tensor가 None인지 체크
                            if attention_tensor is None:
                                st.error(f"❌ Layer {layer}의 Attention을 가져올 수 없습니다.")
                            else:
                                max_head = attention_tensor.shape[1] - 1
                                if selected_head > max_head:
                                    head = max_head
                                else:
                                    head = selected_head
                                
                                attention_weights = attention_tensor[0, head].detach().cpu().numpy()
                                
                                plot_attention_heatmap(attention_weights, all_tokens, layer, head)
                                
                                st.caption(f"📍 Layer {layer}/{max_layer}, Head {head}/{max_head}")
                            
                    except Exception as e:
                        st.error(f"Attention 시각화 오류: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
                
                # 3. Neuron Pattern Analysis
                if show_neurons:
                    st.markdown("---")
                    st.markdown("<h2 class='section-header'>Neuron Pattern Analysis</h2>", unsafe_allow_html=True)
                    
                    try:
                        inputs_dict = lm.tokenizer(generated_text, return_tensors="pt")
                        outputs = lm.model(**inputs_dict, output_hidden_states=True)
                        hidden_states = outputs.hidden_states
                        
                        layer_idx = len(hidden_states) // 2
                        activations = hidden_states[layer_idx][0].detach().cpu().numpy()
                        
                        from utils.visualization import plot_neuron_activations
                        plot_neuron_activations(activations, all_tokens, top_neurons=30)
                        
                        st.info(f"Layer {layer_idx} (중간 레이어)의 뉴런 활성화 패턴")
                    except Exception as e:
                        st.error(f"뉴런 분석 오류: {str(e)}")
                
                # 4. Layer Evolution
                if show_evolution:
                    st.markdown("---")
                    st.markdown("<h2 class='section-header'>Layer Evolution</h2>", unsafe_allow_html=True)
                    
                    try:
                        inputs_dict = lm.tokenizer(generated_text, return_tensors="pt")
                        outputs = lm.model(**inputs_dict, output_hidden_states=True)
                        
                        if outputs.hidden_states is None:
                            st.warning("⚠️ 이 모델은 Hidden States 출력을 지원하지 않습니다.")
                        else:
                            hidden_states = outputs.hidden_states
                            
                            # 레이어별 norm 계산
                            layer_norms = []
                            for layer_idx, hidden_state in enumerate(hidden_states):
                                # [batch, seq_len, hidden_dim] → 평균 norm
                                norm = hidden_state[0].norm(dim=-1).mean().item()
                                layer_norms.append(norm)
                            
                            # 시각화
                            import plotly.graph_objects as go
                            
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=list(range(len(layer_norms))),
                                y=layer_norms,
                                mode='lines+markers',
                                name='Layer Norm',
                                line=dict(color='#1f77b4', width=3),
                                marker=dict(size=8)
                            ))
                            
                            fig.update_layout(
                                title="Layer Evolution: Hidden State Magnitude",
                                xaxis_title="Layer",
                                yaxis_title="Average Norm",
                                height=400,
                                hovermode='x unified'
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # 동적으로 레이어 구간 계산
                            num_layers = len(hidden_states) - 1  # input embedding 제외
                            third = num_layers // 3
                            
                            st.markdown(f"""
                            **해석:**
                            - **초기 레이어 (0-{third})**: 단순한 패턴 학습 (낮은 norm)
                            - **중간 레이어 ({third+1}-{third*2})**: 복잡한 특징 추출 (높은 norm)
                            - **후기 레이어 ({third*2+1}-{num_layers})**: 예측을 위한 정제 (안정화)
                            
                            **Norm이 높다 = 해당 레이어가 활발하게 작동**
                            """)
                            
                    except Exception as e:
                        st.error(f"Layer Evolution 오류: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; padding: 1rem 0;'>
    <div style='margin-bottom: 0.5rem;'>
        Powered by <a href='https://huggingface.co/transformers' target='_blank' style='color: #1f77b4; text-decoration: none;'>Hugging Face Transformers</a>
    </div>
    <div style='font-size: 0.9rem;'>DNLAB - Data Network Laboratory</div>
</div>
""", unsafe_allow_html=True)