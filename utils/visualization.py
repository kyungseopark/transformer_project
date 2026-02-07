"""
시각화 유틸리티
Ecco 출력을 Streamlit에서 표시하기 위한 함수들
"""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import List, Dict, Any

def display_tokens(tokens: List[str], title: str = "Tokens"):
    """
    토큰 목록을 시각화합니다.
    
    Args:
        tokens: 토큰 리스트
        title: 제목
    """
    st.subheader(title)
    
    # 토큰을 색상으로 구분하여 표시
    token_html = ""
    colors = px.colors.qualitative.Pastel
    
    for i, token in enumerate(tokens):
        # 특수 문자 처리
        display_token = token.replace('Ġ', '␣')  # 공백을 ␣로 표시
        display_token = display_token.replace('Ċ', '↵')  # 줄바꿈을 ↵로 표시
        
        color = colors[i % len(colors)]
        token_html += f'<span style="background-color: {color}; padding: 5px; margin: 2px; border-radius: 5px; display: inline-block;" title="원본: {token}">{display_token}</span>'
    
    st.markdown(token_html, unsafe_allow_html=True)
    st.caption(f"총 {len(tokens)}개의 토큰 | ␣ = 공백, ↵ = 줄바꿈")

def plot_attention_heatmap(attention_weights: np.ndarray, tokens: List[str], 
                          layer: int, head: int):
    """
    Attention 가중치를 히트맵으로 시각화합니다.
    
    Args:
        attention_weights: Attention 가중치 배열
        tokens: 토큰 리스트
        layer: 레이어 번호
        head: 헤드 번호
    """
    fig = go.Figure(data=go.Heatmap(
        z=attention_weights,
        x=tokens,
        y=tokens,
        colorscale='Blues',
        text=np.round(attention_weights, 2),
        texttemplate='%{text}',
        textfont={"size": 8},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title=f'Attention Pattern - Layer {layer}, Head {head}',
        xaxis_title='Key',
        yaxis_title='Query',
        height=700,
        autosize=True
    )
    
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True, 'responsive': True})

def plot_token_probabilities(probs: Dict[str, float], top_k: int = 10):
    """
    토큰 확률 분포를 막대 그래프로 표시합니다.
    
    Args:
        probs: {토큰: 확률} 딕셔너리
        top_k: 상위 k개만 표시
    """
    # 확률 순으로 정렬
    sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:top_k]
    tokens, probabilities = zip(*sorted_probs)
    
    fig = go.Figure(data=[
        go.Bar(
            x=list(probabilities),
            y=list(tokens),
            orientation='h',
            marker=dict(
                color=probabilities,
                colorscale='Viridis',
                showscale=True
            ),
            text=[f'{p:.2%}' for p in probabilities],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title=f'Top {top_k} Token Probabilities',
        xaxis_title='Probability',
        yaxis_title='Token',
        height=400,
        yaxis={'categoryorder': 'total ascending'}
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_layer_predictions(layer_predictions: List[Dict[str, Any]], 
                          selected_token: str = None):
    """
    레이어별 예측 변화를 시각화합니다.
    
    Args:
        layer_predictions: 각 레이어의 예측 정보
        selected_token: 추적할 특정 토큰 (선택사항)
    """
    layers = list(range(len(layer_predictions)))
    
    if selected_token:
        # 특정 토큰의 레이어별 순위 변화
        ranks = []
        for pred in layer_predictions:
            token_list = pred.get('tokens', [])
            if selected_token in token_list:
                ranks.append(token_list.index(selected_token) + 1)
            else:
                ranks.append(len(token_list) + 1)
        
        fig = go.Figure(data=[
            go.Scatter(
                x=layers,
                y=ranks,
                mode='lines+markers',
                name=selected_token,
                line=dict(width=3),
                marker=dict(size=10)
            )
        ])
        
        fig.update_layout(
            title=f'Token Rank Evolution: "{selected_token}"',
            xaxis_title='Layer',
            yaxis_title='Rank',
            yaxis={'autorange': 'reversed'},
            height=400
        )
    else:
        # 상위 토큰들의 확률 변화
        fig = go.Figure()
        
        # 여러 토큰 추적 (예: 상위 5개)
        all_tokens = set()
        for pred in layer_predictions:
            all_tokens.update(pred.get('tokens', [])[:5])
        
        for token in list(all_tokens)[:5]:
            probs = []
            for pred in layer_predictions:
                tokens = pred.get('tokens', [])
                probabilities = pred.get('probabilities', [])
                if token in tokens:
                    idx = tokens.index(token)
                    probs.append(probabilities[idx])
                else:
                    probs.append(0)
            
            fig.add_trace(go.Scatter(
                x=layers,
                y=probs,
                mode='lines+markers',
                name=token
            ))
        
        fig.update_layout(
            title='Token Probability Evolution Across Layers',
            xaxis_title='Layer',
            yaxis_title='Probability',
            height=400
        )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_neuron_activations(activations: np.ndarray, tokens: List[str], 
                           top_neurons: int = 50):
    """
    뉴런 활성화 패턴을 히트맵으로 시각화합니다.
    
    Args:
        activations: 뉴런 활성화 값 배열 (tokens x neurons)
        tokens: 토큰 리스트
        top_neurons: 표시할 상위 뉴런 개수
    """
    # 활성화가 높은 상위 뉴런 선택
    neuron_importance = np.sum(np.abs(activations), axis=0)
    top_neuron_indices = np.argsort(neuron_importance)[-top_neurons:]
    
    selected_activations = activations[:, top_neuron_indices]
    
    fig = go.Figure(data=go.Heatmap(
        z=selected_activations.T,
        x=tokens,
        y=[f'Neuron {i}' for i in top_neuron_indices],
        colorscale='RdBu',
        zmid=0
    ))
    
    fig.update_layout(
        title=f'Top {top_neurons} Neuron Activations',
        xaxis_title='Token',
        yaxis_title='Neuron',
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_attribution_scores(tokens: List[str], scores: List[float], 
                           output_token: str):
    """
    Feature attribution 점수를 시각화합니다.
    
    Args:
        tokens: 입력 토큰 리스트
        scores: 각 토큰의 attribution 점수
        output_token: 출력 토큰
    """
    # 정규화
    max_abs_score = max(abs(min(scores)), abs(max(scores)))
    normalized_scores = [s / max_abs_score if max_abs_score > 0 else 0 for s in scores]
    
    # 색상 매핑 (빨강: 음수, 초록: 양수)
    colors = ['rgb(255,{},{})'.format(
        int(255 * (1 - abs(s))), 
        int(255 * (1 - abs(s)))
    ) if s < 0 else 'rgb({},255,{})'.format(
        int(255 * (1 - abs(s))),
        int(255 * (1 - abs(s)))
    ) for s in normalized_scores]
    
    fig = go.Figure(data=[
        go.Bar(
            x=tokens,
            y=scores,
            marker=dict(color=colors),
            text=[f'{s:.3f}' for s in scores],
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title=f'Feature Attribution for Output Token: "{output_token}"',
        xaxis_title='Input Token',
        yaxis_title='Attribution Score',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

def display_comparison_table(data: List[Dict[str, Any]]):
    """
    비교 테이블을 표시합니다.
    
    Args:
        data: 표시할 데이터 리스트
    """
    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True)