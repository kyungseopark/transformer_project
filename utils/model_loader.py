"""
모델 로더 유틸리티
Ecco를 사용하여 다양한 트랜스포머 모델을 로드합니다.
"""
import streamlit as st
import torch

def translate_text(model, tokenizer, text, task="translate English to German"):
    """
    T5 모델로 텍스트 번역
    
    Args:
        model: T5 모델
        tokenizer: T5 토크나이저
        text: 번역할 텍스트
        task: 번역 작업 (예: "translate English to German")
    
    Returns:
        dict: 번역 결과와 토큰 정보
    """
    # 입력 준비
    input_text = f"{task}: {text}"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    
    # 번역 생성
    outputs = model.generate(
        input_ids,
        max_length=128,
        num_beams=4,
        early_stopping=True
    )
    
    # 디코딩
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 토큰 정보
    input_tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    output_tokens = tokenizer.convert_ids_to_tokens(outputs[0])
    
    return {
        'translation': translated_text,
        'input_tokens': input_tokens,
        'output_tokens': output_tokens,
        'input_ids': input_ids[0].tolist(),
        'output_ids': outputs[0].tolist()
    }


@st.cache_resource(ttl=3600, show_spinner=False)
def load_model(model_name: str, _version: int = 3):
    """
    모델을 로드하고 캐시합니다.
    
    Args:
        model_name: Hugging Face 모델 이름 (예: 'gpt2', 'bert-base-uncased')
        _version: 캐시 버전 (변경 시 캐시 무효화)
    
    Returns:
        모델 객체
    """
    try:
        # T5 번역 모델
        if model_name.startswith('t5'):
            from transformers import T5ForConditionalGeneration, T5Tokenizer
            
            model = T5ForConditionalGeneration.from_pretrained(model_name)
            tokenizer = T5Tokenizer.from_pretrained(model_name)
            
            class T5Wrapper:
                def __init__(self, model, tokenizer):
                    self.model = model
                    self.tokenizer = tokenizer
            
            return T5Wrapper(model, tokenizer)
        
        # BERT/RoBERTa Masked LM 모델
        elif 'bert' in model_name.lower() or 'roberta' in model_name.lower():
            from transformers import AutoModelForMaskedLM, AutoTokenizer
            
            model = AutoModelForMaskedLM.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            class MaskedLMWrapper:
                def __init__(self, model, tokenizer):
                    self.model = model
                    self.tokenizer = tokenizer
                    self.model_type = 'masked_lm'
            
            return MaskedLMWrapper(model, tokenizer)
        
        # GPT 모델 (직접 Transformers 사용 - Ecco 문제 해결)
        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                output_attentions=True,
                output_hidden_states=True
            )
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # pad_token 설정
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            class GPTWrapper:
                def __init__(self, model, tokenizer):
                    self.model = model
                    self.tokenizer = tokenizer
                    self.model_name = model_name
            
            return GPTWrapper(model, tokenizer)
            
    except ImportError as e:
        st.error(f"필요한 라이브러리를 찾을 수 없습니다: {str(e)}")
        return None
    except Exception as e:
        st.error(f"모델 로드 실패: {str(e)}")
        st.info("💡 해결 방법:\n1. 인터넷 연결 확인\n2. 모델이 처음 다운로드되는 경우 시간이 걸릴 수 있습니다\n3. 작은 모델(distilgpt2)로 먼저 시도해보세요")
        return None

def safe_generate(lm, input_text, max_tokens=10, temperature=1.0):
    """
    안전한 텍스트 생성 (Transformers 직접 사용)
    
    Args:
        lm: 모델 래퍼 객체
        input_text: 입력 텍스트
        max_tokens: 생성할 최대 토큰 수
        temperature: 온도 (다양성 조절)
    
    Returns:
        dict: {'generation': 생성된 텍스트, 'tokens': 토큰 리스트}
    """
    try:
        inputs = lm.tokenizer(input_text, return_tensors="pt")
        
        # generation_config 설정
        generation_config = {
            'max_new_tokens': max_tokens,
            'temperature': temperature,
            'do_sample': True if temperature > 0 else False,
            'pad_token_id': lm.tokenizer.pad_token_id,
            'eos_token_id': lm.tokenizer.eos_token_id,
        }
        
        outputs = lm.model.generate(**inputs, **generation_config)
        generated_text = lm.tokenizer.decode(outputs[0], skip_special_tokens=False)
        
        # 토큰 추출 (전체 시퀀스)
        tokens = lm.tokenizer.convert_ids_to_tokens(outputs[0])
        
        return {
            'generation': generated_text,
            'tokens': tokens
        }
    except Exception as e:
        raise Exception(f"텍스트 생성 실패: {str(e)}")

def get_available_models():
    """
    사용 가능한 모델 목록을 반환합니다.
    
    Returns:
        dict: 카테고리별 모델 목록
    """
    return {
        "GPT Models (생성)": [
            "distilgpt2",      # 82M - 빠른 테스트용 ⚡
            "gpt2",            # 117M - 일반 사용 추천 ⭐
            "gpt2-medium",     # 345M - 고급 분석용 (느림)
        ],
        "Translation Models (번역)": [
            "t5-small",  # 60M - 영어 번역
        ],
    }

def get_model_info(model_name: str):
    """
    모델에 대한 정보를 반환합니다.
    
    Args:
        model_name: 모델 이름
    
    Returns:
        dict: 모델 정보
    """
    model_info = {
        "distilgpt2": {
            "params": "82M",
            "layers": 6,
            "hidden_size": 768,
            "description": "가벼운 GPT-2 (빠른 테스트용)"
        },
        "gpt2": {
            "params": "117M",
            "layers": 12,
            "hidden_size": 768,
            "description": "OpenAI의 기본 GPT-2 모델"
        },
        "gpt2-medium": {
            "params": "345M",
            "layers": 24,
            "hidden_size": 1024,
            "description": "중간 크기의 GPT-2 모델"
        },
        "gpt2-large": {
            "params": "774M",
            "layers": 36,
            "hidden_size": 1280,
            "description": "큰 크기의 GPT-2 모델"
        },
        "gpt2-xl": {
            "params": "1.5B",
            "layers": 48,
            "hidden_size": 1600,
            "description": "매우 큰 GPT-2 (GPU 권장)"
        },
        "distilbert-base-uncased": {
            "params": "66M",
            "layers": 6,
            "hidden_size": 768,
            "description": "가벼운 BERT (빠른 테스트용)"
        },
        "bert-base-uncased": {
            "params": "110M",
            "layers": 12,
            "hidden_size": 768,
            "description": "Google BERT 기본 모델 (소문자)"
        },
        "bert-large-uncased": {
            "params": "340M",
            "layers": 24,
            "hidden_size": 1024,
            "description": "큰 BERT 모델"
        },
        "distilroberta-base": {
            "params": "82M",
            "layers": 6,
            "hidden_size": 768,
            "description": "가벼운 RoBERTa (빠른 테스트용)"
        },
        "roberta-base": {
            "params": "125M",
            "layers": 12,
            "hidden_size": 768,
            "description": "Facebook의 RoBERTa 기본 모델"
        },
        "roberta-large": {
            "params": "355M",
            "layers": 24,
            "hidden_size": 1024,
            "description": "큰 RoBERTa 모델"
        },
        "t5-small": {
            "params": "60M",
            "layers": 6,
            "hidden_size": 512,
            "description": "가벼운 T5 (빠른 테스트용)"
        },
        "t5-base": {
            "params": "220M",
            "layers": 12,
            "hidden_size": 768,
            "description": "기본 T5 모델"
        },
        "t5-large": {
            "params": "770M",
            "layers": 24,
            "hidden_size": 1024,
            "description": "큰 T5 모델"
        }
    }
    
    return model_info.get(model_name, {
        "params": "Unknown",
        "layers": "Unknown",
        "hidden_size": "Unknown",
        "description": "모델 정보를 찾을 수 없습니다."
    })