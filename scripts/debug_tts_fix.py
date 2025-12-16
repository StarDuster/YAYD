from youdub.core.steps.synthesize_speech import _DEFAULT_SETTINGS, _BYTEDANCE_API_URL
import json
import logging
import requests
import uuid

logging.basicConfig(level=logging.WARNING)

appid = _DEFAULT_SETTINGS.bytedance_appid
access_token = _DEFAULT_SETTINGS.bytedance_access_token

print(f"Testing fix with AppID: {appid}")

# 构造修正后的 Payload，把 token 字段设为真实 token
text = "测试语音修复。"
voice_type = "BV001_streaming"

header = {"Authorization": f"Bearer;{access_token}"}
request_json = {
    "app": {
        "appid": appid,
        "token": access_token,  # <--- FIX: Use the actual variable, not string "access_token"
        "cluster": 'volcano_tts'
    },
    "user": {
        "uid": "test_user_001"
    },
    "audio": {
        "voice_type": voice_type,
        "encoding": "wav",
        "speed_ratio": 1.0,
        "volume_ratio": 1.0,
        "pitch_ratio": 1.0,
    },
    "request": {
        "reqid": str(uuid.uuid4()),
        "text": text,
        "text_type": "plain",
        "operation": "query",
        "with_frontend": 1,
        "frontend_type": "unitTson"
    }
}

try:
    print("Sending request with FIXED token in body...")
    resp = requests.post(_BYTEDANCE_API_URL, json.dumps(request_json), headers=header, timeout=60)
    response_json = resp.json()
    
    if "data" in response_json:
        print("SUCCESS! Received audio data.")
        print(f"Data length: {len(response_json['data'])}")
    else:
        print(f"FAILED. API Response: {response_json}")

except Exception as e:
    print(f"Exception: {e}")
