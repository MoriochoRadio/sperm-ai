"""
19_system_diagram_v2.py
─────────────────────────────────
Figure 1: 시스템 다이어그램 v2 (Mermaid 기반)

목적:
  - 깔끔한 학술 figure
  - 우리 contribution 강조
  - 흐름 명확
"""

import os
import requests
import base64

# 프로젝트 루트 (실행 위치/사용자 경로 비의존)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    print("=" * 80)
    print("  Figure 1 v2: Mermaid 기반 시스템 다이어그램")
    print("=" * 80)

    # ── Mermaid 다이어그램 정의 ─────────────────────────
    mermaid_code = """
flowchart TB
    A[Input<br/>Microscopy Video] --> B{{Quality Assessment<br/>6-dim Score}}
    B -->|F grade| C[Reject<br/>+ Guide]
    B -->|S/A/B/C| D[Smart Normalization<br/>spec-only]
    D --> E[Detection<br/>YOLO11m]
    E --> F[Tracking<br/>ByteTrack]
    F --> G1[Motility<br/>Analysis]
    F --> G2[CASA<br/>Kinematics]
    F --> G3[Morphology v3<br/>EfficientNet-B3<br/>+ VisemStyleAugment]
    G1 --> H[Comprehensive<br/>Report]
    G2 --> H
    G3 --> H
    
    style B fill:#DBEAFE,stroke:#2563EB,stroke-width:3px
    style D fill:#DBEAFE,stroke:#2563EB,stroke-width:3px
    style G3 fill:#DBEAFE,stroke:#2563EB,stroke-width:3px
    style C fill:#FEE2E2,stroke:#DC2626
    style A fill:#FEF3C7,stroke:#F59E0B
    style H fill:#FEF3C7,stroke:#F59E0B
    style E fill:#E0E7FF,stroke:#6366F1
    style F fill:#E0E7FF,stroke:#6366F1
    style G1 fill:#D1FAE5,stroke:#10B981
    style G2 fill:#D1FAE5,stroke:#10B981
"""

    # ── Mermaid 코드 출력 (사용자가 편집 가능) ────────────
    print("\n📋 Mermaid 코드:")
    print("─" * 80)
    print(mermaid_code)
    print("─" * 80)

    # ── 코드 파일로 저장 ──────────────────────────────────
    output_dir = PROJECT_ROOT + r'\paper\figures'
    os.makedirs(output_dir, exist_ok=True)
    
    code_path = os.path.join(output_dir, 'fig1_v2_mermaid.txt')
    with open(code_path, 'w', encoding='utf-8') as f:
        f.write(mermaid_code)
    print(f"\n📁 Mermaid 코드 저장: {code_path}")

    # ── Mermaid.ink API로 PNG 생성 시도 ───────────────────
    print("\n🎨 Mermaid.ink API로 PNG 생성 시도...")
    
    try:
        # Mermaid 코드를 base64 인코딩
        graph_bytes = mermaid_code.strip().encode('utf-8')
        base64_string = base64.urlsafe_b64encode(
            graph_bytes).decode('ascii')
        
        # API URL (PNG)
        api_url = f'https://mermaid.ink/img/{base64_string}'
        print(f"   API URL 길이: {len(api_url)} chars")
        
        # 다운로드
        response = requests.get(api_url, timeout=30)
        
        if response.status_code == 200:
            png_path = os.path.join(output_dir, 
                                      'fig1_v2_system.png')
            with open(png_path, 'wb') as f:
                f.write(response.content)
            print(f"   ✅ PNG 생성 성공!")
            print(f"   📁 저장: {png_path}")
        else:
            print(f"   ⚠️  API 응답 실패: {response.status_code}")
            print(f"   → 수동 생성 필요")
    
    except Exception as e:
        print(f"   ⚠️  자동 생성 실패: {e}")
        print(f"   → 수동 생성 필요")

    # ── 사용자 안내 ───────────────────────────────────────
    print("\n" + "=" * 80)
    print("  📌 사용 방법 안내")
    print("=" * 80)
    print("""
[방법 1] Mermaid Live Editor (추천 ⭐)
   1. https://mermaid.live 접속
   2. fig1_v2_mermaid.txt 내용 복사
   3. Editor 좌측에 붙여넣기
   4. 우측에서 다이어그램 확인
   5. "Export" → "PNG" 또는 "SVG" 다운로드
   6. paper/figures 폴더에 저장
   
[방법 2] 자동 생성된 PNG 사용 (성공 시)
   - fig1_v2_system.png 바로 사용 가능
   
[방법 3] 다른 도구
   - draw.io: https://app.diagrams.net
   - Excalidraw: https://excalidraw.com
   → 더 자유로운 디자인 가능
""")
    
    print("=" * 80)


if __name__ == '__main__':
    main()