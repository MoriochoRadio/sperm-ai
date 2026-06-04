"""
25_system_diagram_horizontal.py
─────────────────────────────────
Figure 1 가로형 (양 단 너비 또는 압축형)

목적:
  - 가로 흐름으로 변경
  - 좌→우 자연스러운 진행
  - 페이지 가로 너비 활용
"""

import os
import requests
import base64

# 프로젝트 루트 (실행 위치/사용자 경로 비의존)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    print("=" * 80)
    print("  Figure 1 가로형: 시스템 다이어그램")
    print("=" * 80)

    # 가로형 Mermaid 코드
    mermaid_code = """
flowchart LR
    A[Input<br/>Video] --> B{{Quality<br/>Assessment<br/>6-dim}}
    B -->|F| C[Reject]
    B -->|S/A/B/C| D[Smart<br/>Normalization]
    D --> E[YOLO11m<br/>Detection]
    E --> F[ByteTrack<br/>Tracking]
    F --> G[Motility]
    F --> H[CASA<br/>Kinematics]
    F --> I[Morphology v3<br/>+ VisemStyleAugment]
    G --> J[Report]
    H --> J
    I --> J
    
    style B fill:#DBEAFE,stroke:#2563EB,stroke-width:3px
    style D fill:#DBEAFE,stroke:#2563EB,stroke-width:3px
    style I fill:#DBEAFE,stroke:#2563EB,stroke-width:3px
    style C fill:#FEE2E2,stroke:#DC2626
    style A fill:#FEF3C7,stroke:#F59E0B
    style J fill:#FEF3C7,stroke:#F59E0B
    style E fill:#E0E7FF,stroke:#6366F1
    style F fill:#E0E7FF,stroke:#6366F1
    style G fill:#D1FAE5,stroke:#10B981
    style H fill:#D1FAE5,stroke:#10B981
"""

    # 파일 저장
    output_dir = PROJECT_ROOT + r'\paper\figures'
    os.makedirs(output_dir, exist_ok=True)
    
    code_path = os.path.join(output_dir, 
                              'fig1_horizontal_mermaid.txt')
    with open(code_path, 'w', encoding='utf-8') as f:
        f.write(mermaid_code)
    print(f"\n📁 Mermaid 코드: {code_path}")

    # Mermaid.ink API
    print("\n🎨 PNG 자동 생성...")
    
    try:
        graph_bytes = mermaid_code.strip().encode('utf-8')
        base64_string = base64.urlsafe_b64encode(
            graph_bytes).decode('ascii')
        
        api_url = f'https://mermaid.ink/img/{base64_string}'
        response = requests.get(api_url, timeout=30)
        
        if response.status_code == 200:
            png_path = os.path.join(output_dir, 
                                      'fig1_horizontal.png')
            with open(png_path, 'wb') as f:
                f.write(response.content)
            print(f"   ✅ PNG 생성!")
            print(f"   📁 {png_path}")
        else:
            print(f"   ⚠️  자동 생성 실패")
    except Exception as e:
        print(f"   ⚠️  오류: {e}")

    print("\n" + "=" * 80)
    print("  📌 사용 방법")
    print("=" * 80)
    print("""
[방법 1] 자동 생성된 PNG 사용
   - fig1_horizontal.png 직접 사용
   
[방법 2] Mermaid Live Editor
   1. https://mermaid.live 접속
   2. fig1_horizontal_mermaid.txt 복사
   3. 붙여넣기 후 PNG 다운로드

[hwp에 배치]
   - 양 단 너비로 삽입 (페이지 가로 전체)
   - 또는 1단 너비로 압축 배치
   - 본문이 위/아래로 흐름
""")


if __name__ == '__main__':
    main()