"""
02_yolo_v2_evaluation.py
─────────────────────────────────
v2 best.pt를 검증 데이터셋에서 직접 평가
"""

from ultralytics import YOLO

# 프로젝트 루트 (실행 위치/사용자 경로 비의존)
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    # 모델 + 데이터 경로
    model_path = (PROJECT_ROOT + r'\models'
                  r'\yolo11_sperm_v2\weights\best.pt')

    data_yaml = (PROJECT_ROOT + r'\data\processed'
                 r'\yolo_dataset\visem.yaml')

    print("=" * 70)
    print("  YOLO11m v2 best.pt 직접 평가")
    print("=" * 70)

    # 모델 로드
    print("\n📦 모델 로딩...")
    model = YOLO(model_path)
    print(f"   클래스: {model.names}")

    # 검증 실행
    print("\n🔍 검증 실행 중...")
    print("─" * 70)

    results = model.val(
        data=data_yaml,
        split='val',
        imgsz=640,
        batch=16,
        conf=0.25,
        iou=0.6,
        device=0,
        save_json=False,
        verbose=True,
        workers=0,        # 멀티프로세싱 비활성화 (Windows 안정성)
    )

    # 결과 정리
    print("\n" + "=" * 70)
    print("  📊 클래스별 성능")
    print("=" * 70)

    print(f"\n전체 평균:")
    print(f"   mAP50:    {results.box.map50:.4f}")
    print(f"   mAP50-95: {results.box.map:.4f}")
    print(f"   Precision: {results.box.mp:.4f}")
    print(f"   Recall:    {results.box.mr:.4f}")

    # 클래스별 mAP50 (ap50: 클래스별 mAP50 배열)
    print(f"\n클래스별 mAP50:")
    class_names = model.names
    if hasattr(results.box, 'ap50'):
        ap50_per_class = results.box.ap50
        for i, name in class_names.items():
            if i < len(ap50_per_class):
                print(f"   [{i}] {name:<12}: mAP50 = "
                      f"{ap50_per_class[i]:.4f}")

    # mAP50-95 클래스별
    print(f"\n클래스별 mAP50-95:")
    if hasattr(results.box, 'maps'):
        for i, name in class_names.items():
            if i < len(results.box.maps):
                print(f"   [{i}] {name:<12}: mAP50-95 = "
                      f"{results.box.maps[i]:.4f}")

    print(f"\n📁 검증 결과 폴더:")
    print(f"   {results.save_dir}")

    print("\n" + "=" * 70)
    print("  ✅ 평가 완료")
    print("=" * 70)


if __name__ == '__main__':
    main()