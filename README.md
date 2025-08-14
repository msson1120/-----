# 가각선 자동 생성 시스템 (Streamlit)

## 로컬 실행
```bash
python -m venv .venv
# PowerShell
. ./.venv/Scripts/Activate.ps1
# (CMD: .venv\Scripts\activate.bat)

pip install -r requirements.txt
streamlit run app.py
```

## 필수 파일
- `app.py` : Streamlit 앱
- `step0_lookup.py` : `lookup_table` 사전이 정의되어 있어야 합니다.

## DXF 레이어 전제
- 입력: `center`, `계획선`
- 출력: `가각선(안)`, `가각선(안)_연장`

## GitHub 업로드
```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/<YOUR-ID>/cornerline-app.git
git push -u origin main
```

## (선택) Streamlit Cloud 배포
- Streamlit Cloud에서 `New app` → 해당 레포 선택 → Branch `main` → main file: `app.py` → Deploy
