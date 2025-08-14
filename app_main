import streamlit as st
import ezdxf
import numpy as np
from shapely.geometry import LineString, Point
import sys
import os
import io
import tempfile
import matplotlib.pyplot as plt

# 세션 상태 초기화
if 'lookup_table' not in st.session_state:
    st.session_state.lookup_table = None

# lookup_table 로드 함수
def load_lookup_table():
    try:
        from step0_lookup import lookup_table
        st.session_state.lookup_table = lookup_table
        return True
    except ImportError:
        st.error("❌ 오류: step0_lookup.py 파일을 찾을 수 없습니다.")
        return False
    except AttributeError:
        st.error("❌ 오류: step0_lookup.py에서 lookup_table을 찾을 수 없습니다.")
        return False

# 기준표 설정
angle_classes = {"60°전후": (45, 75), "90°전후": (75, 105), "120°전후": (105, 135)}
width_classes = [
    "6m이상8m미만", "8m이상10m미만", "10m이상12m미만", "12m이상15m미만",
    "15m이상20m미만", "20m이상25m미만", "25m이상30m미만", "30m이상35m미만",
    "35m이상40m미만", "40m이상"
]

def get_width_class(width):
    width = round(width + 0.05, 1)
    for cls in width_classes:
        if "이상" in cls and "미만" in cls:
            min_v = float(cls.split("이상")[0].replace("m", ""))
            max_v = float(cls.split("이상")[1].replace("m", "").replace("미만", ""))
            if min_v <= width < max_v:
                return cls
        elif "이상" in cls:
            if width >= float(cls.replace("m이상", "")):
                return cls
    return None

def get_angle_class(degree):
    for k, (min_a, max_a) in angle_classes.items():
        if min_a <= degree < max_a:
            return k
    return None

def get_corner_length(angle_deg, w1, w2):
    ac = get_angle_class(angle_deg)
    wc1 = get_width_class(w1)
    wc2 = get_width_class(w2)
    if ac and wc1 and wc2:
        try:
            lookup_table = st.session_state.lookup_table
            if lookup_table:
                return lookup_table.get((ac, wc1, wc2)) or lookup_table.get((ac, wc2, wc1))
            else:
                st.warning(f"⚠️ 경고: lookup_table이 로드되지 않았습니다.")
                return None
        except (KeyError, AttributeError):
            st.warning(f"⚠️ 경고: lookup_table에서 ({ac}, {wc1}, {wc2}) 조합을 찾을 수 없습니다.")
            return None
    return None

def direction_from_intersection(inter_point, seg):
    coords = list(seg.coords)
    d0 = inter_point.distance(Point(coords[0]))
    d1 = inter_point.distance(Point(coords[-1]))
    if d0 < d1:
        x1, y1 = coords[0]
        x2, y2 = coords[1]
    else:
        x1, y1 = coords[-1]
        x2, y2 = coords[-2]
    return np.arctan2(y2 - y1, x2 - x1)

def shortest_perpendicular_distance(pt, lines):
    return min(line.distance(pt) for line in lines)

def process_dxf_file(uploaded_file):
    """DXF 파일을 처리하여 가각선을 생성하는 함수"""
    try:
        # 임시 파일에 업로드된 파일 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix='.dxf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        # DXF 파일 읽기
        doc = ezdxf.readfile(tmp_file_path)
        msp = doc.modelspace()
        
        # 임시 파일 삭제
        os.unlink(tmp_file_path)
        
    except FileNotFoundError:
        st.error("❌ 오류: DXF 파일을 찾을 수 없습니다.")
        return None
    except ezdxf.DXFError as e:
        st.error(f"❌ 오류: DXF 파일 읽기 오류 - {e}")
        return None
    except Exception as e:
        st.error(f"❌ 오류: 파일 처리 중 오류 발생 - {e}")
        return None

    # '가각선(안)_연장' 텍스트 레이어 생성 (없을 경우)
    if "가각선(안)_연장" not in doc.layers:
        doc.layers.new("가각선(안)_연장", dxfattribs={"color": 3}) # Cyan color

    # '가각선(안)_연장' 텍스트 레이어 생성 (없을 경우)
    if "가각선(안)_연장" not in doc.layers:
        doc.layers.new("가각선(안)_연장", dxfattribs={"color": 3}) # Cyan color

    center_lines, segments, corner_points = [], [], []
    processed_intersections = []  # 처리된 교차점 추적

    for e in msp:
        if e.dxf.layer.lower() == "center":
            if e.dxftype() == "LINE":
                center_lines.append(LineString([(e.dxf.start.x, e.dxf.start.y), (e.dxf.end.x, e.dxf.end.y)]))
            elif e.dxftype() == "LWPOLYLINE":
                center_lines.append(LineString([(p[0], p[1]) for p in e.get_points()]))

        elif e.dxf.layer == "계획선":
            if e.dxftype() == "LINE":
                p1, p2 = e.dxf.start, e.dxf.end
                segments.append(LineString([(p1.x, p1.y), (p2.x, p2.y)]))
                corner_points.extend([Point(p1.x, p1.y), Point(p2.x, p2.y)])
            elif e.dxftype() == "LWPOLYLINE":
                pts = e.get_points()
                for i in range(len(pts) - 1):
                    segments.append(LineString([(pts[i][0], pts[i][1]), (pts[i+1][0], pts[i+1][1])]))
                    corner_points.extend([Point(pts[i][0], pts[i][1]), Point(pts[i+1][0], pts[i+1][1])])

    # 중복 점 제거
    unique_corner_points = []
    for pt in corner_points:
        is_duplicate = False
        for existing_pt in unique_corner_points:
            if pt.distance(existing_pt) < 1e-6:  # 매우 가까운 점은 중복으로 간주
                is_duplicate = True
                break
        if not is_duplicate:
            unique_corner_points.append(pt)
    corner_points = unique_corner_points


    corner_points = unique_corner_points

    # 교차점 탐지 및 분석
    for i in range(len(segments)):
        for j in range(i + 1, len(segments)):
            if segments[i].intersects(segments[j]):
                pt = segments[i].intersection(segments[j])
                if not isinstance(pt, Point):
                    continue

                # 이미 처리된 교차점인지 확인
                is_already_processed = False
                for processed_pt in processed_intersections:
                    if pt.distance(processed_pt) < 1e-6:  # 매우 가까운 점은 같은 점으로 간주
                        is_already_processed = True
                        break
                
                if is_already_processed:
                    continue
                
                # 현재 교차점을 처리된 목록에 추가
                processed_intersections.append(pt)

                # 현재 교차점을 처리된 목록에 추가
                processed_intersections.append(pt)

                a1 = direction_from_intersection(pt, segments[i])
                a2 = direction_from_intersection(pt, segments[j])
                vx = np.cos(a1) + np.cos(a2)
                vy = np.sin(a1) + np.sin(a2)
                mid_angle = np.arctan2(vy, vx)

                local_pts = [p for p in corner_points if pt.distance(p) < 20]
                if len(local_pts) < 2:
                    continue
                local_pts.sort(key=lambda p: pt.distance(p))
                corner1, corner2 = local_pts[:2]

                d1 = shortest_perpendicular_distance(corner1, center_lines)
                d2 = shortest_perpendicular_distance(corner2, center_lines)
                w1 = round(d1 * 2, 3)
                w2 = round(d2 * 2, 3)

                corner_len = get_corner_length(abs(np.rad2deg((a2 - a1 + np.pi) % (2*np.pi) - np.pi)), w1, w2)
                if not corner_len:
                    continue

                short_len = corner_len / 2 # 가각선 길이의 절반 (중간선 계산용)
                
                # 연장할 길이 설정 (3m)
                extension_length_for_dotted_line = 3 

                offset = short_len * 1  # 평행이동 거리 = 가각선 길이의 1/2

                intersection_points = []

                for sign in [1, -1]:
                    shift_x = sign * offset * np.cos(mid_angle + np.pi / 2)
                    shift_y = sign * offset * np.sin(mid_angle + np.pi / 2)
                    
                    # 교차점(pt)에서 mid_angle 방향으로 short_len 만큼 떨어진 지점 (원래 점선 끝점)
                    original_end_x = pt.x + short_len * np.cos(mid_angle)
                    original_end_y = pt.y + short_len * np.sin(mid_angle)

                    # 노란색 점선의 시작점 (평행이동된 교차점)
                    start_dotted_x = pt.x + shift_x
                    start_dotted_y = pt.y + shift_y

                    # 노란색 점선의 끝점 (평행이동된 원래 끝점)
                    end_dotted_x = original_end_x + shift_x
                    end_dotted_y = original_end_y + shift_y

                    # 노란색 점선의 방향 벡터 계산
                    dx_dotted = end_dotted_x - start_dotted_x
                    dy_dotted = end_dotted_y - start_dotted_y
                    
                    # 길이가 0이 아닌 경우에만 정규화
                    norm_dotted = np.sqrt(dx_dotted**2 + dy_dotted**2)
                    if norm_dotted > 1e-6: # 아주 작은 값으로 0 방지
                        unit_dx_dotted = dx_dotted / norm_dotted
                        unit_dy_dotted = dy_dotted / norm_dotted
                    else: # 선분 길이가 0에 가까우면 연장하지 않음
                        unit_dx_dotted = 0
                        unit_dy_dotted = 0

                    # 시작점에서 반대 방향으로 3m 연장된 새로운 시작점
                    extended_start_dotted_x = start_dotted_x - extension_length_for_dotted_line * unit_dx_dotted
                    extended_start_dotted_y = start_dotted_y - extension_length_for_dotted_line * unit_dy_dotted

                    # 끝점에서 같은 방향으로 3m 연장된 새로운 끝점
                    extended_end_dotted_x = end_dotted_x + extension_length_for_dotted_line * unit_dx_dotted
                    extended_end_dotted_y = end_dotted_y + extension_length_for_dotted_line * unit_dy_dotted

                    # 연장된 노란색 점선 생성 (시각화만 제거)
                    extended_dotted_line = LineString([
                        (extended_start_dotted_x, extended_start_dotted_y),
                        (extended_end_dotted_x, extended_end_dotted_y)
                    ])

                    for seg in segments:
                        if extended_dotted_line.intersects(seg):
                            inter_pt = extended_dotted_line.intersection(seg)
                            if isinstance(inter_pt, Point):
                                intersection_points.append(inter_pt)

                if len(intersection_points) == 2:
                    # 가각선 LineString 생성
                    final_corner_line = LineString([intersection_points[0], intersection_points[1]])

                    # 가각선 DXF에 추가
                    msp.add_line(
                        (intersection_points[0].x, intersection_points[0].y),
                        (intersection_points[1].x, intersection_points[1].y),
                        dxfattribs={"layer": "가각선(안)"}
                    )

                    # 텍스트 표기
                    # 가각선의 길이 계산
                    corner_line_length = final_corner_line.length

                    # 텍스트 내용 정의 (길이만 표기)
                    text_content = f"길이: {corner_line_length:.2f}m"

                    # 텍스트 위치: 가각선의 중간점
                    mid_point = final_corner_line.interpolate(0.5, normalized=True)

                    # 텍스트 회전 각도: 가각선의 각도
                    line_angle_rad = np.arctan2(
                        final_corner_line.coords[1][1] - final_corner_line.coords[0][1],
                        final_corner_line.coords[1][0] - final_corner_line.coords[0][0]
                    )
                    line_angle_deg = np.degrees(line_angle_rad)

                    # 텍스트가 선과 겹치지 않도록 약간 오프셋 (선에 수직 방향으로)
                    text_offset_distance = 0.5 # 텍스트가 선에서 떨어질 거리
                    text_offset_x = text_offset_distance * np.cos(line_angle_rad + np.pi / 2)
                    text_offset_y = text_offset_distance * np.sin(line_angle_rad + np.pi / 2)

                    text_insert_point = (mid_point.x + text_offset_x, mid_point.y + text_offset_y)

                    # DXF에 MTEXT 엔티티 추가
                    msp.add_mtext(
                        text_content,
                        dxfattribs={
                            "layer": "가각선(안)_연장",
                            "char_height": 0.8,  # 텍스트 높이 (도면 단위)
                            "rotation": line_angle_deg, # 선분의 각도에 맞춰 회전
                            "insert": text_insert_point,
                            "attachment_point": 5 # 5는 Middle Center (중앙에 텍스트가 위치하도록)
                        }
                    )

    # DXF 파일을 바이트 스트림으로 변환하여 반환
    try:
        output_buffer = io.BytesIO()
        with tempfile.NamedTemporaryFile(delete=False, suffix='.dxf') as tmp_file:
            doc.saveas(tmp_file.name)
            with open(tmp_file.name, 'rb') as f:
                output_buffer.write(f.read())
            os.unlink(tmp_file.name)
        
        output_buffer.seek(0)
        return output_buffer
        
    except Exception as e:
        st.error(f"❌ 오류: DXF 파일 처리 오류 - {e}")
        return None


        return None

# 스트림릿 메인 인터페이스
def main():
    st.set_page_config(
        page_title="가각선 자동 생성 시스템",
        page_icon="📐",
        layout="wide"
    )
    
    st.title("📐 가각선 자동 생성 시스템")
    st.markdown("---")
    
    # lookup_table 로드 확인
    if st.session_state.lookup_table is None:
        with st.spinner("lookup_table을 로드하는 중..."):
            if not load_lookup_table():
                st.error("lookup_table을 로드할 수 없습니다. step0_lookup.py 파일을 확인해주세요.")
                st.stop()
            else:
                st.success("✅ lookup_table이 성공적으로 로드되었습니다!")
    
    # 사이드바에 설명 추가
    with st.sidebar:
        st.header("📋 사용 방법")
        st.markdown("""
        1. DXF 파일을 업로드하세요
        2. 파일이 처리될 때까지 기다리세요
        3. 결과 파일을 다운로드하세요
        
        **필요한 레이어:**
        - `center` (중심선)
        - `계획선` (계획선)
        
        **생성되는 레이어:**
        - `가각선(안)` (가각선)
        - `가각선(안)_연장` (길이 텍스트)
        """)
    
    # 파일 업로드
    st.header("📂 DXF 파일 업로드")
    uploaded_file = st.file_uploader(
        "DXF 파일을 선택하세요",
        type=['dxf'],
        accept_multiple_files=False
    )
    
    if uploaded_file is not None:
        st.success(f"✅ 파일 업로드 완료: {uploaded_file.name}")
        
        # 파일 정보 표시
        file_details = {
            "파일명": uploaded_file.name,
            "파일 크기": f"{uploaded_file.size:,} bytes"
        }
        st.json(file_details)
        
        # 처리 버튼
        if st.button("🚀 가각선 생성 시작", type="primary"):
            with st.spinner("DXF 파일을 처리하는 중... 잠시만 기다려주세요."):
                try:
                    # DXF 파일 처리
                    result_buffer = process_dxf_file(uploaded_file)
                    
                    if result_buffer:
                        st.success("✅ 가각선 생성이 완료되었습니다!")
                        
                        # 다운로드 버튼
                        st.download_button(
                            label="📥 결과 파일 다운로드",
                            data=result_buffer.getvalue(),
                            file_name="가각_결과.dxf",
                            mime="application/octet-stream",
                            type="primary"
                        )
                        
                        # 성공 메시지
                        st.balloons()
                        
                    else:
                        st.error("❌ 파일 처리 중 오류가 발생했습니다.")
                        
                except Exception as e:
                    st.error(f"❌ 처리 중 오류 발생: {str(e)}")
    
    else:
        st.info("👆 DXF 파일을 업로드해주세요.")
    
    # 푸터
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center;">
            <small>가각선 자동 생성 시스템 v1.0 | Powered by Streamlit</small>
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
