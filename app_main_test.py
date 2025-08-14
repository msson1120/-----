import streamlit as st
import ezdxf
import numpy as np
from shapely.geometry import LineString, Point, MultiPoint
import sys
import os
import io
import tempfile
import matplotlib.pyplot as plt
import math
from collections import defaultdict

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

def get_width_class(w):
    # 유연한 폭 분류 (±0.2m 여유)
    tolerance = 0.2
    
    if math.isclose(w, 4.0, abs_tol=tolerance) or w < 4.0 + tolerance:
        return "4m미만"
    elif 4.0 - tolerance <= w < 6.0 + tolerance:
        return "4m이상6m미만"
    elif 6.0 - tolerance <= w < 8.0 + tolerance:
        return "6m이상8m미만"
    elif 8.0 - tolerance <= w < 10.0 + tolerance:
        return "8m이상10m미만"
    elif 10.0 - tolerance <= w < 12.0 + tolerance:
        return "10m이상12m미만"
    elif 12.0 - tolerance <= w < 15.0 + tolerance:
        return "12m이상15m미만"
    elif w >= 15.0 - tolerance:
        return "15m이상"
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
    if not lines:  # center선이 없는 경우 None 반환
        return None
    return min(line.distance(pt) for line in lines)

def extract_intersection_points(intersection_result):
    """다양한 intersection 결과에서 Point들을 추출"""
    points = []
    
    if intersection_result.geom_type == "Point":
        points.append(intersection_result)
    elif intersection_result.geom_type == "MultiPoint":
        for pt in intersection_result.geoms:
            if isinstance(pt, Point):
                points.append(pt)
    elif intersection_result.geom_type == "LineString":
        # LineString의 중점을 교차점으로 사용
        midpoint = intersection_result.interpolate(0.5, normalized=True)
        points.append(midpoint)
    elif intersection_result.geom_type == "MultiLineString":
        # 각 LineString의 중점들을 사용
        for line in intersection_result.geoms:
            midpoint = line.interpolate(0.5, normalized=True)
            points.append(midpoint)
    
    return points

def remove_duplicates_fast(points, tolerance=1e-6):
    """공간 인덱싱을 사용한 빠른 중복 점 제거"""
    grid_size = tolerance * 10
    grid = defaultdict(list)
    unique_points = []
    
    for pt in points:
        grid_x = int(pt.x / grid_size)
        grid_y = int(pt.y / grid_size)
        grid_key = (grid_x, grid_y)
        
        is_duplicate = False
        # 인근 그리드 셀만 검사
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                neighbor_key = (grid_x + dx, grid_y + dy)
                for existing_pt in grid[neighbor_key]:
                    if pt.distance(existing_pt) < tolerance:
                        is_duplicate = True
                        break
                if is_duplicate:
                    break
            if is_duplicate:
                break
        
        if not is_duplicate:
            grid[grid_key].append(pt)
            unique_points.append(pt)
    
    return unique_points

def point_to_key(pt, precision=6):
    """Point를 해시 가능한 키로 변환"""
    return (round(pt.x, precision), round(pt.y, precision))

def process_dxf_file(uploaded_file, progress_bar=None, status_text=None):
    """DXF 파일을 처리하여 가각선을 생성하는 함수"""
    
    def update_progress(step, total_steps, message):
        if progress_bar:
            progress_bar.progress(step / total_steps)
        if status_text:
            status_text.text(f"단계 {step}/{total_steps}: {message}")
    
    total_steps = 8
    current_step = 0
    
    # 1단계: 파일 읽기
    current_step += 1
    update_progress(current_step, total_steps, "DXF 파일 읽기 중...")
    
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

    # 2단계: 레이어 생성
    current_step += 1
    update_progress(current_step, total_steps, "레이어 생성 중...")
    
    # '가각선(안)' 레이어 생성 (없을 경우)
    if "가각선(안)" not in doc.layers:
        doc.layers.new("가각선(안)", dxfattribs={"color": 1})  # Red color

    # '가각선(안)_연장' 텍스트 레이어 생성 (없을 경우)
    if "가각선(안)_연장" not in doc.layers:
        doc.layers.new("가각선(안)_연장", dxfattribs={"color": 3}) # Cyan color

    # 3단계: 엔티티 분석
    current_step += 1
    update_progress(current_step, total_steps, "도면 엔티티 분석 중...")
    
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
                # 기존 방식: 연속된 점들로 세그먼트 생성
                for i in range(len(pts) - 1):
                    segments.append(LineString([(pts[i][0], pts[i][1]), (pts[i+1][0], pts[i+1][1])]))
                    corner_points.extend([Point(pts[i][0], pts[i][1]), Point(pts[i+1][0], pts[i+1][1])])
                
                # 곡선부 대응: 긴 세그먼트를 중간 점들로 세분화
                for i in range(len(pts) - 1):
                    seg_length = Point(pts[i][0], pts[i][1]).distance(Point(pts[i+1][0], pts[i+1][1]))
                    if seg_length > 2:  # 2m 이상 긴 세그먼트
                        # 중간 점들 추가 (1m 간격)
                        num_subdivisions = int(seg_length / 1)
                        for k in range(1, num_subdivisions):
                            ratio = k / num_subdivisions
                            mid_x = pts[i][0] + ratio * (pts[i+1][0] - pts[i][0])
                            mid_y = pts[i][1] + ratio * (pts[i+1][1] - pts[i][1])
                            corner_points.append(Point(mid_x, mid_y))

    # 4단계: 중복 점 제거
    current_step += 1
    update_progress(current_step, total_steps, "중복 점 제거 중...")
    
    # 최적화된 중복 점 제거
    corner_points = remove_duplicates_fast(corner_points)

    # 5단계: 데이터 검증
    current_step += 1
    update_progress(current_step, total_steps, "데이터 검증 중...")
    
    # center_lines 필수 검증 추가
    if not center_lines:
        st.error("❌ 오류: 'center' 레이어를 찾을 수 없습니다. center 레이어가 있어야 가각선을 생성할 수 있습니다.")
        return None
    
    if not segments:
        st.error("❌ 오류: '계획선' 레이어를 찾을 수 없습니다.")
        return None

    # 6단계: 교차점 탐지 및 가각선 생성
    current_step += 1
    update_progress(current_step, total_steps, "교차점 탐지 및 가각선 생성 중...")
    
    total_intersections = 0
    processed_intersections_count = 0
    
    # 전체 교차점 개수 계산
    for i in range(len(segments)):
        for j in range(i + 1, len(segments)):
            if segments[i].intersects(segments[j]):
                total_intersections += 1
    
    if status_text:
        status_text.text(f"단계 {current_step}/{total_steps}: {total_intersections}개 교차점 처리 중...")
    
    # Set을 사용한 빠른 중복 검사
    processed_intersections_set = set()
    
    # UI 업데이트 빈도 제한
    update_interval = max(1, total_intersections // 20)  # 최대 20번만 업데이트
    
    # 교차점 탐지 및 분석 (최적화된 버전)
    for i in range(len(segments)):
        for j in range(i + 1, len(segments)):
            if segments[i].intersects(segments[j]):
                processed_intersections_count += 1
                
                # UI 업데이트 빈도 제한
                if processed_intersections_count % update_interval == 0 and status_text:
                    status_text.text(f"단계 {current_step}/{total_steps}: 교차점 처리 중... ({processed_intersections_count}/{total_intersections})")
                
                intersection_result = segments[i].intersection(segments[j])
                
                # 다양한 intersection 결과 처리
                intersection_points = extract_intersection_points(intersection_result)
                
                for pt in intersection_points:
                    pt_key = point_to_key(pt)
                    
                    # Set을 사용한 빠른 중복 검사
                    if pt_key in processed_intersections_set:
                        continue
                    
                    processed_intersections_set.add(pt_key)

                    a1 = direction_from_intersection(pt, segments[i])
                    a2 = direction_from_intersection(pt, segments[j])
                    vx = np.cos(a1) + np.cos(a2)
                    vy = np.sin(a1) + np.sin(a2)
                    mid_angle = np.arctan2(vy, vx)

                    # 최적화된 corner_points 탐색
                    # 한 번에 모든 거리 계산 후 정렬
                    distances = [(p, pt.distance(p)) for p in corner_points]
                    distances.sort(key=lambda x: x[1])
                    
                    # 필요한 개수만 선택
                    local_pts = []
                    for p, dist in distances:
                        if dist < 40:  # 첫 번째 범위
                            local_pts.append(p)
                        if len(local_pts) >= 2:
                            break
                    
                    # 여전히 부족하면 확장 범위에서 추가
                    if len(local_pts) < 2:
                        for p, dist in distances:
                            if 40 <= dist < 80:  # 확장 범위
                                local_pts.append(p)
                            if len(local_pts) >= 2:
                                break
                    
                    if len(local_pts) < 2:
                        st.warning(f"⚠️ 교차점 ({pt.x:.2f}, {pt.y:.2f}) 근처에 충분한 corner_points가 없습니다. (발견: {len(local_pts)}개)")
                        continue
                        
                    corner1, corner2 = local_pts[:2]

                    d1 = shortest_perpendicular_distance(corner1, center_lines)
                    d2 = shortest_perpendicular_distance(corner2, center_lines)
                    
                    # center선이 없어서 도로폭을 계산할 수 없는 경우
                    if d1 is None or d2 is None:
                        st.warning(f"⚠️ center선이 없어서 교차점 ({pt.x:.2f}, {pt.y:.2f})에서 도로폭을 계산할 수 없습니다.")
                        continue
                    
                    w1 = round(d1 * 2, 3)
                    w2 = round(d2 * 2, 3)

                    # 교차각 계산
                    intersection_angle = abs(np.rad2deg((a2 - a1 + np.pi) % (2*np.pi) - np.pi))
                    
                    corner_len = get_corner_length(intersection_angle, w1, w2)
                    if not corner_len:
                        # 기본값 사용 (lookup_table에서 찾지 못한 경우)
                        st.warning(f"⚠️ lookup_table에서 조건을 찾지 못했습니다. 각도: {intersection_angle:.1f}°, 폭: {w1}m, {w2}m")
                        # 각도와 폭에 따른 경험적 공식 사용
                        avg_width = (w1 + w2) / 2
                        if intersection_angle < 75:  # 예각
                            corner_len = avg_width * 0.8
                        elif intersection_angle > 105:  # 둔각
                            corner_len = avg_width * 1.2
                        else:  # 직각 근처
                            corner_len = avg_width * 1.0
                        st.info(f"📏 경험적 공식으로 가각선 길이 계산: {corner_len:.2f}m")
                    
                    if corner_len <= 0:
                        continue

                    short_len = corner_len / 2 # 가각선 길이의 절반 (중간선 계산용)
                    
                    # 연장할 길이 설정 (3m)
                    extension_length_for_dotted_line = 3 

                    offset = short_len * 1  # 평행이동 거리 = 가각선 길이의 1/2

                    intersection_points_final = []

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
                                    intersection_points_final.append(inter_pt)

                    if len(intersection_points_final) == 2:
                        # 가각선 LineString 생성
                        final_corner_line = LineString([intersection_points_final[0], intersection_points_final[1]])

                        # 가각선 DXF에 추가
                        msp.add_line(
                            (intersection_points_final[0].x, intersection_points_final[0].y),
                            (intersection_points_final[1].x, intersection_points_final[1].y),
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

    # 7단계: 결과 파일 생성
    current_step += 1
    update_progress(current_step, total_steps, "결과 DXF 파일 생성 중...")

    # DXF 파일을 바이트 스트림으로 변환하여 반환
    try:
        output_buffer = io.BytesIO()
        with tempfile.NamedTemporaryFile(delete=False, suffix='.dxf') as tmp_file:
            doc.saveas(tmp_file.name)
            with open(tmp_file.name, 'rb') as f:
                output_buffer.write(f.read())
            os.unlink(tmp_file.name)
        
        output_buffer.seek(0)
        
        # 8단계: 완료
        current_step += 1
        update_progress(current_step, total_steps, "처리 완료!")
        
        return output_buffer
        
    except Exception as e:
        st.error(f"❌ 오류: DXF 파일 처리 오류 - {e}")
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
        - `center` (중심선) - **필수**
        - `계획선` (계획선) - **필수**
        
        **생성되는 레이어:**
        - `가각선(안)` (가각선)
        - `가각선(안)_연장` (길이 텍스트)
        """)
        
        with st.expander("⚙️ 기술적 세부사항"):
            st.markdown("""
            **교차점 처리:**
            - Point, MultiPoint, LineString 모두 지원
            - 중복 교차점 자동 제거
            
            **corner_points 탐색:**
            - 1차: 40m 범위 탐색
            - 2차: 80m 범위 확장 탐색
            - 긴 세그먼트 자동 세분화 (1m 간격)
            
            **폭 계산:**
            - center선 없으면 처리 중단
            - ±0.2m 허용 오차로 유연한 분류
            
            **가각선 길이:**
            - lookup_table 우선 사용
            - 실패시 경험적 공식 적용
            
            **처리 진행률:**
            - 8단계 세분화된 처리 과정
            - 실시간 상태 표시
            - 교차점별 처리 진행률
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
            # 프로그레스 바와 상태 텍스트 생성
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # DXF 파일 처리 (프로그레스 바 포함)
                result_buffer = process_dxf_file(uploaded_file, progress_bar, status_text)
                
                if result_buffer:
                    st.success("✅ 가각선 생성이 완료되었습니다!")
                    
                    # 프로그레스 바와 상태 텍스트 제거
                    progress_bar.empty()
                    status_text.empty()
                    
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
                    # 프로그레스 바와 상태 텍스트 제거
                    progress_bar.empty()
                    status_text.empty()
                    st.error("❌ 파일 처리 중 오류가 발생했습니다.")
                    
            except Exception as e:
                # 프로그레스 바와 상태 텍스트 제거
                progress_bar.empty()
                status_text.empty()
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
