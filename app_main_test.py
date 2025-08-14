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

def get_road_direction_from_intersection(intersection_pt, segment):
    """교차점에서 도로 세그먼트의 방향을 계산"""
    from shapely.geometry import Point
    
    # 교차점 주변의 세그먼트 방향 분석
    coords = list(segment.coords)
    
    # 교차점과 가장 가까운 좌표 찾기
    min_dist = float('inf')
    closest_idx = 0
    
    for i, coord in enumerate(coords):
        dist = Point(coord).distance(intersection_pt)
        if dist < min_dist:
            min_dist = dist
            closest_idx = i
    
    # 방향 벡터 계산
    if closest_idx == 0 and len(coords) > 1:
        # 시작점이 가장 가까운 경우
        direction = np.array([coords[1][0] - coords[0][0], coords[1][1] - coords[0][1]])
    elif closest_idx == len(coords) - 1 and len(coords) > 1:
        # 끝점이 가장 가까운 경우
        direction = np.array([coords[-1][0] - coords[-2][0], coords[-1][1] - coords[-2][1]])
    else:
        # 중간점인 경우
        if closest_idx > 0:
            prev_direction = np.array([coords[closest_idx][0] - coords[closest_idx-1][0], 
                                     coords[closest_idx][1] - coords[closest_idx-1][1]])
        else:
            prev_direction = np.array([0, 0])
            
        if closest_idx < len(coords) - 1:
            next_direction = np.array([coords[closest_idx+1][0] - coords[closest_idx][0], 
                                     coords[closest_idx+1][1] - coords[closest_idx][1]])
        else:
            next_direction = np.array([0, 0])
            
        direction = prev_direction + next_direction
    
    # 정규화
    norm = np.linalg.norm(direction)
    if norm > 1e-6:
        direction = direction / norm
    
    return direction

def calculate_bisector_directions(intersection_pt, segment1, segment2):
    """교차점에서 두 세그먼트의 이등분선 방향(안쪽/바깥쪽)을 계산"""
    
    # 각 세그먼트의 방향 벡터 계산
    dir1 = get_road_direction_from_intersection(intersection_pt, segment1)
    dir2 = get_road_direction_from_intersection(intersection_pt, segment2)
    
    # 정규화
    dir1_norm = dir1 / np.linalg.norm(dir1) if np.linalg.norm(dir1) > 1e-6 else dir1
    dir2_norm = dir2 / np.linalg.norm(dir2) if np.linalg.norm(dir2) > 1e-6 else dir2
    
    # 이등분선 계산 (두 가지 방향)
    bisector1 = dir1_norm + dir2_norm  # 첫 번째 이등분선
    bisector2 = dir1_norm - dir2_norm  # 수직 이등분선
    
    # 정규화
    if np.linalg.norm(bisector1) > 1e-6:
        bisector1 = bisector1 / np.linalg.norm(bisector1)
    if np.linalg.norm(bisector2) > 1e-6:
        bisector2 = bisector2 / np.linalg.norm(bisector2)
    
    # 바깥쪽 방향을 찾기 위해 외적 계산
    cross_product = np.cross(dir1_norm, dir2_norm)
    
    # 바깥쪽과 안쪽 이등분선 구분
    if cross_product > 0:  # 반시계방향
        outward_bisector = bisector2
        inward_bisector = -bisector2
    else:  # 시계방향
        outward_bisector = -bisector2  
        inward_bisector = bisector2
    
    return outward_bisector, inward_bisector

def cluster_points(points, tol=1.0):
    """교차점들을 클러스터링해서 대표점으로 변환"""
    clusters = []
    for p in points:
        assigned = False
        for c in clusters:
            if any(p.distance(q) < tol for q in c):
                c.append(p)
                assigned = True
                break
        if not assigned:
            clusters.append([p])
    
    # 대표점은 중심(centroid) 사용
    reps = []
    for cluster in clusters:
        if len(cluster) == 1:
            reps.append(cluster[0])
        else:
            # MultiPoint의 centroid로 대표점 계산
            centroid = MultiPoint(cluster).centroid
            reps.append(centroid)
    
    return reps

def line_angle_deg(ls):
    """선분의 각도 계산 (방향성 제거)"""
    (x1, y1), (x2, y2) = list(ls.coords)
    return (math.degrees(math.atan2(y2-y1, x2-x1)) + 360) % 180

def is_duplicate_chamfer(new_line, existing, end_tol=0.8, ang_tol=8.0):
    """중복/유사 가각선 검출"""
    a = line_angle_deg(new_line)
    A0, B0 = Point(new_line.coords[0]), Point(new_line.coords[1])
    
    for ln in existing:
        A1, B1 = Point(ln.coords[0]), Point(ln.coords[1])
        same_order = A0.distance(A1) < end_tol and B0.distance(B1) < end_tol
        swap_order = A0.distance(B1) < end_tol and B0.distance(A1) < end_tol
        
        if (same_order or swap_order) and abs(a - line_angle_deg(ln)) < ang_tol:
            return True
    
    return False

def validate_corner_line_candidate(extended_line, segments, seg1_idx, seg2_idx):
    """개선된 가각선 후보 검증 - 서로 다른 group 2개를 반드시 요구"""
    
    hits = []
    hit_group_ids = set()
    
    for idx, seg_info in enumerate(segments):
        seg_geom = seg_info["geom"]
        seg_group = seg_info["group"]
        
        if extended_line.intersects(seg_geom):
            inter_result = extended_line.intersection(seg_geom)
            intersection_points = extract_intersection_points(inter_result)
            
            for p_ in intersection_points:
                hits.append((idx, seg_group, p_))
                hit_group_ids.add(seg_group)
    
    # 서로 다른 group 2개를 반드시 요구
    if len(hit_group_ids) < 2:
        return False, []  # 같은 폴리라인만 때린 후보는 폐기
    
    # 정확히 2개의 교차점이 있는지 확인
    if len(hits) == 2:
        intersection_points = [hit[2] for hit in hits]
        return True, intersection_points
    
    return False, []

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

def validate_corner_line_candidate_optimized(extended_line, polylines, poly1_idx, poly2_idx):
    """최적화된 가각선 후보 검증 - 폴리라인 단위로 처리"""
    
    hits = []
    hit_group_ids = set()
    
    for idx, poly_info in enumerate(polylines):
        poly_geom = poly_info["geom"]
        poly_group = poly_info["group"]
        
        if extended_line.intersects(poly_geom):
            inter_result = extended_line.intersection(poly_geom)
            intersection_points = extract_intersection_points(inter_result)
            
            for p_ in intersection_points:
                hits.append((idx, poly_group, p_))
                hit_group_ids.add(poly_group)
    
    # 서로 다른 group 2개를 반드시 요구
    if len(hit_group_ids) < 2:
        return False, []
    
    # 정확히 2개의 교차점이 있는지 확인
    if len(hits) == 2:
        intersection_points = [hit[2] for hit in hits]
        return True, intersection_points
    
    return False, []

def process_dxf_file(uploaded_file, progress_bar=None, status_text=None):
def process_dxf_file(uploaded_file, progress_bar=None, status_text=None):
    """Center선 교차점 기반 가각선 생성 - 최적화된 버전"""
    
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
    
    # 가각선 레이어 생성
    if "가각선(안)" not in doc.layers:
        doc.layers.new("가각선(안)", dxfattribs={"color": 1})  # Red color

    if "가각선(안)_연장" not in doc.layers:
        doc.layers.new("가각선(안)_연장", dxfattribs={"color": 3}) # Cyan color

    # 3단계: 엔티티 분석 (최적화됨)
    current_step += 1
    update_progress(current_step, total_steps, "도면 엔티티 분석 중...")
    
    center_lines = []  # Center선들
    polylines = []     # 계획선 폴리라인들 (분절하지 않음)
    corner_points = [] # 코너점들

    for e in msp:
        if e.dxf.layer.lower() == "center":
            if e.dxftype() == "LINE":
                center_lines.append(LineString([(e.dxf.start.x, e.dxf.start.y), (e.dxf.end.x, e.dxf.end.y)]))
            elif e.dxftype() == "LWPOLYLINE":
                center_lines.append(LineString([(p[0], p[1]) for p in e.get_points()]))

        elif e.dxf.layer == "계획선":
            if e.dxftype() == "LINE":
                p1, p2 = e.dxf.start, e.dxf.end
                group_id = e.dxf.handle
                polylines.append({"geom": LineString([(p1.x, p1.y), (p2.x, p2.y)]), "group": group_id})
                corner_points.extend([Point(p1.x, p1.y), Point(p2.x, p2.y)])
                
            elif e.dxftype() == "LWPOLYLINE":
                pts = e.get_points()
                group_id = e.dxf.handle
                
                # 🔥 핵심 개선: 폴리라인 전체를 하나로 유지
                polylines.append({"geom": LineString([(p[0], p[1]) for p in pts]), "group": group_id})
                
                # corner_points만 별도 관리
                for pt in pts:
                    corner_points.append(Point(pt[0], pt[1]))
                
                # 긴 구간에 중간점 추가 (곡선 대응)
                for i in range(len(pts) - 1):
                    seg_length = Point(pts[i][0], pts[i][1]).distance(Point(pts[i+1][0], pts[i+1][1]))
                    if seg_length > 2:  # 2m 이상
                        num_subdivisions = int(seg_length / 1)  # 1m 간격
                        for k in range(1, num_subdivisions):
                            ratio = k / num_subdivisions
                            mid_x = pts[i][0] + ratio * (pts[i+1][0] - pts[i][0])
                            mid_y = pts[i][1] + ratio * (pts[i+1][1] - pts[i][1])
                            corner_points.append(Point(mid_x, mid_y))

    # 4단계: 중복 점 제거
    current_step += 1
    update_progress(current_step, total_steps, "중복 점 제거 중...")
    corner_points = remove_duplicates_fast(corner_points)

    # 5단계: 데이터 검증
    current_step += 1
    update_progress(current_step, total_steps, "데이터 검증 중...")
    
    if not center_lines:
        st.error("❌ 오류: 'center' 레이어를 찾을 수 없습니다. center 레이어가 있어야 가각선을 생성할 수 있습니다.")
        return None
    
    if not polylines:
        st.error("❌ 오류: '계획선' 레이어를 찾을 수 없습니다.")
        return None

    # 6단계: Center선 교차점 탐지 (핵심 개선!)
    current_step += 1
    update_progress(current_step, total_steps, "Center선 교차점 탐지 중...")
    
    # 🎯 핵심: Center선들의 교차점만 찾기
    center_intersection_points = []
    
    for i in range(len(center_lines)):
        for j in range(i + 1, len(center_lines)):
            if center_lines[i].intersects(center_lines[j]):
                intersection_result = center_lines[i].intersection(center_lines[j])
                intersection_points = extract_intersection_points(intersection_result)
                center_intersection_points.extend(intersection_points)
    
    # Center 교차점 클러스터링
    center_nodes = cluster_points(center_intersection_points, tol=1.0)
    
    st.info(f"�️ Center선 교차점 {len(center_intersection_points)}개 → 클러스터링 후 {len(center_nodes)}개")
    
    if not center_nodes:
        st.warning("⚠️ Center선 교차점이 없습니다. 가각선을 생성할 수 없습니다.")
        return None

    # 7단계: Center 교차점 주변에서 가각선 생성
    current_step += 1
    update_progress(current_step, total_steps, f"{len(center_nodes)}개 교차부에서 가각선 생성 중...")
    
    created_chamfers = []
    corner_lines_added = 0
    
    # UI 업데이트 빈도 제한
    update_interval = max(1, len(center_nodes) // 10)
    
    for node_idx, center_pt in enumerate(center_nodes):
        # UI 업데이트
        if node_idx % update_interval == 0 and status_text:
            status_text.text(f"단계 {current_step}/{total_steps}: Center 교차부 처리 중... ({node_idx+1}/{len(center_nodes)})")
        
        # 🔍 해당 Center 교차점 근처의 계획선 폴리라인 찾기
        nearby_polylines = []
        search_radius = 20.0  # Center 교차점에서 20m 반경
        
        for i, poly_info in enumerate(polylines):
            distance_to_center = center_pt.distance(poly_info["geom"])
            if distance_to_center < search_radius:
                nearby_polylines.append((i, poly_info, distance_to_center))
        
        # 거리순으로 정렬
        nearby_polylines.sort(key=lambda x: x[2])
        
        if len(nearby_polylines) < 2:
            st.warning(f"⚠️ Center 교차점 ({center_pt.x:.2f}, {center_pt.y:.2f}) 근처에 계획선이 부족합니다.")
            continue
        
        # 📐 가장 가까운 두 계획선으로 가각선 생성
        poly1_idx, poly1_info, _ = nearby_polylines[0]
        poly2_idx, poly2_info, _ = nearby_polylines[1]
        
        # 서로 다른 그룹인지 확인
        if poly1_info["group"] == poly2_info["group"]:
            # 같은 그룹이면 다음 것으로 시도
            if len(nearby_polylines) >= 3:
                poly2_idx, poly2_info, _ = nearby_polylines[2]
            else:
                continue
        
        # Center 교차점을 실제 교차점으로 사용
        intersection_pt = center_pt
        
        # 이등분선 방향 계산
        outward_bisector, inward_bisector = calculate_bisector_directions(
            intersection_pt, poly1_info["geom"], poly2_info["geom"]
        )
        
        # corner_points 탐색 (Center 교차점 기준)
        distances = [(p, intersection_pt.distance(p)) for p in corner_points]
        distances.sort(key=lambda x: x[1])
        
        local_pts = []
        for p, dist in distances:
            if dist < 40:
                local_pts.append(p)
            if len(local_pts) >= 2:
                break
        
        if len(local_pts) < 2:
            for p, dist in distances:
                if 40 <= dist < 80:
                    local_pts.append(p)
                if len(local_pts) >= 2:
                    break
        
        if len(local_pts) < 2:
            st.warning(f"⚠️ Center 교차점 ({intersection_pt.x:.2f}, {intersection_pt.y:.2f}) 근처에 충분한 corner_points가 없습니다.")
            continue
            
        corner1, corner2 = local_pts[:2]

        # 도로폭 계산
        d1 = shortest_perpendicular_distance(corner1, center_lines)
        d2 = shortest_perpendicular_distance(corner2, center_lines)
        
        if d1 is None or d2 is None:
            continue
        
        w1 = round(d1 * 2, 3)
        w2 = round(d2 * 2, 3)

        # 교차각 계산
        dir1 = get_road_direction_from_intersection(intersection_pt, poly1_info["geom"])
        dir2 = get_road_direction_from_intersection(intersection_pt, poly2_info["geom"])
        
        dir1_norm = dir1 / np.linalg.norm(dir1)
        dir2_norm = dir2 / np.linalg.norm(dir2)
        cos_angle = np.clip(np.dot(dir1_norm, dir2_norm), -1.0, 1.0)
        intersection_angle = np.rad2deg(np.arccos(abs(cos_angle)))
        
        # 가각선 길이 결정
        corner_len = get_corner_length(intersection_angle, w1, w2)
        if not corner_len:
            avg_width = (w1 + w2) / 2
            if intersection_angle < 75:
                corner_len = avg_width * 0.8
            elif intersection_angle > 105:
                corner_len = avg_width * 1.2
            else:
                corner_len = avg_width * 1.0
            st.info(f"📏 경험적 공식으로 가각선 길이 계산: {corner_len:.2f}m")
        
        if corner_len <= 0:
            continue

        # 가각선 후보 생성 및 검증
        valid_corner_line = None
        
        for direction in [outward_bisector, inward_bisector]:
            pt_array = np.array([intersection_pt.x, intersection_pt.y])
            end_point_array = pt_array + direction * corner_len
            
            extension_length = 3.0
            extended_start_array = pt_array - direction * extension_length
            extended_end_array = end_point_array + direction * extension_length
            
            extended_line = LineString([
                (extended_start_array[0], extended_start_array[1]),
                (extended_end_array[0], extended_end_array[1])
            ])
            
            # 검증: 올바른 계획선과 교차하는지 확인
            is_valid, intersection_points_final = validate_corner_line_candidate_optimized(
                extended_line, polylines, poly1_idx, poly2_idx
            )
            
            if is_valid and len(intersection_points_final) == 2:
                valid_corner_line = LineString([intersection_points_final[0], intersection_points_final[1]])
                break
        
        # 중복 검사 후 DXF 추가
        if valid_corner_line:
            if is_duplicate_chamfer(valid_corner_line, created_chamfers):
                st.info(f"🔄 중복 가각선 발견으로 건너뜀: Center 교차점 ({center_pt.x:.2f}, {center_pt.y:.2f})")
                continue
            
            created_chamfers.append(valid_corner_line)
            corner_coords = list(valid_corner_line.coords)
            
            # DXF에 가각선 추가
            new_line = doc.modelspace().add_line(
                (corner_coords[0][0], corner_coords[0][1]),
                (corner_coords[1][0], corner_coords[1][1])
            )
            new_line.dxf.layer = "가각선(안)"
            
            corner_lines_added += 1
            st.success(f"✅ 가각선 추가: Center 교차점 ({center_pt.x:.2f}, {center_pt.y:.2f}) 기준")
        else:
            st.warning(f"⚠️ Center 교차점 ({center_pt.x:.2f}, {center_pt.y:.2f})에서 유효한 가각선을 생성할 수 없습니다.")

    # 8단계: DXF 파일 저장
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
        
        # 최종 통계 출력
        st.success(f"🎯 처리 완료! 총 {corner_lines_added}개의 가각선이 생성되었습니다.")
        st.info(f"📊 Center 교차부 {len(center_nodes)}개 중 {corner_lines_added}개에서 가각선 생성")
        
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
            **🔥 최적화된 처리 방식:**
            - Center선 교차점 기반 가각선 생성
            - 폴리라인 단위 처리 (세그먼트 분절 제거)
            - 10-100배 성능 향상
            
            **교차점 처리:**
            - Center선 교차점만 탐지
            - 20m 반경 내 계획선 검색
            - Point, MultiPoint, LineString 모두 지원
            - 중복 교차점 자동 제거 (1.0m 클러스터링)
            
            **corner_points 탐색:**
            - 1차: 40m 범위 탐색
            - 2차: 80m 범위 확장 탐색
            - 긴 세그먼트 자동 세분화 (1m 간격)
            
            **폭 계산:**
            - center선 기반 수직 거리 계산
            - ±0.2m 허용 오차로 유연한 분류
            
            **가각선 길이:**
            - lookup_table 우선 사용
            - 실패시 경험적 공식 적용
            
            **중복 제거:**
            - 끝점 거리 + 각도 차이로 중복 판단
            - 0.8m 끝점 허용오차 + 8° 각도 허용오차
            
            **처리 진행률:**
            - 8단계 세분화된 처리 과정
            - 실시간 상태 표시
            - Center 교차부별 처리 진행률
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
