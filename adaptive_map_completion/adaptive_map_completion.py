from tptk.common.spatial_func import distance, project_pt_to_segment, cal_loc_along_line, SPoint, LAT_PER_METER, LNG_PER_METER, angle, project_pt_to_line
from tptk.common.mbr import MBR
from tptk.common.road_network import load_rn_shp, store_rn_shp
import copy
import networkx as nx
from rtree import index, Rtree
from tptk.common.road_network import UndirRoadNetwork
import math
from walkway_completion.mc_utils import DistanceSegmentation, make_directed_rn_undirected
import os
from tqdm import tqdm
from tptk.common.trajectory import parse_traj_file


class SplitOP:
    def __init__(self, end_node, target_segment, split_edge_idx, split_edge_offset):
        self.end_node = end_node
        self.target_segment = target_segment
        self.split_edge_idx = split_edge_idx
        self.split_edge_offset = split_edge_offset

    def __repr__(self):
        return '(index:{},offset:{})'.format(self.split_edge_idx, self.split_edge_offset)


def compress_rn(raw_rn):
    print('raw rn #nodes:{}'.format(nx.number_of_nodes(raw_rn)))
    print('raw rn #edges:{}'.format(nx.number_of_edges(raw_rn)))
    compressed_rn = copy.deepcopy(raw_rn)
    modify_ops = []
    # 会不会有正着插入一遍，反着插入一遍，有两个modify_ops的情况？-> line 80 to make sure
    for node, degree in raw_rn.degree():
        if degree >= 3:
            roads = get_all_road_segments(node, raw_rn)
            for road in roads:
                if len(road) > 2:
                    modify_ops.append(road)
    print('nb modification roads:{}'.format(len(modify_ops)))
    for road in modify_ops:
        if not compressed_rn.has_edge(road[0], road[-1]):
            add_new_edge(road, 0, len(road) - 1, compressed_rn)
        else:
            # if there is already a road segment between new_start_node and new_end_node,
            # we split the new edge to create two edges
            mid_idx = int(len(road) / 2.0)
            add_new_edge(road, 0, mid_idx, compressed_rn)
            add_new_edge(road, mid_idx, len(road) - 1, compressed_rn)
    compressed_rn.remove_nodes_from(list(nx.isolates(compressed_rn)))
    print('compressed rn #nodes:{}'.format(nx.number_of_nodes(compressed_rn)))
    print('compressed rn #edges:{}'.format(nx.number_of_edges(compressed_rn)))
    return compressed_rn


def add_new_edge(nodes, start_idx, end_idx, g):
    # end_idx is inclusive
    start_node = nodes[start_idx]
    end_node = nodes[end_idx]
    if not g.has_edge(nodes[start_idx], nodes[start_idx+1]):
        return
    first_edge = g[start_node][nodes[start_idx+1]]
    for i in range(start_idx, end_idx):
        if g.has_edge(nodes[i], nodes[i+1]):
            g.remove_edge(nodes[i], nodes[i+1])
    coords = []
    for pt_arr in nodes[start_idx:end_idx+1]:
        coords.append(SPoint(pt_arr[1], pt_arr[0]))
    g.add_edge(start_node, end_node, eid=first_edge['eid'], coords=coords)


def get_all_road_segments(int_node, g):
    all_road_segments = []
    for u, v in g.edges(int_node):
        first_adj_node = u if u != int_node else v
        # make sure a road would not be inserted for twice, this is achieved by only adding direction match edges
        if (g[int_node][first_adj_node]['coords'][0].lng, g[int_node][first_adj_node]['coords'][0].lat) == int_node and \
                (g[int_node][first_adj_node]['coords'][-1].lng, g[int_node][first_adj_node]['coords'][-1].lat) == first_adj_node:
            road_segment = construct_road_segment(first_adj_node, g, [int_node, first_adj_node])
            all_road_segments.append(road_segment)
    return all_road_segments


def construct_road_segment(first_adj_node, g, seq):
    cur_node = first_adj_node
    while g.degree(cur_node) == 2:
        pre_node = seq[-2]
        for u, v in g.edges(cur_node):
            if u != pre_node and u != cur_node:
                next_node = u
                break
            elif v != pre_node and v != cur_node:
                next_node = v
                break
        seq.append(next_node)
        cur_node = next_node
    return seq


def densify_rn(rn, edge_split_dist=9999):
    # currently, only support undirected rn
    g = nx.Graph()
    edge_idx = {}
    edge_spatial_idx = Rtree()
    avail_eid = 0
    for u, v, data in rn.edges(data=True):
        coords = copy.deepcopy(data['coords'])
        for i in range(len(coords) - 1):
            consecutive_dist = distance(coords[i], coords[i+1])
            nb_splitted_seg = math.ceil(consecutive_dist / edge_split_dist)
            if nb_splitted_seg == 1:
                g.add_edge((coords[i].lng, coords[i].lat), (coords[i+1].lng, coords[i+1].lat),
                           coords=coords[i:i+2], eid=avail_eid)
                mbr = MBR.cal_mbr([coords[i], coords[i+1]])
                edge_spatial_idx.insert(avail_eid, (mbr.min_lng, mbr.min_lat, mbr.max_lng, mbr.max_lat))
                avail_eid += 1
            else:
                pos_seq = [coords[i]]
                for j in range(1, nb_splitted_seg):
                    pos_seq.append(cal_loc_along_line(coords[i], coords[i+1], j/nb_splitted_seg))
                pos_seq.append(coords[i+1])
                for j in range(len(pos_seq)-1):
                    g.add_edge((pos_seq[j].lng, pos_seq[j].lat), (pos_seq[j+1].lng, pos_seq[j+1].lat),
                               coords=pos_seq[j:j+2], eid=avail_eid)
                    mbr = MBR.cal_mbr([pos_seq[j], pos_seq[j+1]])
                    edge_spatial_idx.insert(avail_eid, (mbr.min_lng, mbr.min_lat, mbr.max_lng, mbr.max_lat))
                    avail_eid += 1
    return UndirRoadNetwork(g, edge_spatial_idx, edge_idx)


class DelvMapConnector:
    def __init__(self, out_compressed=True, min_trans_cnt=2):
        self.NEW_NODE_THRESH = 32
        self.CLOSE_EDGE_THRESH = 10
        self.INF_RN_NODE_GAP = 50
        self.TRANS_SUPP_THRESH = min_trans_cnt
        self.CHECK_RADIUS = 30
        self.TRAJ_MATCH_THRESH = 15
        self.out_compressed = out_compressed

    def adaptive_fuse(self, existing_rn, inferred_rn, trajs):
        traj_index, rtid2t = index_trajs(trajs)
        avail_eid = max([eid for u, v, eid in existing_rn.edges.data(data='eid')]) + 1
        complete_rn = copy.deepcopy(existing_rn)
        com_edge_spatial_idx = Rtree()
        for u, v, data in existing_rn.edges(data=True):
            mbr = MBR.cal_mbr(data['coords'])
            com_edge_spatial_idx.insert(data['eid'], (mbr.min_lng, mbr.min_lat, mbr.max_lng, mbr.max_lat))
        complete_rn.edge_spatial_idx = com_edge_spatial_idx

        inf_coord2com_coord = {}
        densified_inferred_rn = densify_rn(inferred_rn, edge_split_dist=self.INF_RN_NODE_GAP)

        for node in tqdm(densified_inferred_rn):
            if node in inf_coord2com_coord:
                continue
            if self.cal_closest_proj_w_thresh(SPoint(node[1], node[0]), existing_rn, self.NEW_NODE_THRESH) is not None:
                continue
            inf_coord2com_coord[node] = node
            q = [node]
            cur_new_edges = []
            while len(q) > 0:
                cur = q[-1]
                cur_com_pt = SPoint(inf_coord2com_coord[cur][1], inf_coord2com_coord[cur][0])
                q = q[:-1]
                for u, v in densified_inferred_rn.edges(cur):
                    tgt = u if u != cur else v
                    if tgt in inf_coord2com_coord:
                        if not complete_rn.has_edge(inf_coord2com_coord[cur], inf_coord2com_coord[tgt]):
                            complete_rn.add_edge(inf_coord2com_coord[cur], inf_coord2com_coord[tgt],
                                                 coords=[cur_com_pt, SPoint(inf_coord2com_coord[tgt][1],
                                                                            inf_coord2com_coord[tgt][0])],
                                                 eid=avail_eid)
                            avail_eid += 1
                            cur_new_edges.append((inf_coord2com_coord[cur], inf_coord2com_coord[tgt]))
                        continue
                    candi_projs_in_base = self.cal_candi_projs_w_thresh(SPoint(tgt[1], tgt[0]), existing_rn, self.CLOSE_EDGE_THRESH)
                    if len(candi_projs_in_base) == 0:
                        inf_coord2com_coord[tgt] = tgt
                        complete_rn.add_edge(inf_coord2com_coord[cur], inf_coord2com_coord[tgt],
                                             coords=[cur_com_pt,
                                                     SPoint(inf_coord2com_coord[tgt][1], inf_coord2com_coord[tgt][0])],
                                             eid=avail_eid)
                        avail_eid += 1
                        cur_new_edges.append((inf_coord2com_coord[cur], inf_coord2com_coord[tgt]))
                        q.append(tgt)
                        continue
                    if len(candi_projs_in_base) == 1:
                        closest_proj_in_base = min(candi_projs_in_base, key=lambda x: x[0][0])
                        (_, edge_idx_in_base, edge_offset_rate_in_base), closest_edge_key_in_base = closest_proj_in_base
                        select_proj_pt_in_base, _ = self.cal_absoluate_pt(edge_idx_in_base, edge_offset_rate_in_base,
                                                                          closest_edge_key_in_base, existing_rn)
                    else:
                        max_trans_proj_in_base, trans_cnt = self.get_max_trans_edge(candi_projs_in_base, tgt, cur, traj_index, rtid2t, existing_rn)
                        if trans_cnt >= self.TRANS_SUPP_THRESH:
                            # P1: trajectory-based connection
                            (_, edge_idx_in_base, edge_offset_rate_in_base), max_trans_edge_key_in_base = max_trans_proj_in_base
                            select_proj_pt_in_base, _ = self.cal_absoluate_pt(edge_idx_in_base, edge_offset_rate_in_base,
                                                                           max_trans_edge_key_in_base, existing_rn)
                        else:
                            # P2: spatial-proximity-based connection
                            closest_proj_in_base = min(candi_projs_in_base, key=lambda x: x[0][0])
                            (_, edge_idx_in_base, edge_offset_rate_in_base), closest_edge_key_in_base = closest_proj_in_base
                            select_proj_pt_in_base, _ = self.cal_absoluate_pt(edge_idx_in_base, edge_offset_rate_in_base,
                                                                               closest_edge_key_in_base, existing_rn)
                    (_, edge_idx_in_com,
                     edge_offset_rate_in_com), select_edge_key_in_com = self.cal_closest_proj_w_thresh(
                        select_proj_pt_in_base, complete_rn, 500)
                    select_proj_pt_in_com, adjusted_edge_offset_rate_in_com = \
                        self.cal_absoluate_pt(edge_idx_in_com, edge_offset_rate_in_com, select_edge_key_in_com,
                                              complete_rn, end_fuse=False)
                    inf_coord2com_coord[tgt] = (select_proj_pt_in_com.lng, select_proj_pt_in_com.lat)
                    ori_coords = complete_rn[select_edge_key_in_com[0]][select_edge_key_in_com[1]]['coords']
                    if inf_coord2com_coord[tgt] not in [select_edge_key_in_com[0], select_edge_key_in_com[1]]:
                        complete_rn.remove_edge(select_edge_key_in_com[0], select_edge_key_in_com[1])
                        left_coords = []
                        right_coords = []
                        if adjusted_edge_offset_rate_in_com != 0.0 and adjusted_edge_offset_rate_in_com != 1.0:
                            for i in range(edge_idx_in_com + 1):
                                left_coords.append(ori_coords[i])
                            left_coords.append(select_proj_pt_in_com)
                            right_coords.append(select_proj_pt_in_com)
                            for i in range(edge_idx_in_com + 1, len(ori_coords)):
                                right_coords.append(ori_coords[i])
                        elif adjusted_edge_offset_rate_in_com == 0.0:
                            left_coords = ori_coords[:(edge_idx_in_com+1)]
                            right_coords = ori_coords[edge_idx_in_com:]
                        elif adjusted_edge_offset_rate_in_com == 1.0:
                            left_coords = ori_coords[:(edge_idx_in_com+2)]
                            right_coords = ori_coords[(edge_idx_in_com + 1):]
                        if (left_coords[0].lng, left_coords[0].lat) == (right_coords[-1].lng, right_coords[-1].lat):
                            split_left = True if len(left_coords) > 2 else False
                            if split_left:
                                mid_idx = int(len(left_coords) / 2.0)
                                left_mid_tuple = left_coords[mid_idx].lng, left_coords[mid_idx].lat
                                complete_rn.add_edge((left_coords[0].lng, left_coords[0].lat), left_mid_tuple,
                                                     coords=left_coords[:mid_idx+1], eid=avail_eid)
                                avail_eid += 1
                                complete_rn.add_edge(left_mid_tuple, inf_coord2com_coord[tgt],
                                                     coords=left_coords[mid_idx:], eid=avail_eid)
                                avail_eid += 1
                                complete_rn.add_edge(inf_coord2com_coord[tgt],
                                                     (right_coords[-1].lng, right_coords[-1].lat),
                                                     coords=right_coords, eid=avail_eid)
                                avail_eid += 1
                            else:
                                complete_rn.add_edge((left_coords[0].lng, left_coords[0].lat), inf_coord2com_coord[tgt],
                                                     coords=left_coords, eid=avail_eid)
                                avail_eid += 1
                                mid_idx = int(len(right_coords) / 2.0)
                                right_mid_tuple = right_coords[mid_idx].lng, right_coords[mid_idx].lat
                                complete_rn.add_edge(inf_coord2com_coord[tgt],
                                                     right_mid_tuple, coords=right_coords[:mid_idx+1], eid=avail_eid)
                                avail_eid += 1
                                complete_rn.add_edge(right_mid_tuple,
                                                     (right_coords[-1].lng, right_coords[-1].lat), coords=right_coords[mid_idx:], eid=avail_eid)
                                avail_eid += 1
                        else:
                            complete_rn.add_edge((left_coords[0].lng, left_coords[0].lat), inf_coord2com_coord[tgt],
                                                 coords=left_coords, eid=avail_eid)
                            avail_eid += 1
                            complete_rn.add_edge(inf_coord2com_coord[tgt], (right_coords[-1].lng, right_coords[-1].lat),
                                                 coords=right_coords, eid=avail_eid)
                            avail_eid += 1
                    #  Step 4. 新路段插入到complete map中
                    if not complete_rn.has_edge(inf_coord2com_coord[cur], inf_coord2com_coord[tgt]):
                        complete_rn.add_edge(inf_coord2com_coord[cur], inf_coord2com_coord[tgt],
                                             coords=[cur_com_pt, select_proj_pt_in_com], eid=avail_eid)
                        avail_eid += 1
                        cur_new_edges.append((inf_coord2com_coord[cur], inf_coord2com_coord[tgt]))
        if self.out_compressed:
            complete_rn_compressed = compress_rn(complete_rn)
            return complete_rn_compressed
        else:
            return complete_rn

    def get_max_trans_edge(self, candi_projs, v, oppo_v, traj_index, rtid2t, rn):
        """
        candi_projs: list of (dist, edge_idx, edge_offset_rate), edge_key 必定非空
        v: the adjacent vertex to be connected
        oppo_v: the far vertex
        traj_index: trajectory rtree
        """
        edge_key2trans_cnt = {}
        for _, edge_key in candi_projs:
            edge_key2trans_cnt[edge_key] = 0
        edge_key2candi_proj = {}
        for candi_proj in candi_projs:
            edge_key2candi_proj[candi_proj[1]] = candi_proj
        v_pt = SPoint(v[1], v[0])
        oppo_v_pt = SPoint(oppo_v[1], oppo_v[0])
        traj_query_mbr = MBR(v_pt.lat - self.CHECK_RADIUS * LAT_PER_METER,
                             v_pt.lng - self.CHECK_RADIUS * LNG_PER_METER,
                             v_pt.lat + self.CHECK_RADIUS * LAT_PER_METER,
                             v_pt.lng + self.CHECK_RADIUS * LNG_PER_METER)
        rtids = traj_index.intersection((traj_query_mbr.min_lng, traj_query_mbr.min_lat,
                                         traj_query_mbr.max_lng, traj_query_mbr.max_lat))
        close_trajs = [rtid2t[rtid] for rtid in rtids]
        max_trans_proj = None
        max_cnt = 0
        if len(close_trajs) == 0:
            return max_trans_proj, max_cnt
        # [(distance(t_pt, v_pt), pt_idx) for traj in close_trajs for t_pt in traj.pt_list]
        for traj in close_trajs:
            nb_pts = len(traj.pt_list)
            dist_with_idx = [(distance(t_pt, v_pt), pt_idx) for pt_idx, t_pt in enumerate(traj.pt_list)]
            # 定位距离v最近的点
            closest_t_pt_dist, closest_t_pt_idx = min(dist_with_idx, key=lambda x: x[0])
            if closest_t_pt_dist > self.CHECK_RADIUS:
                continue
            # 统计该点之前恰好R米开外的点
            pre_pt = None
            for i in range(closest_t_pt_idx - 1, -1, -1):
                if dist_with_idx[i][0] > self.CHECK_RADIUS:
                    pre_pt = traj.pt_list[i]
                    break
            # 统计该点之后恰好R米开外的点
            nxt_pt = None
            for i in range(closest_t_pt_idx + 1, nb_pts):
                if dist_with_idx[i][0] > self.CHECK_RADIUS:
                    nxt_pt = traj.pt_list[i]
                    break
            if pre_pt is None or nxt_pt is None:
                continue
            pre_pt_new_edge_dist, nxt_pt_new_edge_dist = perpendicular_intersection(pre_pt, [v_pt, oppo_v_pt])[0], perpendicular_intersection(nxt_pt, [v_pt, oppo_v_pt])[0]
            if pre_pt_new_edge_dist > nxt_pt_new_edge_dist and nxt_pt_new_edge_dist < self.TRAJ_MATCH_THRESH:
                to_check_pt = pre_pt
            elif pre_pt_new_edge_dist < nxt_pt_new_edge_dist and pre_pt_new_edge_dist < self.TRAJ_MATCH_THRESH:
                to_check_pt = nxt_pt
            else:
                continue
            to_check_pt_min_dist, min_dist_edge_key = min([(perpendicular_intersection(to_check_pt, rn[edge_key[0]][edge_key[1]]['coords'])[0], edge_key) for _, edge_key in candi_projs],
                                                          key=lambda x: x[0])
            if to_check_pt_min_dist > self.TRAJ_MATCH_THRESH:
                continue
            edge_key2trans_cnt[min_dist_edge_key] += 1
        max_trans_edge_key = max(edge_key2trans_cnt, key=edge_key2trans_cnt.get)
        return edge_key2candi_proj[max_trans_edge_key], edge_key2trans_cnt[max_trans_edge_key]

    def cal_closest_proj_w_thresh(self, pt, rn, radius):
        close_edge_query_mbr = MBR(pt.lat - radius * LAT_PER_METER,
                                   pt.lng - radius * LNG_PER_METER,
                                   pt.lat + radius * LAT_PER_METER,
                                   pt.lng + radius * LNG_PER_METER)
        candi_projs = [(perpendicular_intersection(pt, rn[edge_key[0]][edge_key[1]]['coords']), edge_key) for edge_key in rn.range_query(close_edge_query_mbr)]
        candi_projs = [candi_intersect for candi_intersect in candi_projs if candi_intersect[0][0] < radius]
        closest_proj = min(candi_projs, key=lambda x: x[0][0]) if len(candi_projs) > 0 else None
        return closest_proj

    def cal_candi_projs_w_thresh(self, pt, rn, radius):
        close_edge_query_mbr = MBR(pt.lat - radius * LAT_PER_METER,
                                   pt.lng - radius * LNG_PER_METER,
                                   pt.lat + radius * LAT_PER_METER,
                                   pt.lng + radius * LNG_PER_METER)
        candi_projs = [(perpendicular_intersection(pt, rn[edge_key[0]][edge_key[1]]['coords']), edge_key) for edge_key in rn.range_query(close_edge_query_mbr)]
        candi_projs = [candi_intersect for candi_intersect in candi_projs if candi_intersect[0][0] < radius]
        return candi_projs

    def cal_absoluate_pt(self, edge_idx, edge_offset_rate, edge_key, rn, end_fuse=False):
        u, v = edge_key
        edge_coords = rn[u][v]['coords']
        real_abs_pt = cal_loc_along_line(edge_coords[edge_idx], edge_coords[edge_idx + 1], edge_offset_rate)
        if not end_fuse:
            return real_abs_pt, edge_offset_rate
        else:
            from_pt, to_pt = edge_coords[edge_idx], edge_coords[edge_idx + 1]
            adjusted_abs_pt, adjusted_edge_offset_rate = real_abs_pt, edge_offset_rate
            if distance(from_pt, real_abs_pt) < 5:
                adjusted_abs_pt = from_pt
                adjusted_edge_offset_rate = 0.0
            elif distance(to_pt, real_abs_pt) < 5:
                adjusted_abs_pt = to_pt
                adjusted_edge_offset_rate = 1.0
            return adjusted_abs_pt, adjusted_edge_offset_rate


def perpendicular_intersection(pt, target_coords):
    (_, split_edge_offset, min_dist), split_edge_idx = min(
        [(project_pt_to_segment(target_coords[i], target_coords[i + 1], pt), i)
         for i in range(len(target_coords) - 1)], key=lambda x: x[0][2])
    return min_dist, split_edge_idx, split_edge_offset


def index_trajs(trajs):
    rtid = 0
    rtid2t = {}
    data = []
    for t in trajs:
        mbr_t = t.get_mbr()
        t_mbr_tuple = (mbr_t.min_lng, mbr_t.min_lat, mbr_t.max_lng, mbr_t.max_lat)
        data.append((rtid, t_mbr_tuple, None))
        rtid2t[rtid] = t
        rtid += 1
    rt = index.Index(data)
    return rt, rtid2t


def obtain_segmented_trajs(mm_traj_path, traj_seg_dist=20):
    dist_seg = DistanceSegmentation(max_dist=traj_seg_dist)
    min_traj_len = 10
    all_trajs = []
    for courier_id in [f for f in os.listdir(mm_traj_path) if not f.startswith('.')]:
        courier_dir = mm_traj_path + '{}/'.format(courier_id)
        for filename in tqdm([f for f in os.listdir(courier_dir) if not f.startswith('.')]):
            mm_trajs = parse_traj_file(courier_dir + filename, traj_type='mm')
            trajs_per_file_dist_seg = []
            for mm_traj in mm_trajs:
                segmented_trajs = dist_seg.segment(mm_traj)
                for i in range(len(segmented_trajs)):
                    t = segmented_trajs[i]
                    if t.get_length() <= min_traj_len:
                        continue
                    trajs_per_file_dist_seg.append(t)
            all_trajs.extend(trajs_per_file_dist_seg)
    return all_trajs


if __name__ == '__main__':
    # our implementation
    method2rn_filename = {
        'DelvMap': '',
    }

    method = 'DelvMap'

    sta_ids = ['']
    osm_base_dir = ''
    osm_rn_path_template = osm_base_dir + 'rn-osm-wgs/rn-{}-220920'
    mm_sta_dir_template = osm_base_dir + 'test/{}/'
    min_trans_cnt = 1
    mc = DelvMapConnector(out_compressed=True, min_trans_cnt=min_trans_cnt)
    traj_seg_dist = 20

    for sta_id in sta_ids:
        mm_sta_dir = mm_sta_dir_template.format(sta_id)
        segmented_trajs = obtain_segmented_trajs(mm_sta_dir, traj_seg_dist)
        existing_rn = load_rn_shp(osm_rn_path_template.format(sta_id))
        existing_rn, _ = make_directed_rn_undirected(existing_rn)
        inferred_rn = load_rn_shp('./baselines/{}/{}'.format(method, method2rn_filename[method]))
        inferred_rn, _ = make_directed_rn_undirected(inferred_rn)
        fused_rn = mc.adaptive_fuse(existing_rn, inferred_rn, segmented_trajs)
        store_rn_shp(fused_rn, './baselines/{}/{}_fused_s{}'.format(method,
                                                                    method2rn_filename[method],
                                                                    min_trans_cnt))
