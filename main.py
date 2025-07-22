from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import shutil
import numpy as np
import torch

import torch.nn.functional as F
from cotracker.models.core.cotracker.cotracker3_offline import CoTrackerThreeOffline
from cotracker.models.core.cotracker.losses import sequence_loss
from cotracker.utils.visualizer import Visualizer, read_video_from_path
from cotracker.predictor import CoTrackerPredictor

app = FastAPI()

# Allow frontend connection (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://localhost:3000"] for strict mode
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve uploaded videos from /uploads
UPLOAD_DIR = "uploads"
DEFAULT_DEVICE = (
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)
os.makedirs(UPLOAD_DIR, exist_ok=True)
video, video_H, video_W = None, None, None
fmaps_pyramid = None
model = CoTrackerThreeOffline(
                stride=4,
                corr_radius=3,
                window_len=60,
                model_resolution=(384, 512),
                linear_layer_for_vis_conf=True,
            )
ckpt = torch.load('./dependencies/cotracker3/checkpoints/scaled_offline.pth')
if "model" in ckpt:
    model.load_state_dict(ckpt["model"])
else:
    model.load_state_dict(ckpt)

for param in model.parameters():
    param.requires_grad = False
model = model.to(DEFAULT_DEVICE)
model.eval()
interp_shape = model.model_resolution

app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

def process_video(file_path):
    video = read_video_from_path(file_path)
    video = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float()
    B, T, C, H, W = video.shape
    video = video.reshape(B * T, C, H, W)
    video = F.interpolate(
        video, tuple(interp_shape), mode="bilinear", align_corners=True
    )
    video = video.reshape(B, T, 3, interp_shape[0], interp_shape[1])
    video = video.to(DEFAULT_DEVICE)
    
    return video, H, W

def extract_features(video):
    B, T, C, H, W = video.shape
    C_ = C
    fmaps_chunk_size = 200
    if T > fmaps_chunk_size:
        fmaps = []
        for t in range(0, T, fmaps_chunk_size):
            video_chunk = video[:, t : t + fmaps_chunk_size]
            fmaps_chunk = model.fnet(video_chunk.reshape(-1, C_, H, W))
            T_chunk = video_chunk.shape[1]
            C_chunk, H_chunk, W_chunk = fmaps_chunk.shape[1:]
            fmaps.append(fmaps_chunk.reshape(B, T_chunk, C_chunk, H_chunk, W_chunk))
        fmaps = torch.cat(fmaps, dim=1).reshape(-1, C_chunk, H_chunk, W_chunk)
    else:
        fmaps = model.fnet(video.reshape(-1, C_, H, W))

    fmaps = fmaps.permute(0, 2, 3, 1)
    fmaps = fmaps / torch.sqrt(
        torch.maximum(
            torch.sum(torch.square(fmaps), axis=-1, keepdims=True),
            torch.tensor(1e-12, device=fmaps.device),
        )
    )
    fmaps = fmaps.permute(0, 3, 1, 2).reshape(
        B, -1, 128, H // 4, W // 4
    )
    fmaps = fmaps.to(video.dtype)
    
    fmaps_pyramid = []
    fmaps_pyramid.append(fmaps)
    for i in range(4 - 1):
        fmaps_ = fmaps.reshape(
            B * T, 128, fmaps.shape[-2], fmaps.shape[-1]
        )
        fmaps_ = F.avg_pool2d(fmaps_, 2, stride=2)
        fmaps = fmaps_.reshape(
            B, T, 128, fmaps_.shape[-2], fmaps_.shape[-1]
        )
        fmaps_pyramid.append(fmaps)
    
    return fmaps_pyramid

def get_kp_feats(queries_all, fmaps_pyramid):
    track_feat_support_pyramid = []
    for id, queries_t in queries_all.items():
        queries_t = queries_t.cuda()
        
        queried_frames_t = queries_t[:, :, 0].long()
        queried_coords_t = queries_t[..., 1:]
        queried_coords_t = queried_coords_t / 4.0
        
        track_feat, track_feat_support = model.get_track_feat(
            fmaps_pyramid[0],
            queried_frames_t,
            queried_coords_t,
            support_radius=3,
        )
        # track_feat_pyramid.append(track_feat.repeat(1, T, 1, 1))
        track_feat_support_pyramid.append(track_feat_support.mean(dim=-2))

    track_feat_support_pyramid = torch.cat(track_feat_support_pyramid, dim=0).permute(1, 0, 2)[None, None, None]
    
    return track_feat_support_pyramid

@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    global video, video_H, video_W
    video, video_H, video_W = process_video(file_path)
    video_url = f"http://localhost:8000/uploads/{file.filename}"
    return JSONResponse({"url": video_url})
# ---------------------------
# Models
# ---------------------------
class Annotation(BaseModel):
    id: int
    keyframes: int
    details: str
    is_background: bool

class Keypoint(BaseModel):
    x: float
    y: float
    frame: int
    id: int
    isManual: bool

class TrackRequest(BaseModel):
    keypoints: List[Keypoint]
    method: str  # NEW
    iterations: Optional[int] = 500
    learning_rate: Optional[float] = 0.0001

# ---------------------------
# In-memory store
# ---------------------------
ANNOTATIONS = []
optimization_progress = {
    "current": 0,
    "total": 1,
    "in_progress": False
}

# ---------------------------
# Endpoints
# ---------------------------

@app.get("/annotations")
async def get_annotations():
    return ANNOTATIONS

@app.post("/annotations")
async def add_annotation(annotation: Annotation):
    ANNOTATIONS.append(annotation)
    return {"status": "ok"}

@app.delete("/annotations/{id}")
async def delete_annotation(id: int):
    global ANNOTATIONS
    ANNOTATIONS = [a for a in ANNOTATIONS if a.id != id]
    return {"status": "deleted"}

@app.delete("/annotations/")
async def delete_all_annotations():
    global ANNOTATIONS
    ANNOTATIONS = []
    return {"status": "deleted all"}

@app.get("/track/progress")
def get_progress():
    return optimization_progress

@app.post("/track")
async def track_keypoints(req: TrackRequest):
    global video, video_H, video_W
    if video is None:
        video, video_H, video_W = process_video("uploads/video.mp4")
    
    isManualDict = {}
    if req.method == "cotracker3":
        queries = []
        queries_gt = {}
        for kp in req.keypoints:
            scaled_x = kp.x / (video_W - 1) * (interp_shape[1] - 1)
            scaled_y = kp.y / (video_H - 1) * (interp_shape[0] - 1)
            if kp.frame == 0 or kp.isManual:
                if kp.id not in queries_gt:
                    queries_gt[kp.id] = []
                queries_gt[kp.id].append([kp.frame, scaled_x, scaled_y])
                isManualDict[f'{kp.id}_{kp.frame}'] = True
        sorted_ids = sorted(queries_gt.keys())
        for i in range(len(queries_gt)):
            id = sorted_ids[i]
            first_coord = sorted(queries_gt[id], key=lambda x: x[0])[0]
            queries.append(first_coord)
        queries = torch.tensor(queries)[None].to(DEFAULT_DEVICE)
        pred_tracks, pred_visibility, *_ = model.forward(
            video, queries, iters=6
        )
        pred_tracks *= pred_tracks.new_tensor(
            [(video_W - 1) / (interp_shape[1] - 1), (video_H - 1) / (interp_shape[0] - 1)]
        )
        pred_visibility = pred_visibility > 0.9
    elif req.method == "ours":
        queries = []
        queries_gt = {}
        for kp in req.keypoints:
            scaled_x = kp.x / (video_W - 1) * (interp_shape[1] - 1)
            scaled_y = kp.y / (video_H - 1) * (interp_shape[0] - 1)
            if kp.frame == 0:
                queries.append([kp.frame, scaled_x, scaled_y])
            if kp.frame == 0 or kp.isManual:
                if kp.id not in queries_gt:
                    queries_gt[kp.id] = []
                queries_gt[kp.id].append([kp.frame, scaled_x, scaled_y])
                isManualDict[f'{kp.id}_{kp.frame}'] = True
        queries = torch.tensor(queries)[None].to(DEFAULT_DEVICE)
        queries_gt = {k: torch.tensor(v)[None].to(DEFAULT_DEVICE) for k, v in queries_gt.items()}
        pred_tracks, pred_visibility, *_ = tto(video, queries, queries_gt, iters=req.iterations, lr=req.learning_rate)
        # threading.Thread(target=tto, args=(video, queries, queries_gt, req.iterations, req.learning_rate)).start()
        pred_tracks *= pred_tracks.new_tensor(
            [(video_W - 1) / (interp_shape[1] - 1), (video_H - 1) / (interp_shape[0] - 1)]
        )
        pred_visibility = pred_visibility > 0.9
    else:
        raise ValueError(f"Unknown method: {req.method}")
    pred_tracks = pred_tracks[0].detach().cpu().numpy()
    out_tracks = []
    for i in range(len(pred_tracks)):
        for j in range(len(pred_tracks[i])):
            out_tracks.append({
                "x": int(pred_tracks[i][j][0]),
                "y": int(pred_tracks[i][j][1]),
                "frame": i,
                "id": j,
                "isManual": isManualDict.get(f'{j}_{i}', False)
            })
    
    return {"keypoints": out_tracks}

def tto(video, queries, queries_gt, iters=500, lr=0.0001):
    global fmaps_pyramid
    if fmaps_pyramid is None:
        fmaps_pyramid = extract_features(video)
    kp_feats = get_kp_feats(queries_gt, fmaps_pyramid)
    ori_feats = kp_feats.detach().clone()
    kp_feats.requires_grad = True
    S = video.shape[1]
    seq_len = (S // 2) + 1
    
    trajs_gt = torch.zeros(1, S, len(queries_gt), 2).to(DEFAULT_DEVICE)
    vis_gt = torch.ones(1, S, len(queries_gt)).to(DEFAULT_DEVICE)
    valids = torch.zeros(1, S, len(queries_gt)).to(DEFAULT_DEVICE)
    for id, queries_t in queries_gt.items():
        query_frame, query_coords = queries_t[:, :, 0].long(), queries_t[:, :, 1:]
        trajs_gt[:, query_frame, id] = query_coords
        valids[:, query_frame, id] = 1

    vis_gts = []
    invis_gts = []
    traj_gts = []
    valids_gts = []

    for ind in range(0, seq_len - S // 2, S // 2):
        vis_gts.append(vis_gt[:, ind : ind + S])
        invis_gts.append(1 - vis_gt[:, ind : ind + S])
        traj_gts.append(trajs_gt[:, ind : ind + S])
        valids_gts.append(valids[:, ind : ind + S])
    
    optimizer = torch.optim.Adam([kp_feats], lr=lr)
    # scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=1e-5/lr, total_iters=iters)
    global optimization_progress
    optimization_progress["current"] = 0
    optimization_progress["total"] = iters
    optimization_progress["in_progress"] = True
    
    for iter in range(iters):
        optimizer.zero_grad()
        pred_tracks, pred_visibility, _, train_data = model.forward(
            video, queries, iters=4,
            is_train=True,
            fmaps_pyramid=fmaps_pyramid,
            track_feat_support_pyramid=kp_feats.repeat(4, 1, 1, 1, 1, 1),
        )
        coord_predictions, vis_predictions, confidence_predictions, valid_mask = train_data
        seq_loss = sequence_loss(coord_predictions, traj_gts, valids_gts, vis=vis_gts, gamma=0.8, add_huber_loss=True)
        track_loss = seq_loss.mean()
        ent_loss = torch.sum(torch.sqrt(torch.sum((pred_tracks - trajs_gt) ** 2, dim=-1)) * valids) / (valids.sum() + 1e-6)
        reg_loss = torch.abs(kp_feats - ori_feats).mean()
        loss = track_loss + 0.01 * reg_loss
        loss.backward()
        optimizer.step()
        # scheduler.step()
        optimization_progress["current"] = iter + 1
        if iter % 50 == 0:
            print(f"iter {iter}, loss {loss.item()}, track_loss {track_loss.item()}, reg_loss {reg_loss.item()}, ent_loss {ent_loss.item()}, lr {lr}")
    
    optimization_progress["in_progress"] = False
    return pred_tracks, pred_visibility
