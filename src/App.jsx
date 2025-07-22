// Updated: 'NEW' button toggles newKeyMode to force new annotation
import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import {
  PlayIcon,
  PauseIcon,
  BackwardIcon,
  ForwardIcon
} from '@heroicons/react/24/solid';


const colorPalette = [
  'border-l-4 border-red-500',
  'border-l-4 border-blue-500',
  'border-l-4 border-green-500',
  'border-l-4 border-yellow-500',
  'border-l-4 border-purple-500',
  'border-l-4 border-pink-500',
  'border-l-4 border-orange-500',
  'border-l-4 border-teal-500',
  'border-l-4 border-cyan-500',
  'border-l-4 border-lime-500',
  'border-l-4 border-indigo-500',
  'border-l-4 border-fuchsia-500',
  'border-l-4 border-rose-500',
  'border-l-4 border-amber-500',
  'border-l-4 border-violet-500',
  'border-l-4 border-sky-500',
  'border-l-4 border-emerald-500',
  'border-l-4 border-gray-500',
];

const tailwindFillColors = {
  red: '#ef4444',
  blue: '#3b82f6',
  green: '#10b981',
  yellow: '#eab308',
  purple: '#8b5cf6',
  pink: '#ec4899',
  orange: '#f97316',
  teal: '#14b8a6',
  cyan: '#06b6d4',
  lime: '#84cc16',
  indigo: '#6366f1',
  fuchsia: '#d946ef',
  rose: '#f43f5e',
  amber: '#f59e0b',
  violet: '#a78bfa',
  sky: '#0ea5e9',
  emerald: '#10b981',
  gray: '#6b7280',
  neutral: '#737373',
  blueDark: '#1e40af',
  redDark: '#b91c1c',
};



function App() {
  const [annotations, setAnnotations] = useState([]);
  const [details, setDetails] = useState('');
  const [isBackground, setIsBackground] = useState(false);
  const [videoFile, setVideoFile] = useState(null);
  const [currentFrame, setCurrentFrame] = useState(0);
  const [duration, setDuration] = useState(0);
  const [zoom, setZoom] = useState(1);
  const [keypoints, setKeypoints] = useState([]);
  const [clickedPoint, setClickedPoint] = useState(null);
  const [newKeyMode, setNewKeyMode] = useState(false);
  const [editingId, setEditingId] = useState(null);
  const [editDetails, setEditDetails] = useState('');
  const [editError, setEditError] = useState(false);
  const [selectedId, setSelectedId] = useState(null);
  const [tracking, setTracking] = useState(false);
  const [mousePos, setMousePos] = useState(null);
  const [selectedMethod, setSelectedMethod] = useState("cotracker3"); // default method
  const [fps, setFps] = useState(15); // e.g., from OpenCV or hardcoded
  const frameDuration = 1 / fps;
  const [optIterations, setOptIterations] = useState(50);
  const [optLrExp, setOptLrExp] = useState(-3); // log10(1e-4)
  const optLr = Math.pow(10, optLrExp); // This is your final learning rate
  const [isOptimizing, setIsOptimizing] = useState(false);
  const [progress, setProgress] = useState(0);
  
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const totalFrames = Math.floor((videoRef.current?.duration || 0) * fps);
  const annotationListRef = useRef(null);

  useEffect(() => {
    fetchAnnotations();
  }, []);

  const fetchAnnotations = async () => {
    const res = await axios.get('http://localhost:8000/annotations');
    setAnnotations(res.data);
  };

  const generateNextAvailableId = () => {
    // const currentAnnotations = await axios.get('http://localhost:8000/annotations'); 
    const usedIds = new Set(annotations.map(a => a.id));
    let i = 0;
    while (usedIds.has(i)) i++;
    return i;
  };  

  const handleNew = async () => {
    const newId = generateNextAvailableId();
  
    const newAnnotation = {
      id: newId,
      details: '',
      keyframes: 0,
      is_background: false,
    };
 
    setAnnotations(prev => [...prev, newAnnotation]);
    setSelectedId(newId);
    setTimeout(() => {
      if (annotationListRef.current) {
        annotationListRef.current.scrollTop = annotationListRef.current.scrollHeight;
      }
    }, 0);
    await axios.post('http://localhost:8000/annotations', newAnnotation);
  };

  const addAnnotation = async () => {
    // let existing = !newKeyMode && annotations.find(a => a.details === details);
    // let targetId = existing ? existing.id : annotations.length;

    // setSelectedId(targetId);
    
    // if (!existing) {
    //   const newAnnotation = {
    //     id: targetId,
    //     keyframes: currentFrame,
    //     details: details,
    //     is_background: isBackground
    //   };
    //   await axios.post('http://localhost:8000/annotations', newAnnotation);
    // }
    let targetId;

    // ✅ Use selectedId if not in "new key" mode
    if (!newKeyMode && selectedId !== null) {
      targetId = selectedId;
      console.log('selectedId', selectedId);
    } else {
      // Create a new annotation
      targetId = annotations.length;
      const newAnnotation = {
        id: targetId,
        keyframes: currentFrame,
        details: details,
        is_background: isBackground
      };
      await axios.post('http://localhost:8000/annotations', newAnnotation);
      console.log('before: selectedId', selectedId, targetId, annotations.length);
      setSelectedId(targetId); // auto-select new
      console.log('selectedId', selectedId, targetId, annotations.length);
    }

    if (clickedPoint) {
      console.log('targetId', targetId);
      setKeypoints([...keypoints, { ...clickedPoint, frame: currentFrame, id: targetId, isManual: true }]);
      setClickedPoint(null);
    }

    setDetails('');
    setIsBackground(false);
    setNewKeyMode(false);
    fetchAnnotations();
  };

  const deleteAnnotation = async (id) => {
    setSelectedId(null);
    await axios.delete(`http://localhost:8000/annotations/${id}`);
    setKeypoints(prev => prev.filter(kp => kp.id !== id));
    fetchAnnotations();
  };

  const handleVideoUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
  
    const formData = new FormData();
    formData.append('file', file);
  
    try {
      const res = await axios.post('http://localhost:8000/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
  
      const videoUrl = res.data.url; // backend returns the video path
      setVideoFile(videoUrl); // use this URL to load video
    } catch (err) {
      console.error("Failed to upload video", err);
    }
  };

  const handleSubmit = () => {
    const exportData = {
      annotations,
      keypoints, // or frameKeypoints if that's your structure
    };
  
    const json = JSON.stringify(exportData, null, 2); // formatted JSON
    const blob = new Blob([json], { type: "application/json" });
    const url = URL.createObjectURL(blob);
  
    const a = document.createElement("a");
    a.href = url;
    a.download = "annotations.json"; // ✅ filename
    a.click();
  
    URL.revokeObjectURL(url);
  };
  
  const handleDeleteAll = async () => {
    const confirmed = window.confirm("Are you sure you want to delete all annotations?");
    if (!confirmed) return;
  
    try {
      await axios.delete("http://localhost:8000/annotations/"); // ✅ delete all
      console.log("✅ Deleted all annotations");
      setAnnotations([]);
      setKeypoints([]);
      setSelectedId(null);
    } catch (err) {
      console.error("Failed to delete all annotations:", err);
    }
    fetchAnnotations();
  };  

  const onLoadedMetadata = () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (video && canvas) {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
    }

    setDuration(video.duration)
    // if (videoRef.current) {
    //   setDuration(videoRef.current.duration);
    // }
  };

  const formatTime = (time) => {
    const mins = Math.floor(time / 60);
    const secs = Math.floor(time % 60).toString().padStart(2, '0');
    return `${mins}:${secs}`;
  };

  const handleZoomChange = (e) => {
    setZoom(parseFloat(e.target.value));
  };

  const handleCanvasClick = (e) => {
    if (selectedId === null) return;

    const rect = canvasRef.current.getBoundingClientRect();
    const scaleX = canvasRef.current.width / rect.width;
    const scaleY = canvasRef.current.height / rect.height;

    const x = (e.clientX - rect.left) * scaleX;
    const y = (e.clientY - rect.top) * scaleY;
    setClickedPoint({ x, y, frame: currentFrame });
  };

  const handleTrack = async () => {
    if (keypoints.length === 0) {
      window.alert("Please add keypoints before tracking. Click 'NEW' to create an annotation, then click on the video to place keypoints.");
      return;
    }

    setTracking(true); // start spinner
    
    if (selectedMethod === 'ours') {
      setIsOptimizing(true);
      setProgress(0);
      // pollProgress();
    } 

    try {
      const response = await axios.post('http://localhost:8000/track', {
        keypoints: keypoints,
        method: selectedMethod,
        iterations: optIterations,
        learning_rate: optLr,
      });
  
      if (response.data?.keypoints) {
        setKeypoints(response.data.keypoints);
        if (selectedMethod === 'cotracker3') {
          setSelectedMethod('ours'); // default to CoTracker3
        }
      } else {
        console.warn("⚠️ No keypoints returned");
      }
    } catch (err) {
      console.error("❌ Tracking failed", err);
    }
  
    setTracking(false); // stop spinner
  };

  const stepFrames = (delta) => {
    if (!videoRef.current) return;
  
    const newTime = videoRef.current.currentTime + delta / fps;
    videoRef.current.currentTime = Math.max(0, newTime);
  
    const frame = Math.max(0, Math.min(totalFrames - 1, Math.floor(videoRef.current.currentTime * fps)));
    // frame = Math.max(0, Math.min(totalFrames - 1, frame));
    setCurrentFrame(frame); // 🔥 Manually sync frame state
  };

  const pollProgress = () => {
    const interval = setInterval(async () => {
      const res = await axios.get("http://localhost:8000/track/progress");
      setProgress(res.data.current);
  
      if (!res.data.in_progress) {
        clearInterval(interval);
        setIsOptimizing(false);
      }
    }, 100);
  };
  
  
  useEffect(() => {
    let rafId = null;
  
    const updateFrame = () => {
      if (videoRef.current && !videoRef.current.paused) {
        const currentTime = videoRef.current.currentTime;
        const frame = Math.floor(currentTime * fps);
        setCurrentFrame((prev) => {
          if (frame !== prev) return frame;
          return prev;
        });
      }
  
      rafId = requestAnimationFrame(updateFrame);
    };
  
    rafId = requestAnimationFrame(updateFrame);
    return () => cancelAnimationFrame(rafId);
  }, [fps]);
  

  useEffect(() => {
    setClickedPoint(null);
  }, [currentFrame]);

  useEffect(() => {
    const ctx = canvasRef.current?.getContext('2d');
    if (ctx && videoRef.current) {
      ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);

      keypoints.forEach(({ x, y, frame, id, isManual }) => {
        if (frame === currentFrame) {
          ctx.beginPath();
          ctx.arc(x, y, 5, 0, 2 * Math.PI);
          ctx.fillStyle = tailwindFillColors[getColorFromId(id)];
          ctx.fill();
        }
      });

      if (mousePos && !clickedPoint && selectedId !== null) {
        ctx.beginPath();
        ctx.setLineDash([3, 3]);
        // console.log('selectedId', selectedId, 'color', getColorFromId(selectedId));
        ctx.strokeStyle = tailwindFillColors[getColorFromId(selectedId || 0)];
        ctx.lineWidth = 1;
      
        // vertical
        ctx.moveTo(mousePos.x, 0);
        ctx.lineTo(mousePos.x, canvasRef.current.height);
      
        // horizontal
        ctx.moveTo(0, mousePos.y);
        ctx.lineTo(canvasRef.current.width, mousePos.y);
      
        ctx.stroke();
        ctx.setLineDash([]);
      
        // optional ghost point
        ctx.beginPath();
        ctx.arc(mousePos.x, mousePos.y, 5, 0, 2 * Math.PI);
        ctx.strokeStyle = tailwindFillColors[getColorFromId(selectedId || 0)];
        ctx.lineWidth = 1.5;
        ctx.stroke();
      }

      if (clickedPoint && clickedPoint.frame === currentFrame) {
        
        // setKeypoints([...keypoints, { ...clickedPoint, id: newKeyMode ? annotations.length : selectedId || 0, isManual: true }]);
        // Check if a keypoint for this frame + id already exists
        const existingIndex = keypoints.findIndex(
          (kp) => kp.id === selectedId && kp.frame === currentFrame
        );

        if (existingIndex !== -1) {
          // 🔁 Replace the existing keypoint
          const updated = [...keypoints];
          const { x, y } = clickedPoint;
          updated[existingIndex] = { ...updated[existingIndex], x, y , isManual: true };
          setKeypoints(updated);
        } else {
          // ➕ Add a new keypoint
          setKeypoints([...keypoints, { ...clickedPoint, id: selectedId || 0, isManual: true }]);
        }
        setClickedPoint(null);
        // ctx.beginPath();
        // ctx.setLineDash([4, 2]);
        // ctx.arc(clickedPoint.x, clickedPoint.y, 5, 0, 2 * Math.PI);
        // ctx.strokeStyle = getColorFromId(newKeyMode ? annotations.length : selectedId || 0);
        // ctx.lineWidth = 2;
        // ctx.stroke();
        // ctx.setLineDash([]);
      }
    }
  }, [keypoints, currentFrame, zoom, mousePos, clickedPoint, newKeyMode, annotations, details]);

  const getColorFromId = (id) => {
    const colors = [
      'red', 'blue', 'green', 'yellow', 'purple', 'pink', 'orange', 'teal',
      'cyan', 'lime', 'indigo', 'fuchsia', 'rose', 'amber', 'violet',
      'sky', 'emerald', 'gray'
    ];
    // console.log('id', id, 'color', colors[id % colors.length]);
    return colors[id % colors.length];
  };
  

  return (
    <div className="p-4 flex flex-col gap-6">
      {/* <div className="flex justify-end gap-4 mb-2">
        <button
          className="bg-indigo-700 text-white px-4 py-1 rounded shadow"
          onClick={() => setNewKeyMode(true)}
        >
          NEW
        </button>
        <button className="bg-gray-200 text-black px-4 py-1 rounded shadow">
          SUBMIT
        </button>
      </div> */}
      <div className="flex gap-8">
      <div className="w-2/3 bg-gray-100 p-2 border flex flex-col items-center relative overflow-hidden" style={{ transform: `scale(${zoom})` }}>
          <div className="mb-2 w-full">
            <input type="file" accept="video/*" onChange={handleVideoUpload} className="mb-2" />
          </div>
          {/* VIDEO + CANVAS OVERLAY */}
          <div className="relative w-full mb-2">
            {videoFile && (
              <>
                <video
                  ref={videoRef}
                  src={videoFile}
                  width="100%"
                  className="mb-0"
                  onLoadedMetadata={onLoadedMetadata}
                  // onTimeUpdate={() => {
                  //   if (videoRef.current) {
                  //     const fps = 15;
                  //     const frame = Math.floor(videoRef.current.currentTime * fps);
                  //     setCurrentFrame(frame);
                  //   }
                  // }}
                  muted
                  playsInline
                />
                <canvas
                  ref={canvasRef}
                  // width={videoRef.current?.videoWidth || 640}
                  // height={videoRef.current?.videoHeight || 360}
                  className="absolute top-0 left-0 w-full h-full"
                  onClick={handleCanvasClick}
                  onMouseMove={(e) => {
                    const rect = canvasRef.current.getBoundingClientRect();
                    const scaleX = canvasRef.current.width / rect.width;
                    const scaleY = canvasRef.current.height / rect.height;
                    const x = (e.clientX - rect.left) * scaleX;
                    const y = (e.clientY - rect.top) * scaleY;
                    setMousePos({ x, y });
                  }}
                  onMouseLeave={() => setMousePos(null)}
                />
              </>
            )}
          </div>

          {/* TICK MARK GRID (above progress bar) */}
          {/* {videoRef.current && (
            <div className="w-full flex items-end h-6 overflow-hidden relative">
              {[...Array(Math.floor(duration * 25)).keys()].map((_, i) => (
                <div
                  key={i}
                  className="border-l border-cyan-400"
                  style={{
                    height: '100%',
                    width: `${100 / Math.floor(duration * 25)}%`,
                    minWidth: '1px'
                  }}
                />
              ))}
            </div>
          )} */}
          {videoRef.current && (
            <div className="relative w-full h-6 overflow-hidden">
              {/* Blue base grid: every frame */}
              <div className="absolute inset-0 flex">
                {[...Array(Math.floor(duration * 15)).keys()].map((_, i) => (
                  <div
                    key={`blue-${i}`}
                    className="h-full border-l border-blue-300"
                    style={{
                      width: `${100 / Math.floor(duration * 15)}%`,
                      minWidth: '1px'
                    }}
                  />
                ))}
              </div>

              {/* Purple overlay: only annotated frames */}
              {selectedId !== null && (
                <div className="absolute inset-0 flex">
                  {[...Array(Math.floor(duration * 15)).keys()].map((frame, i) => {
                    // const isAnnotated = keypoints.some(k => k.id === selectedId && k.frame === frame);
                    const frameKeypoints = keypoints.filter(k => k.id === selectedId && k.frame === frame);
                    const isManual = frameKeypoints.some(k => k.isManual);
                    const isTracked = frameKeypoints.some(k => !k.isManual);

                    let borderColor = '';
                    if (isManual) {
                      borderColor = 'border-green-500';
                    } else if (isTracked) {
                      borderColor = 'border-purple-500';
                    }
                    return (
                      <div
                        key={`tick-${i}`}
                        className={borderColor ? `h-full border-l-2 ${borderColor}` : ''}
                        style={{
                          width: `${100 / Math.floor(duration * 15)}%`,
                          minWidth: '1px'
                        }}
                      />
                    );
                  })}
                </div>
              )}
            </div>
          )}


          {/* PROGRESS BAR */}
          <input
            type="range"
            min={0}
            max={totalFrames - 1}
            step={1}
            value={currentFrame}
            onChange={(e) => {
              const frame = parseInt(e.target.value);
              const time = frame / fps;
              if (videoRef.current) {
                videoRef.current.currentTime = time;
              }
              setCurrentFrame(frame);
            }}
            className="w-full h-[3px] bg-gray-300 accent-blue-500 appearance-none mb-2"
          /> 

          {/* BOTTOM CONTROL BAR */}
          <div className="flex items-center justify-between gap-6 px-2 py-1 bg-white text-sm text-black border-t shadow-sm">
            <div className="flex items-center gap-3">
              <button onClick={() => videoRef.current?.paused ? videoRef.current.play() : videoRef.current.pause()}>
                {videoRef.current?.paused ? (
                  <PlayIcon className="w-5 h-5 text-black" />
                ) : (
                  <PauseIcon className="w-5 h-5 text-black" />
                )}
              </button>
              <button onClick={() => stepFrames(-20)}>
                <BackwardIcon className="w-5 h-5 text-gray-600" />
                <span className="text-xs">20</span>
              </button>
              <button onClick={() => stepFrames(-5)}>
                <BackwardIcon className="w-5 h-5 text-gray-600" />
                <span className="text-xs">5</span>
              </button>
              <button onClick={() => stepFrames(-1)}>
                <BackwardIcon className="w-5 h-5 text-gray-600" />
                <span className="text-xs">1</span>
              </button>
              <button onClick={() => stepFrames(+1)}>
                <ForwardIcon className="w-5 h-5 text-gray-600" />
                <span className="text-xs">1</span>
              </button>
              <button onClick={() => stepFrames(+5)}>
                <ForwardIcon className="w-5 h-5 text-gray-600" />
                <span className="text-xs">5</span>
              </button>
              <button onClick={() => stepFrames(+20)}>
                <ForwardIcon className="w-5 h-5 text-gray-600" />
                <span className="text-xs">20</span>
              </button>
              <span className="ml-2">FR</span>
              <span>1x</span>
              <span>{formatTime(videoRef.current?.currentTime || 0)} / {formatTime(duration)}</span>
            </div>
            {/* Right side: Zoom + Frame */}
            <div className="flex items-center gap-3">
              <div className="flex items-center gap-1">
                <span className="text-lg">🔍</span>
                <span className="text-black font-medium">Zoom:</span>
                <input
                  type="range"
                  min="0.9"
                  max="1.2"
                  step="0.01"
                  value={zoom}
                  onChange={handleZoomChange}
                  className="h-2 accent-blue-600"
                />
                <span className="text-red-600 font-semibold">
                  {(zoom * 100).toFixed(2)}%
                </span>
              </div>
              <span className="text-gray-800 font-medium">
                Frames: {currentFrame}
              </span>
            </div>
            {/* <div className="flex items-center gap-4">
              <span>🧪</span>
              <span>💧</span>
              <span className="text-red-600">Zoom: {(zoom * 100).toFixed(2)}%</span>
              <span>Frames: {currentFrame}/{Math.floor(duration * 25)}</span>
            </div> */}
          </div> 
        </div>

        <div className="w-1/3">
          <div className="flex justify-end gap-4 mb-2">
          <button
            onClick={handleTrack}
            disabled={tracking}
            className={`px-4 py-1 rounded shadow text-white ${
              tracking ? 'bg-gray-400 cursor-not-allowed' : 'bg-red-500 hover:bg-red-600'
            }`}
          >
            {tracking ? (
              <div className="flex items-center gap-2">
                <svg className="animate-spin h-4 w-4 text-white" viewBox="0 0 24 24">
                  <circle
                    className="opacity-25"
                    cx="12" cy="12" r="10"
                    stroke="currentColor"
                    strokeWidth="4"
                    fill="none"
                  />
                  <path
                    className="opacity-75"
                    fill="currentColor"
                    d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z"
                  />
                </svg>
                Tracking...
              </div>
            ) : (
              'TRACK'
            )}
          </button> 
            <button
              className="bg-indigo-700 text-white px-4 py-1 rounded shadow"
              onClick={handleNew}
            >
              NEW
            </button>
            <button onClick={handleSubmit} className="bg-gray-200 text-black px-4 py-1 rounded shadow">
              EXPORT 
            </button>
          </div>
          <div className="flex justify-between items-center mb-2">
            <h2 className="text-xl font-bold">Annotations</h2>
            <button className="text-blue-600 text-sm hover:underline"
                    onClick={handleDeleteAll}>DELETE ALL</button>
          </div>

          <div ref={annotationListRef} className="space-y-3 overflow-y-auto max-h-[60vh] pr-2">
            {annotations.map((anno) => (
              <div
                key={anno.id}
                onClick={() => setSelectedId(anno.id)}
                className={`border rounded-lg shadow-sm relative overflow-hidden ${colorPalette[anno.id % colorPalette.length]} ${selectedId === anno.id ? 'ring-2 ring-blue-400' : ''}`}
              >
                <div className="pl-4 pr-2 py-2 ml-2">
                  <div className="flex justify-between items-center">
                    <span className="font-semibold">Point</span>
                    <div className="space-x-2">
                      {/* <button className="text-gray-600 hover:text-black">✏️</button> */}
                      <button
                        onClick={() => deleteAnnotation(anno.id)}
                        className="text-red-500 hover:text-red-700"
                      >🗑</button>
                      {/* <button className="text-green-600">✔</button> */}
                    </div>
                  </div>
                  <p className="text-sm mt-1">
                    ID: {anno.id} Keyframes: {keypoints.filter(k => k.id === anno.id).length}
                  </p>
                  {editingId === anno.id ? (
                    <div className="mt-2">
                      {editError && (
                        <p className="text-red-500 text-sm mb-1">❗ Missing required label</p>
                      )}
                      <label className="text-sm font-medium flex items-center gap-1 mb-1">
                        details <span className="text-red-500">{editError && '❗'}</span>
                      </label>
                      <input
                        type="text"
                        value={editDetails}
                        onChange={(e) => {
                          setEditDetails(e.target.value);
                          if (e.target.value.trim() !== '') setEditError(false);
                        }}
                        placeholder="details"
                        className={`w-full border-b focus:outline-none px-1 pb-0.5 ${
                          editError ? 'border-red-500' : 'border-gray-400'
                        }`}
                      />
                      <div className="flex justify-end gap-2 mt-2">
                        <button
                          className="text-green-600"
                          onClick={async () => {
                            if (editDetails.trim() === '') {
                              setEditError(true);
                              return;
                            }
                            const updated = { ...anno, details: editDetails };
                            await axios.delete(`http://localhost:8000/annotations/${anno.id}`);
                            await axios.post('http://localhost:8000/annotations', updated);
                            setEditingId(null);
                            setEditError(false);
                            fetchAnnotations();
                          }}
                        >
                          ✔
                        </button>
                        <button
                          onClick={() => {
                            setEditingId(null);
                            setEditError(false);
                          }}
                          className="text-gray-500"
                        >
                          ❌
                        </button>
                      </div>
                    </div>
                  ) : (
                    <div className="flex justify-between items-center mt-1">
                      <p className="text-sm text-gray-600">details: {anno.details}</p>
                      <button
                        onClick={() => {
                          setEditingId(anno.id);
                          setEditDetails(anno.details);
                        }}
                        className="text-sm text-gray-500 hover:text-black"
                      >
                        ⏷
                      </button>
                    </div>
                  )}
 
                </div>
              </div>
            ))}
          </div>

          {/* <div className="mt-6 border-t pt-4">
            <h3 className="font-semibold mb-2">Add New Annotation</h3>
            <input
              type="text"
              placeholder="Details"
              value={details}
              onChange={(e) => setDetails(e.target.value)}
              className="border p-1 w-full mb-2"
            />
            <div className="flex gap-4 items-center mb-2">
              <label>
                <input
                  type="radio"
                  checked={!isBackground}
                  onChange={() => setIsBackground(false)}
                /> Point
              </label>
              <label>
                <input
                  type="radio"
                  checked={isBackground}
                  onChange={() => setIsBackground(true)}
                /> Background Point
              </label>
            </div>
            <button onClick={addAnnotation} className="bg-blue-500 text-white px-3 py-1 rounded">Add</button>
          </div> */}
          <div className="mt-6 border-t pt-4">
            <h3 className="font-semibold mb-2">Tracking Method</h3>
            <select
              value={selectedMethod}
              onChange={(e) => setSelectedMethod(e.target.value)}
              className="border px-2 py-1 rounded w-full"
            >
              <option value="cotracker3">Step 1: Rough Tracking through CoTracker3 ❄️</option>
              <option value="ours">Step 2: Optimize the Tracker 🔥</option>
            </select>

            {selectedMethod === 'ours' && (
              <div className="space-y-5 mt-4">
              {/* Optimization Iterations Slider */}
              <div>
                <label className="block text-sm font-medium mb-1">
                  Optimization Iterations: <span className="font-semibold">{optIterations}</span>
                </label>
                <input
                  type="range"
                  min={10}
                  max={1000}
                  step={10}
                  value={optIterations}
                  onChange={(e) => setOptIterations(parseInt(e.target.value))}
                  className="w-full accent-blue-500"
                />
              </div>
          
              {/* Learning Rate Slider */}
              <div>
                <label className="block text-sm font-medium mb-1">
                  Learning Rate: <span className="font-semibold">{optLr.toExponential(1)}</span>
                </label>
                <input
                  type="range"
                  min={-5}
                  max={-2}
                  step={0.1}
                  value={optLrExp}
                  onChange={(e) => setOptLrExp(parseFloat(e.target.value))}
                  className="w-full accent-green-500"
                />
              </div>

              {/* {isOptimizing && (
                <div className="w-full bg-gray-200 rounded h-4 overflow-hidden my-4">
                  <div
                    className="bg-blue-500 h-full transition-all duration-100 ease-linear"
                    style={{ width: `${(progress / optIterations) * 100}%` }}
                  />
                </div>
              )} */}

            </div>
            )}

          </div>
        </div>
      </div>
 
    </div>
  );
}

export default App;
