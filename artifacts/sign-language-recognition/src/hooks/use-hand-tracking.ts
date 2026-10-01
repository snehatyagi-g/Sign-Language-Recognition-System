import { useCallback, useEffect, useRef, useState } from 'react';
import { FilesetResolver, HandLandmarker } from '@mediapipe/tasks-vision';

export type CameraState = 'idle' | 'requesting' | 'loading-model' | 'tracking' | 'error';

type Point3 = { x: number; y: number; z: number };

function normalizeLandmarks(points: Point3[]): number[] | null {
  if (points.length !== 21) return null;
  const wrist = points[0];
  const translated = points.map((point) => ({
    x: point.x - wrist.x,
    y: point.y - wrist.y,
    z: point.z - wrist.z,
  }));
  const scale = Math.max(...translated.map(({ x, y, z }) => Math.hypot(x, y, z)));
  if (!Number.isFinite(scale) || scale < 1e-8) return null;
  const features = translated.flatMap(({ x, y, z }) => [x / scale, y / scale, z / scale]);
  return features.length === 63 && features.every(Number.isFinite) ? features : null;
}

const CONNECTIONS: [number, number][] = [
  [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12], [9, 13], [13, 14], [14, 15],
  [15, 16], [13, 17], [0, 17], [17, 18], [18, 19], [19, 20],
];

export function useHandTracking() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const landmarkerRef = useRef<HandLandmarker | null>(null);
  const frameRef = useRef<number | null>(null);
  const sessionRef = useRef(0);
  const lastPublishRef = useRef(0);
  const [cameraState, setCameraState] = useState<CameraState>('idle');
  const [cameraError, setCameraError] = useState('');
  const [features, setFeatures] = useState<number[] | null>(null);
  const [handFound, setHandFound] = useState(false);

  const stop = useCallback(() => {
    sessionRef.current += 1;
    if (frameRef.current !== null) cancelAnimationFrame(frameRef.current);
    frameRef.current = null;
    landmarkerRef.current?.close();
    landmarkerRef.current = null;
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    const video = videoRef.current;
    if (video) {
      video.pause();
      video.srcObject = null;
    }
    const canvas = canvasRef.current;
    if (canvas) canvas.getContext('2d')?.clearRect(0, 0, canvas.width, canvas.height);
    setFeatures(null);
    setHandFound(false);
    setCameraState('idle');
    setCameraError('');
  }, []);

  const start = useCallback(async () => {
    if (!navigator.mediaDevices?.getUserMedia) {
      setCameraState('error');
      setCameraError('Camera access is unavailable in this browser. Use a secure HTTPS connection.');
      return;
    }
    const session = ++sessionRef.current;
    setCameraError('');
    setFeatures(null);
    setHandFound(false);
    setCameraState('requesting');
    let stream: MediaStream | null = null;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 800 } },
      });
      if (session !== sessionRef.current) {
        stream.getTracks().forEach((track) => track.stop());
        return;
      }
      streamRef.current = stream;
      const video = videoRef.current;
      if (!video) throw new Error('The camera preview could not be initialized.');
      video.srcObject = stream;
      await video.play();
      setCameraState('loading-model');

      const vision = await FilesetResolver.forVisionTasks(
        'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/wasm',
      );
      const modelAssetPath = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';
      let landmarker: HandLandmarker;
      try {
        landmarker = await HandLandmarker.createFromOptions(vision, {
          baseOptions: { modelAssetPath, delegate: 'GPU' },
          runningMode: 'VIDEO',
          numHands: 1,
          minHandDetectionConfidence: 0.55,
          minHandPresenceConfidence: 0.5,
          minTrackingConfidence: 0.5,
        });
      } catch {
        landmarker = await HandLandmarker.createFromOptions(vision, {
          baseOptions: { modelAssetPath, delegate: 'CPU' },
          runningMode: 'VIDEO',
          numHands: 1,
          minHandDetectionConfidence: 0.55,
          minHandPresenceConfidence: 0.5,
          minTrackingConfidence: 0.5,
        });
      }
      if (session !== sessionRef.current) {
        landmarker.close();
        return;
      }
      landmarkerRef.current = landmarker;
      setCameraState('tracking');

      const drawFrame = () => {
        const currentVideo = videoRef.current;
        const canvas = canvasRef.current;
        if (!currentVideo || !canvas || !landmarkerRef.current || session !== sessionRef.current) return;
        if (currentVideo.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
          const width = currentVideo.videoWidth;
          const height = currentVideo.videoHeight;
          if (width && height) {
            if (canvas.width !== width || canvas.height !== height) {
              canvas.width = width;
              canvas.height = height;
            }
            const ctx = canvas.getContext('2d');
            ctx?.clearRect(0, 0, width, height);
            const result = landmarkerRef.current.detectForVideo(currentVideo, performance.now());
            const points = result.landmarks[0] as Point3[] | undefined;
            const normalized = points ? normalizeLandmarks(points) : null;
            setHandFound(Boolean(normalized));
            if (normalized) {
              const now = performance.now();
              if (now - lastPublishRef.current > 120) {
                lastPublishRef.current = now;
                setFeatures(normalized);
              }
              if (ctx && points) {
                ctx.lineWidth = Math.max(2, width / 360);
                ctx.strokeStyle = '#f2926e';
                ctx.fillStyle = '#f5eee0';
                for (const [from, to] of CONNECTIONS) {
                  ctx.beginPath();
                  ctx.moveTo(points[from].x * width, points[from].y * height);
                  ctx.lineTo(points[to].x * width, points[to].y * height);
                  ctx.stroke();
                }
                points.forEach(({ x, y }) => {
                  ctx.beginPath();
                  ctx.arc(x * width, y * height, Math.max(2.2, width / 300), 0, Math.PI * 2);
                  ctx.fill();
                });
              }
            } else {
              setFeatures(null);
            }
          }
        }
        frameRef.current = requestAnimationFrame(drawFrame);
      };
      frameRef.current = requestAnimationFrame(drawFrame);
    } catch (error) {
      if (session !== sessionRef.current) return;
      stream?.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
      landmarkerRef.current?.close();
      landmarkerRef.current = null;
      const failedVideo = videoRef.current;
      if (failedVideo) {
        failedVideo.pause();
        failedVideo.srcObject = null;
      }
      const reason = error instanceof Error ? error.message : 'Unknown camera error';
      const permission = error instanceof DOMException && (error.name === 'NotAllowedError' || error.name === 'PermissionDeniedError');
      setCameraError(permission
        ? 'Camera permission was denied. Allow camera access in your browser settings, then try again.'
        : `Could not start hand tracking: ${reason}`);
      setCameraState('error');
    }
  }, []);

  useEffect(() => () => {
    sessionRef.current += 1;
    if (frameRef.current !== null) cancelAnimationFrame(frameRef.current);
    landmarkerRef.current?.close();
    streamRef.current?.getTracks().forEach((track) => track.stop());
  }, []);

  return { videoRef, canvasRef, cameraState, cameraError, features, handFound, start, stop };
}