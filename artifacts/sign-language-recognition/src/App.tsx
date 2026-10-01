import { createContext, type ChangeEvent, type ReactNode, useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react';
import { QueryClient, QueryClientProvider, useQueryClient } from '@tanstack/react-query';
import {
  getGetGestureLabelsQueryKey,
  getGetGestureModelStatusQueryKey,
  getGetTrainingStatusQueryKey,
  getHealthCheckQueryKey,
  useGetGestureLabels,
  useGetGestureModelStatus,
  useGetTrainingStatus,
  useHealthCheck,
  useImportGestureDataset,
  usePredictGesture,
  useStartGestureTraining,
  type GesturePrediction,
  type GestureSample,
  type ModelStatus,
  type TrainingStatus,
} from '@workspace/api-client-react';
import {
  Activity,
  AlertCircle,
  Camera,
  CameraOff,
  Check,
  ChevronRight,
  CircleHelp,
  Database,
  Download,
  FileUp,
  Fingerprint,
  FlaskConical,
  Hand,
  Radio,
  RefreshCw,
  ShieldCheck,
  Trash2,
  Upload,
  Volume2,
  VolumeX,
  Wifi,
} from 'lucide-react';
import { Link, Route, Switch, useLocation, Router as WouterRouter } from 'wouter';
import { ErrorBoundary } from '@/components/error-boundary';
import { Toaster } from '@/components/ui/toaster';
import { TooltipProvider } from '@/components/ui/tooltip';
import NotFound from '@/pages/not-found';
import { useHandTracking } from '@/hooks/use-hand-tracking';

const queryClient = new QueryClient();

type DatasetResultView = {
  importedSamples: number;
  totalSamples: number;
  classCounts: { label: string; count: number }[];
};

type RecognitionEvent = {
  id: string;
  label: string;
  confidence: number;
  occurredAt: string;
  accepted: boolean;
};

const RECOGNITION_HISTORY_KEY = 'handstudy-recognition-history';
const MAX_RECOGNITION_HISTORY = 30;

function formatGestureLabel(label: string): string {
  return label
    .toLowerCase()
    .replace(/[_-]+/g, ' ')
    .replace(/\b\w/g, (character) => character.toUpperCase());
}

function loadRecognitionHistory(): RecognitionEvent[] {
  try {
    const saved = localStorage.getItem(RECOGNITION_HISTORY_KEY);
    if (!saved) return [];
    const value = JSON.parse(saved) as RecognitionEvent[];
    return Array.isArray(value)
      ? value.filter((event) =>
          typeof event?.id === 'string' &&
          typeof event.label === 'string' &&
          Number.isFinite(event.confidence) &&
          typeof event.occurredAt === 'string' &&
          typeof event.accepted === 'boolean',
        ).slice(0, MAX_RECOGNITION_HISTORY)
      : [];
  } catch {
    return [];
  }
}

type WorkspaceProps = {
  labels: string[];
  labelsLoading: boolean;
  labelsError: boolean;
  retryLabels: () => void;
  model: ModelStatus | undefined;
  modelLoading: boolean;
  modelError: boolean;
  retryModel: () => void;
  samples: GestureSample[];
  setSamples: (samples: GestureSample[]) => void;
  uploadedResult: DatasetResultView | null;
  setUploadedResult: (value: DatasetResultView | null) => void;
  importState: ReturnType<typeof useImportGestureDataset>;
  training: TrainingStatus | undefined;
  trainingLoading: boolean;
  trainingError: boolean;
  retryTraining: () => void;
  trainingMutation: ReturnType<typeof useStartGestureTraining>;
  healthOnline: boolean;
  healthDisplay: string;
  prediction: GesturePrediction | null;
  predictionError: string;
  confidenceThreshold: number;
  setConfidenceThreshold: (value: number) => void;
  recognitionHistory: RecognitionEvent[];
  recognizedText: string;
  clearRecognitionHistory: () => void;
  camera: ReturnType<typeof useHandTracking>;
  speechEnabled: boolean;
  setSpeechEnabled: (value: boolean) => void;
};

type WorkspaceContextValue = WorkspaceProps & {
  onImport: (samples: GestureSample[]) => void;
  onTrain: (epochs: number, batchSize: number) => void;
};

const WorkspaceContext = createContext<WorkspaceContextValue | null>(null);

function errorText(error: unknown): string {
  if (error instanceof Error) return error.message;
  if (typeof error === 'string') return error;
  return 'The request could not be completed. Check the API connection and try again.';
}

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <TooltipProvider>
        <WouterRouter base={import.meta.env.BASE_URL.replace(/\/$/, '')}>
          <Workspace />
        </WouterRouter>
        <Toaster />
      </TooltipProvider>
    </QueryClientProvider>
  );
}

function Workspace() {
  const queryClient = useQueryClient();
  const healthQuery = useHealthCheck({ query: { queryKey: getHealthCheckQueryKey(), refetchInterval: 15000 } });
  const labelsQuery = useGetGestureLabels({ query: { queryKey: getGetGestureLabelsQueryKey() } });
  const modelQuery = useGetGestureModelStatus({ query: { queryKey: getGetGestureModelStatusQueryKey(), refetchInterval: 12000 } });
  const trainingQuery = useGetTrainingStatus({ query: { queryKey: getGetTrainingStatusQueryKey(), refetchInterval: 2500 } });
  const importState = useImportGestureDataset();
  const trainingMutation = useStartGestureTraining();
  const predictionMutation = usePredictGesture();
  const camera = useHandTracking();
  const labels = labelsQuery.data?.labels ?? [];
  const [samples, setSamples] = useState<GestureSample[]>(() => {
    try {
      const saved = localStorage.getItem('asl-research-samples');
      if (!saved) return [];
      const value = JSON.parse(saved) as GestureSample[];
      return Array.isArray(value) ? value.filter((sample) =>
        typeof sample?.label === 'string' && sample.label.trim().length > 0 &&
        Array.isArray(sample.features) && sample.features.length === 63 &&
        sample.features.every((feature) => Number.isFinite(feature)),
      ) : [];
    } catch {
      return [];
    }
  });
  const [uploadedResult, setUploadedResult] = useState<DatasetResultView | null>(null);
  const [prediction, setPrediction] = useState<GesturePrediction | null>(null);
  const [predictionError, setPredictionError] = useState('');
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.6);
  const [recognitionHistory, setRecognitionHistory] = useState<RecognitionEvent[]>(loadRecognitionHistory);
  const [speechEnabled, setSpeechEnabled] = useState(false);
  const predictionMutateRef = useRef(predictionMutation.mutate);
  const lastPredictionAtRef = useRef(0);
  const modelReadyRef = useRef(false);
  const speechEnabledRef = useRef(speechEnabled);
  const confidenceThresholdRef = useRef(confidenceThreshold);
  const lastSpokenLabelRef = useRef('');
  const lastHistoryLabelRef = useRef<string | null>(recognitionHistory[0]
    ? recognitionHistory[0].accepted ? recognitionHistory[0].label : 'UNKNOWN'
    : null);
  predictionMutateRef.current = predictionMutation.mutate;
  modelReadyRef.current = modelQuery.data?.modelReady === true;
  speechEnabledRef.current = speechEnabled;
  confidenceThresholdRef.current = confidenceThreshold;

  useEffect(() => {
    try {
      localStorage.setItem('asl-research-samples', JSON.stringify(samples));
    } catch {
      // The workbench remains usable when browser storage is disabled or full.
    }
  }, [samples]);

  useEffect(() => {
    try {
      localStorage.setItem(RECOGNITION_HISTORY_KEY, JSON.stringify(recognitionHistory));
    } catch {
      // Recognition remains usable when browser storage is disabled or full.
    }
  }, [recognitionHistory]);

  useEffect(() => {
    if (!camera.features) {
      setPrediction(null);
      setPredictionError('');
      lastHistoryLabelRef.current = null;
      return;
    }
    if (!modelQuery.data?.modelReady) {
      setPrediction(null);
      return;
    }
    const now = performance.now();
    if (now - lastPredictionAtRef.current < 850) return;
    lastPredictionAtRef.current = now;
    predictionMutateRef.current({ data: { features: camera.features } }, {
      onSuccess: (result) => {
        if (!modelReadyRef.current) return;
        setPrediction(result);
        setPredictionError('');
        const accepted = Number.isFinite(result.confidence) && result.confidence >= confidenceThresholdRef.current;
        const historyLabel = accepted ? result.label : 'UNKNOWN';
        if (historyLabel !== lastHistoryLabelRef.current) {
          lastHistoryLabelRef.current = historyLabel;
          const occurredAt = new Date().toISOString();
          setRecognitionHistory((current) => [{
            id: `${Date.now()}-${result.label}`,
            label: result.label,
            confidence: result.confidence,
            occurredAt,
            accepted,
          }, ...current].slice(0, MAX_RECOGNITION_HISTORY));
        }
        if (accepted && speechEnabledRef.current && result.label && result.label !== lastSpokenLabelRef.current && 'speechSynthesis' in window) {
          window.speechSynthesis.cancel();
          window.speechSynthesis.speak(new SpeechSynthesisUtterance(formatGestureLabel(result.label)));
          lastSpokenLabelRef.current = result.label;
        } else if (!accepted) {
          lastSpokenLabelRef.current = '';
        }
      },
      onError: (error) => {
        setPredictionError(errorText(error));
        setPrediction(null);
      },
    });
  }, [camera.features, modelQuery.data?.modelReady]);

  useEffect(() => {
    if (!speechEnabled && 'speechSynthesis' in window) window.speechSynthesis.cancel();
  }, [speechEnabled]);

  const recognizedText = useMemo(
    () => recognitionHistory
      .filter((event) => event.accepted)
      .slice()
      .reverse()
      .map((event) => formatGestureLabel(event.label))
      .join(' · '),
    [recognitionHistory],
  );

  const clearRecognitionHistory = useCallback(() => {
    const current = prediction
      ? prediction.confidence >= confidenceThreshold ? prediction.label : 'UNKNOWN'
      : null;
    lastHistoryLabelRef.current = current;
    setRecognitionHistory([]);
  }, [confidenceThreshold, prediction]);

  const handleImport = useCallback((newSamples: GestureSample[]) => {
    if (!newSamples.length) return;
    importState.mutate({ data: { samples: newSamples } }, {
      onSuccess: (result) => {
        setUploadedResult(result);
        setSamples([]);
        void queryClient.invalidateQueries({ queryKey: getGetGestureModelStatusQueryKey() });
      },
    });
  }, [importState.mutate, queryClient]);

  const handleTraining = useCallback((epochs: number, batchSize: number) => {
    trainingMutation.mutate({ data: { epochs, batchSize } }, {
      onSuccess: () => {
        void queryClient.invalidateQueries({ queryKey: getGetTrainingStatusQueryKey() });
        void queryClient.invalidateQueries({ queryKey: getGetGestureModelStatusQueryKey() });
      },
    });
  }, [queryClient, trainingMutation.mutate]);

  const props: WorkspaceContextValue = {
    labels,
    labelsLoading: labelsQuery.isLoading,
    labelsError: labelsQuery.isError,
    retryLabels: () => { void labelsQuery.refetch(); },
    model: modelQuery.data,
    modelLoading: modelQuery.isLoading,
    modelError: modelQuery.isError,
    retryModel: () => { void modelQuery.refetch(); },
    samples,
    setSamples,
    uploadedResult,
    setUploadedResult,
    importState,
    training: trainingQuery.data,
    trainingLoading: trainingQuery.isLoading,
    trainingError: trainingQuery.isError,
    retryTraining: () => { void trainingQuery.refetch(); },
    trainingMutation,
    healthOnline: healthQuery.data?.status?.toLowerCase() === 'ok' || healthQuery.data?.status?.toLowerCase() === 'healthy',
    healthDisplay: healthQuery.data?.status
      ? healthQuery.data.status.toUpperCase()
      : healthQuery.isLoading ? 'CHECKING' : 'OFFLINE',
    prediction,
    predictionError,
    confidenceThreshold,
    setConfidenceThreshold,
    recognitionHistory,
    recognizedText,
    clearRecognitionHistory,
    camera,
    speechEnabled,
    setSpeechEnabled,
    onImport: handleImport,
    onTrain: handleTraining,
  };

  return (
    <RoutedErrorBoundary>
      <WorkspaceContext.Provider value={props}>
        <AppShell healthOnline={props.healthOnline} healthDisplay={props.healthDisplay}>
          <Switch>
            <Route path="/" component={LiveRoute} />
            <Route path="/dataset" component={DatasetRoute} />
            <Route component={NotFound} />
          </Switch>
        </AppShell>
      </WorkspaceContext.Provider>
    </RoutedErrorBoundary>
  );
}

function LiveRoute() {
  const workspace = useContext(WorkspaceContext);
  return workspace ? <LiveWorkbench {...workspace} /> : null;
}

function DatasetRoute() {
  const workspace = useContext(WorkspaceContext);
  return workspace ? <DatasetWorkbench {...workspace} onImport={workspace.onImport} onTrain={workspace.onTrain} /> : null;
}

function RoutedErrorBoundary({ children }: { children: ReactNode }) {
  const [location] = useLocation();
  return <ErrorBoundary resetKey={location}>{children}</ErrorBoundary>;
}

function AppShell({ children, healthOnline, healthDisplay }: { children: ReactNode; healthOnline: boolean; healthDisplay: string }) {
  const [location] = useLocation();
  return (
    <div className="app-shell">
      <aside className="side-rail">
        <Link href="/" className="brand-lockup" aria-label="Handstudy home">
          <span className="brand-mark"><Fingerprint size={19} strokeWidth={1.7} /></span>
          <span className="brand-title">Handstudy<br />Lab notebook</span>
        </Link>
        <div className="rail-caption">Workspace</div>
        <nav className="rail-nav" aria-label="Main navigation">
          <Link href="/" className={`nav-item ${location === '/' ? 'active' : ''}`} data-testid="link-live-workbench">
            <Activity size={17} /> <span>Live workbench</span>
          </Link>
          <Link href="/dataset" className={`nav-item ${location === '/dataset' ? 'active' : ''}`} data-testid="link-dataset">
            <Database size={17} /> <span>Dataset & training</span>
          </Link>
        </nav>
        <div className="rail-bottom">
          <div className="rail-status"><span className={`status-dot ${healthOnline ? '' : 'offline'}`} /><span>API {healthDisplay.toLowerCase()}</span></div>
          <p className="rail-note">BTech AIML · independent hand-gesture research</p>
        </div>
      </aside>
      <div className="workspace">
        <header className="topbar">
          <div className="topbar-title">Hand landmark recognition / <span className="mono">research workspace</span></div>
          <div className="connection-pill" data-testid="status-api-connectivity">
            {healthOnline ? <Wifi size={13} /> : <Radio size={13} />}
            {`NODE API · ${healthDisplay}`}
          </div>
        </header>
        <main className="content">{children}</main>
      </div>
    </div>
  );
}

function LiveWorkbench(props: WorkspaceProps) {
  const {
    camera, model, modelLoading, modelError, retryModel, labels, labelsLoading, labelsError,
    samples, setSamples, prediction, predictionError, confidenceThreshold, setConfidenceThreshold,
    recognitionHistory, recognizedText, clearRecognitionHistory, speechEnabled, setSpeechEnabled,
  } = props;
  const [selectedLabel, setSelectedLabel] = useState('');
  useEffect(() => {
    if (labels.length && !labels.includes(selectedLabel)) setSelectedLabel(labels[0]);
  }, [labels, selectedLabel]);

  const captureSample = () => {
    if (!selectedLabel || !labels.includes(selectedLabel) || !camera.features || !camera.handFound || camera.features.length !== 63 || !camera.features.every(Number.isFinite)) return;
    setSamples([...samples, { label: selectedLabel, features: [...camera.features] }]);
  };
  const handCount = camera.handFound ? '1 detected' : 'no hand';
  const featureCount = camera.features ? camera.features.length : 0;
  const probabilities = useMemo(() => prediction?.probabilities
    ? Object.entries(prediction.probabilities).filter(([, value]) => Number.isFinite(value)).sort((a, b) => b[1] - a[1]).slice(0, 4)
    : [], [prediction]);
  const ready = model?.modelReady === true;
  const predictionAccepted = Boolean(prediction && Number.isFinite(prediction.confidence) && prediction.confidence >= confidenceThreshold);
  const canCapture = camera.cameraState === 'tracking' && camera.handFound && Boolean(selectedLabel);

  return (
    <>
      <div className="page-heading reveal">
        <div>
          <div className="eyebrow">01 / Real-time observation</div>
          <h1 className="page-title">Live hand workbench</h1>
          <p className="page-subtitle">Track a real hand, inspect normalized landmarks, and collect labeled examples. Recognition only appears when the server reports a trained model.</p>
        </div>
        <Link href="/dataset" className="button button-soft" data-testid="link-open-dataset">
          Dataset & training <ChevronRight size={15} />
        </Link>
      </div>

      <section className="camera-grid reveal-delay" aria-label="Camera and model output">
        <div className="panel camera-panel">
          <div className="camera-stage" data-testid="status-camera-preview">
            <video ref={camera.videoRef} muted playsInline aria-label="Live mirrored camera preview" />
            <canvas ref={camera.canvasRef} aria-hidden="true" />
            {camera.cameraState !== 'tracking' && (
              <div className="camera-placeholder">
                <div className="placeholder-glyph">{camera.cameraState === 'error' ? <AlertCircle size={24} /> : <Hand size={25} />}</div>
                <h3>{camera.cameraState === 'requesting' ? 'Waiting for permission' : camera.cameraState === 'loading-model' ? 'Preparing landmark model' : camera.cameraState === 'error' ? 'Camera not ready' : 'Camera is paused'}</h3>
                <p>{camera.cameraError || (camera.cameraState === 'loading-model'
                  ? 'MediaPipe is loading its hand-landmarker task and WebAssembly runtime from the public CDN.'
                  : 'Start the camera to detect 21 hand landmarks in your browser. Video stays on this device.')}</p>
              </div>
            )}
            <div className="camera-overlay">
              <span className="camera-chip"><span className={`status-dot ${camera.cameraState === 'tracking' ? '' : 'offline'}`} />{camera.cameraState === 'tracking' ? 'CAMERA ACTIVE' : camera.cameraState.replace('-', ' ').toUpperCase()}</span>
              <span className="camera-chip"><span className="live-dot" />{handCount}</span>
            </div>
          </div>
          <div className="camera-controls">
            <div className="control-group">
              {camera.cameraState === 'idle' || camera.cameraState === 'error' ? (
                <button className="button button-primary" onClick={() => void camera.start()} data-testid="button-start-camera">
                  <Camera size={15} /> Start camera
                </button>
              ) : (
                <button className="button" onClick={camera.stop} data-testid="button-stop-camera">
                  <CameraOff size={15} /> Stop camera
                </button>
              )}
              <button className="button button-soft" onClick={captureSample} disabled={!canCapture} data-testid="button-capture-sample">
                <Hand size={15} /> Capture sample
              </button>
            </div>
            <div className="camera-meta"><span>{handCount}</span><span>{featureCount} / 63 values</span></div>
          </div>
        </div>

        <div className="panel recognition-panel">
          <div className="panel-head" style={{ padding: '0 0 15px' }}>
            <div><div className="panel-title">Recognition output</div><div className="panel-kicker" style={{ marginTop: 5 }}>server model only</div></div>
            <div className="panel-kicker">LIVE</div>
          </div>
          {modelLoading ? (
            <div className="skeleton" style={{ height: 70 }} aria-label="Loading model status" />
          ) : modelError ? (
            <div className="model-state unready" data-testid="status-model-unavailable">
              <AlertCircle className="state-icon" size={17} />
              <div><div className="state-title">Model status unavailable</div><div className="state-copy">The API did not return a model state. Predictions are paused until it responds.</div><button className="button" onClick={retryModel} style={{ marginTop: 9 }} data-testid="button-retry-model"><RefreshCw size={13} /> Retry</button></div>
            </div>
          ) : ready ? (
            <div className="model-state" data-testid="status-model-ready">
              <ShieldCheck className="state-icon" size={17} />
              <div><div className="state-title">Trained model ready</div><div className="state-copy">{model?.message || 'The API confirms a model is loaded. Predictions below come from the server.'}</div></div>
            </div>
          ) : (
            <div className="model-state unready" data-testid="status-model-not-trained">
              <AlertCircle className="state-icon" size={17} />
              <div><div className="state-title">No trained model available</div><div className="state-copy">{model?.message || 'The server has not reported a trained model. Captured landmarks are measurements, not predictions.'}</div></div>
            </div>
          )}
          <div className="prediction-box" aria-live="polite" data-testid="status-prediction">
            {prediction && ready ? (
              <>
                <div className="panel-kicker">API PREDICTION</div>
                <div className={`prediction-label ${predictionAccepted ? '' : 'prediction-unknown'}`}>{predictionAccepted ? formatGestureLabel(prediction.label) : 'Unknown'}</div>
                <div className="prediction-score">
                  confidence · {Number.isFinite(prediction.confidence) ? prediction.confidence.toFixed(4) : 'not reported'}
                  {!predictionAccepted && Number.isFinite(prediction.confidence) ? ` · below ${(confidenceThreshold * 100).toFixed(0)}% threshold` : ''}
                </div>
              </>
            ) : (
              <div className="prediction-empty">
                {predictionError ? `Prediction request failed: ${predictionError}` : ready
                  ? 'Show a hand to request a classification from the trained API model.'
                  : 'No label or confidence will be shown until the API confirms model readiness.'}
              </div>
            )}
          </div>
          {probabilities.length > 0 && ready && (
            <div className="probability-list" aria-label="Model probabilities">
              {probabilities.map(([label, value]) => (
                <div className="prob-row" key={label} data-testid={`row-probability-${label}`}>
                  <span>{label}</span><div className="prob-track"><div className="prob-fill" style={{ transform: `scaleX(${Math.max(0, Math.min(1, value))})` }} /></div><span style={{ textAlign: 'right' }}>{value.toFixed(3)}</span>
                </div>
              ))}
            </div>
          )}
          <div className="confidence-control">
            <label className="toggle-line" htmlFor="confidence-threshold">
              <span>Minimum confidence</span>
              <strong>{(confidenceThreshold * 100).toFixed(0)}%</strong>
            </label>
            <input
              id="confidence-threshold"
              className="confidence-range"
              type="range"
              min="0.3"
              max="0.95"
              step="0.05"
              value={confidenceThreshold}
              onChange={(event) => setConfidenceThreshold(Number(event.target.value))}
              aria-label="Minimum prediction confidence"
              data-testid="input-confidence-threshold"
            />
            <div className="help-text">Predictions below this value are shown as Unknown and are not spoken.</div>
          </div>
          <div className="section-divider" />
          <div className="toggle-line">
            <span style={{ display: 'inline-flex', alignItems: 'center', gap: 8 }}>{speechEnabled ? <Volume2 size={15} /> : <VolumeX size={15} />} Speak API predictions</span>
            <input className="toggle" type="checkbox" checked={speechEnabled} onChange={(event) => setSpeechEnabled(event.target.checked)} aria-label="Speak API predictions" data-testid="input-speech-output" />
          </div>
          <div className="help-text" style={{ marginTop: 8 }}>Speech uses your browser’s voice synthesis and only speaks labels returned by the API.</div>
        </div>
      </section>

      <section className="instrument-grid" aria-label="Live measurements">
        <Metric label="Landmarks" value={camera.handFound ? '21 points' : '—'} detail="MediaPipe detection" />
        <Metric label="Feature vector" value={camera.features ? `${camera.features.length} values` : '—'} detail="wrist-origin · scale-normalized" />
        <Metric label="Model state" value={modelLoading ? 'Loading' : modelError ? 'Unavailable' : ready ? 'Ready' : model ? 'Not trained' : 'Unknown'} detail={model?.modelPath ? `model path · ${model.modelPath}` : 'reported by Python ML API'} />
      </section>

      <section className="history-grid" aria-label="Recognized text and history">
        <div className="panel recognized-text-panel">
          <div className="panel-head">
            <div>
              <div className="panel-title">Recognized text</div>
              <div className="panel-kicker" style={{ marginTop: 5 }}>confident static class labels · not sentence translation</div>
            </div>
            <button
              className="button button-danger"
              onClick={clearRecognitionHistory}
              disabled={!recognitionHistory.length}
              data-testid="button-clear-recognition-history"
            >
              <Trash2 size={14} /> Clear history
            </button>
          </div>
          <div className="recognized-text-value" data-testid="text-recognized-text" aria-live="polite">
            {recognizedText || 'No confident gestures recognized yet.'}
          </div>
          <div className="help-text">Only model responses that meet the confidence threshold appear here. Class labels do not form translated sentences.</div>
        </div>
        <div className="panel recognition-history-panel">
          <div className="panel-head">
            <div><div className="panel-title">Recognition history</div><div className="panel-kicker" style={{ marginTop: 5 }}>latest {recognitionHistory.length} API result{recognitionHistory.length === 1 ? '' : 's'} · saved in this browser</div></div>
          </div>
          {recognitionHistory.length ? (
            <div className="recognition-history-list" aria-live="polite">
              {recognitionHistory.slice(0, 8).map((event) => (
                <div className="recognition-history-row" key={event.id} data-testid="row-recognition-history">
                  <div>
                    <strong className={event.accepted ? '' : 'history-unknown'}>{event.accepted ? formatGestureLabel(event.label) : 'Unknown'}</strong>
                    <span>{new Date(event.occurredAt).toLocaleTimeString()}</span>
                  </div>
                  <span className="recognition-history-confidence">
                    {Number.isFinite(event.confidence) ? `${(event.confidence * 100).toFixed(1)}%` : '—'}
                  </span>
                </div>
              ))}
            </div>
          ) : (
            <div className="empty-inline" data-testid="empty-recognition-history">History will appear when the API returns a prediction from a trained model.</div>
          )}
        </div>
      </section>

      <section className="collection-layout">
        <div className="panel collection-panel">
          <div className="collection-top">
            <div><div className="panel-title">Labelled sample collection</div><div className="help-text" style={{ marginTop: 4 }}>Each capture stores the current real 63-value hand vector in this browser session.</div></div>
            <span className="capture-count" data-testid="text-local-sample-count">{samples.length} unsynced</span>
          </div>
          {labelsLoading ? (
            <div className="skeleton" style={{ height: 36, marginTop: 14 }} aria-label="Loading configured labels" />
          ) : labelsError ? (
            <div className="toast-note toast-error" data-testid="error-labels">Could not load configured labels. <button className="button" onClick={props.retryLabels} data-testid="button-retry-labels">Retry</button></div>
          ) : labels.length === 0 ? (
            <div className="empty-inline" data-testid="empty-labels">The API has not configured any gesture labels. Sample capture is unavailable until labels are provided.</div>
          ) : (
            <div className="capture-row">
              <select className="select" value={selectedLabel} onChange={(event) => setSelectedLabel(event.target.value)} aria-label="Label for captured sample" data-testid="select-sample-label">
                {labels.map((label) => <option key={label} value={label}>{label}</option>)}
              </select>
              <span className="button button-soft" aria-label="Selected configured label"><Check size={13} /> API label</span>
            </div>
          )}
          {labels.length > 0 && <div className="label-chip selected" style={{ display: 'inline-flex', marginTop: 10 }}>{selectedLabel || 'Select a configured label'}</div>}
          {samples.length > 0 ? (
            <div className="sample-list" aria-live="polite">
              {samples.slice(-5).reverse().map((sample, index) => (
                <div className="sample-row" key={`${sample.label}-${samples.length - index}`} data-testid={`row-collected-sample-${samples.length - index}`}>
                  <span>{sample.label}</span><span>63 finite features · local</span>
                </div>
              ))}
            </div>
          ) : (
            <div className="empty-inline" data-testid="empty-local-samples">No samples collected in this browser yet.</div>
          )}
        </div>
        <div className="info-banner">
          <strong>Measurement, not inference.</strong><br />
          Landmark tracking and video stay in your browser. The first MediaPipe task download may take a moment; when the server reports a trained model, one normalized feature vector is sent for prediction. Labeled vectors are sent only when you explicitly upload the dataset.
        </div>
      </section>
    </>
  );
}

function Metric({ label, value, detail }: { label: string; value: string; detail: string }) {
  return (
    <div className="panel metric-panel" data-testid={`metric-${label.toLowerCase().replace(/\s+/g, '-')}`}>
      <div className="metric-label">{label}</div><div className="metric-value">{value}</div><div className="help-text">{detail}</div>
    </div>
  );
}

function parseCsvRow(line: string): string[] {
  const fields: string[] = [];
  let field = '';
  let quoted = false;
  for (let index = 0; index < line.length; index += 1) {
    const character = line[index];
    if (character === '"' && quoted && line[index + 1] === '"') {
      field += '"';
      index += 1;
    } else if (character === '"') {
      quoted = !quoted;
    } else if (character === ',' && !quoted) {
      fields.push(field);
      field = '';
    } else {
      field += character;
    }
  }
  fields.push(field);
  return fields;
}

function csvEscape(value: string) {
  return `"${value.replaceAll('"', '""')}"`;
}

function DatasetWorkbench(props: WorkspaceProps & { onImport: (samples: GestureSample[]) => void; onTrain: (epochs: number, batchSize: number) => void }) {
  const { model, modelLoading, modelError, retryModel, labels, labelsLoading, labelsError, retryLabels, samples, setSamples, uploadedResult, importState, training, trainingLoading, trainingError, retryTraining, trainingMutation, onImport, onTrain } = props;
  const fileRef = useRef<HTMLInputElement>(null);
  const [fileError, setFileError] = useState('');
  const [fileSuccess, setFileSuccess] = useState('');
  const [epochs, setEpochs] = useState(30);
  const [batchSize, setBatchSize] = useState(32);
  const classCounts = uploadedResult?.classCounts ?? [];
  const serverSamples = model?.datasetSamples ?? 0;
  const classesWithSamples = model?.classesWithSamples ?? 0;
  const pendingByLabel = useMemo(() => samples.reduce<Record<string, number>>((counts, sample) => {
    counts[sample.label] = (counts[sample.label] ?? 0) + 1;
    return counts;
  }, {}), [samples]);
  const trainingIsActive = Boolean(training && ['running', 'pending', 'queued', 'starting'].includes(training.status.toLowerCase()));
  const canTrain = !modelError && !modelLoading && serverSamples > 0 && !trainingIsActive && !trainingMutation.isPending;

  const onFileSelected = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    setFileError('');
    setFileSuccess('');
    if (!file) return;
    try {
      const text = await file.text();
      const rows = text.split(/\r?\n/).filter((row) => row.trim().length > 0);
      if (!rows.length) throw new Error('The selected CSV contains no rows.');
      const parsed = rows.map(parseCsvRow);
      const firstCell = parsed[0][0]?.trim().toLowerCase();
      const dataRows = firstCell === 'label' ? parsed.slice(1) : parsed;
      const validSamples: GestureSample[] = [];
      for (let rowIndex = 0; rowIndex < dataRows.length; rowIndex += 1) {
        const fields = dataRows[rowIndex];
        if (fields.length !== 64) throw new Error(`Row ${rowIndex + (firstCell === 'label' ? 2 : 1)} has ${Math.max(fields.length - 1, 0)} feature columns; expected 63.`);
        const label = fields[0].trim();
        const features = fields.slice(1).map((value) => Number(value.trim()));
        if (!label) throw new Error(`Row ${rowIndex + 1} has an empty label.`);
        if (labels.length && !labels.includes(label)) throw new Error(`Row ${rowIndex + (firstCell === 'label' ? 2 : 1)} uses unconfigured label "${label}".`);
        if (features.some((value) => !Number.isFinite(value))) throw new Error(`Row ${rowIndex + 1} contains a non-finite feature value.`);
        validSamples.push({ label, features });
      }
      if (!validSamples.length) throw new Error('No sample rows were found in the CSV.');
      setSamples([...samples, ...validSamples]);
      setFileSuccess(`${validSamples.length} validated sample${validSamples.length === 1 ? '' : 's'} added to the local collection. Upload to send them to the API.`);
    } catch (error) {
      setFileError(errorText(error));
    } finally {
      event.target.value = '';
    }
  };

  const exportCsv = () => {
    if (!samples.length) return;
    const header = ['label', ...Array.from({ length: 63 }, (_, index) => `f${index}`)].map(csvEscape).join(',');
    const body = samples.map((sample) => [csvEscape(sample.label), ...sample.features.map(String)].join(','));
    const blob = new Blob([[header, ...body].join('\n')], { type: 'text/csv;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = 'handstudy-landmark-samples.csv';
    anchor.click();
    URL.revokeObjectURL(url);
  };

  const uploadSamples = () => {
    setFileError('');
    if (samples.length) onImport(samples);
  };

  const clearSamples = () => {
    if (!samples.length) return;
    if (window.confirm(`Remove ${samples.length} locally collected samples? This does not delete samples already stored by the API.`)) setSamples([]);
  };

  return (
    <>
      <div className="page-heading reveal">
        <div>
          <div className="eyebrow">02 / Dataset operations</div>
          <h1 className="page-title">Dataset & training</h1>
          <p className="page-subtitle">Inspect API-reported dataset counts, stage normalized landmark samples, and start training against the server dataset.</p>
        </div>
        <Link href="/" className="button button-soft" data-testid="link-back-live">
          <Camera size={15} /> Back to live workbench
        </Link>
      </div>

      <div className="dataset-layout reveal-delay">
        <div>
          <section className="panel" aria-label="Dataset summary">
            <div className="panel-head">
              <div><div className="panel-title">Server dataset summary</div><div className="panel-kicker" style={{ marginTop: 5 }}>measured from model status API</div></div>
              <button className="button" onClick={retryModel} disabled={modelLoading} data-testid="button-refresh-dataset"><RefreshCw size={13} /> Refresh</button>
            </div>
            {modelLoading ? (
              <div style={{ padding: '0 19px 19px' }}><div className="skeleton" style={{ height: 70 }} aria-label="Loading dataset summary" /></div>
            ) : modelError ? (
              <div className="toast-note toast-error" style={{ margin: '0 19px 19px' }} data-testid="error-dataset-status">Dataset status could not be loaded. <button className="button" onClick={retryModel} data-testid="button-retry-dataset">Try again</button></div>
            ) : (
              <>
                <div className="dataset-summary">
                  <SummaryCell label="Server samples" value={String(serverSamples)} />
                  <SummaryCell label="Classes populated" value={String(classesWithSamples)} />
                  <SummaryCell label="Samples / class min." value={String(model?.minimumSamplesPerClass ?? 0)} />
                </div>
                <div className="table-wrap">
                  <table className="data-table">
                    <thead><tr><th>Configured label</th><th>Server count</th><th>Local staged</th></tr></thead>
                    <tbody>
                      {labelsLoading ? <tr><td colSpan={3}>Loading configured labels…</td></tr> : labelsError ? <tr><td colSpan={3}>Configured labels could not be loaded. <button className="button" onClick={retryLabels} data-testid="button-retry-dataset-labels">Retry</button></td></tr> : labels.length ? labels.map((label) => {
                        const reported = classCounts.find((item) => item.label === label);
                        return <tr key={label} data-testid={`row-dataset-label-${label}`}><td>{label}</td><td className="mono">{reported ? reported.count : uploadedResult ? 'not reported' : '—'}</td><td className="mono">{pendingByLabel[label] ?? 0}</td></tr>;
                      }) : <tr><td colSpan={3}>No configured labels were returned by the API.</td></tr>}
                    </tbody>
                  </table>
                </div>
                {model && <div className="help-text" style={{ padding: '12px 19px 17px' }} data-testid="text-model-status-message">{model.message}{model.modelPath ? ` · ${model.modelPath}` : ''}</div>}
              </>
            )}
          </section>

          <section className="panel" style={{ marginTop: 16 }} aria-label="Dataset import and export">
            <div className="panel-head">
              <div><div className="panel-title">Import / export samples</div><div className="panel-kicker" style={{ marginTop: 5 }}>CSV · label + 63 numeric features</div></div>
              <span className="panel-kicker">{samples.length} local</span>
            </div>
            <div style={{ padding: '0 19px 19px' }}>
              <div className="upload-zone">
                <FileUp size={20} color="hsl(var(--primary))" />
                <p>Import a CSV produced by this workbench or another compatible capture tool. Each sample must contain one label and exactly 63 finite feature values.</p>
                <input ref={fileRef} type="file" accept=".csv,text/csv" className="file-input" onChange={(event) => void onFileSelected(event)} data-testid="input-import-csv" />
                <button className="button" onClick={() => fileRef.current?.click()} data-testid="button-choose-csv"><Upload size={14} /> Choose CSV</button>
              </div>
              {fileError && <div className="toast-note toast-error" role="alert" data-testid="error-csv-import">{fileError}</div>}
              {fileSuccess && <div className="toast-note toast-success" role="status" data-testid="success-csv-import">{fileSuccess}</div>}
              {importState.isError && <div className="toast-note toast-error" role="alert" data-testid="error-api-upload">{errorText(importState.error)}</div>}
              {uploadedResult && <div className="toast-note toast-success" role="status" data-testid="success-api-upload">API accepted {uploadedResult.importedSamples} samples. Reported dataset total: {uploadedResult.totalSamples}.</div>}
              <div className="control-group" style={{ marginTop: 12 }}>
                <button className="button button-primary" onClick={uploadSamples} disabled={!samples.length || importState.isPending} data-testid="button-upload-dataset">
                  <Upload size={14} /> {importState.isPending ? 'Uploading…' : `Upload ${samples.length || ''} staged samples`}
                </button>
                <button className="button" onClick={exportCsv} disabled={!samples.length} data-testid="button-export-csv"><Download size={14} /> Export CSV</button>
                <button className="button button-danger" onClick={clearSamples} disabled={!samples.length} data-testid="button-clear-samples"><Trash2 size={14} /> Clear local</button>
              </div>
              {!samples.length && <div className="help-text" style={{ marginTop: 10 }} data-testid="empty-staged-dataset">No local samples to upload or export. Capture a hand on the live workbench or import a CSV.</div>}
            </div>
          </section>
        </div>

        <aside>
          <section className="panel training-card" aria-label="Training controls">
            <div className="panel-head" style={{ padding: '0 0 15px' }}>
              <div><div className="panel-title">Training run</div><div className="panel-kicker" style={{ marginTop: 5 }}>server-side Keras classifier</div></div>
              <FlaskConical size={18} color="hsl(var(--primary))" />
            </div>
            {trainingLoading ? (
              <div className="skeleton" style={{ height: 74 }} aria-label="Loading training status" />
            ) : trainingError ? (
              <div className="toast-note toast-error" data-testid="error-training-status">Training status unavailable. <button className="button" onClick={retryTraining} data-testid="button-retry-training-status">Retry</button></div>
            ) : training ? (
              <div className="training-status" data-testid="status-training">
                <div><strong>{training.status}</strong><div className="help-text" style={{ marginTop: 4 }}>{training.message || 'No additional status detail.'}</div></div>
                <span>{training.accuracy === null ? 'evaluation pending' : `${(training.accuracy * 100).toFixed(1)}% held-out`}</span>
              </div>
            ) : (
              <div className="training-status" data-testid="empty-training-status"><strong>No run reported</strong><span>—</span></div>
            )}
            {training?.accuracy !== null && training?.accuracy !== undefined && training.testSamples !== null && (
              <>
                <div className="evaluation-summary" data-testid="status-model-evaluation">
                  <div className="evaluation-cell">
                    <span>Held-out test accuracy</span>
                    <strong>{(training.accuracy * 100).toFixed(1)}%</strong>
                  </div>
                  <div className="evaluation-cell">
                    <span>Test samples</span>
                    <strong>{training.testSamples}</strong>
                  </div>
                </div>
                <div className="help-text" style={{ marginTop: 8 }}>
                  Measured on the stratified test split held out during training. This does not establish real-world accuracy.
                </div>
              </>
            )}
            {training && !trainingLoading && !trainingError && (
              <>
                <div className="progress-track" aria-label={`Training progress ${training.progress}`}><div className="progress-value" style={{ transform: `scaleX(${Math.max(0, Math.min(100, training.progress)) / 100})` }} /></div>
                <div className="help-text mono" data-testid="text-training-progress">progress · {training.progress}{training.startedAt ? ` · started ${training.startedAt}` : ''}{training.completedAt ? ` · completed ${training.completedAt}` : ''}</div>
              </>
            )}
            <div className="section-divider" />
            <div className="training-fields">
              <label><span className="field-label">Epochs</span><input className="input" type="number" min={5} max={100} value={epochs} onChange={(event) => setEpochs(Number(event.target.value))} data-testid="input-training-epochs" /></label>
              <label><span className="field-label">Batch size</span><input className="input" type="number" min={8} max={128} value={batchSize} onChange={(event) => setBatchSize(Number(event.target.value))} data-testid="input-training-batch-size" /></label>
            </div>
            <button className="button button-primary" style={{ width: '100%' }} disabled={!canTrain || epochs < 5 || epochs > 100 || batchSize < 8 || batchSize > 128} onClick={() => onTrain(epochs, batchSize)} data-testid="button-start-training">
              <FlaskConical size={15} /> {trainingMutation.isPending ? 'Starting…' : 'Start training'}
            </button>
            {trainingMutation.isError && <div className="toast-note toast-error" role="alert" data-testid="error-start-training">{errorText(trainingMutation.error)}</div>}
            {!modelError && !modelLoading && serverSamples === 0 && <div className="help-text" style={{ marginTop: 10 }} data-testid="hint-training-data">Upload samples first. Local staged examples are not yet part of the server dataset.</div>}
            {model?.datasetSamples ? <div className="help-text" style={{ marginTop: 10 }}>The API reports {serverSamples} dataset samples across {classesWithSamples} populated classes. Minimum per-class target: {model?.minimumSamplesPerClass}.</div> : null}
            <div className="section-divider" />
            <div className="help-text" style={{ display: 'flex', gap: 8 }}><CircleHelp size={14} style={{ flex: 'none' }} /> A model is considered ready only when the status API says so. Training completion and recognition readiness are separate server-reported states.</div>
          </section>

          <section className="panel" style={{ padding: 17, marginTop: 15 }} aria-label="Data handling details">
            <div className="panel-kicker">Data handling</div>
            <div style={{ display: 'grid', gap: 11, marginTop: 14 }}>
              <DataNote icon={<Hand size={15} />} title="Browser capture" body="Landmarks are detected on-device. Video is never uploaded." />
              <DataNote icon={<Database size={15} />} title="Explicit upload" body="Labeled feature vectors are sent to the dataset API only after you choose upload." />
              <DataNote icon={<ShieldCheck size={15} />} title="No synthetic output" body="Predictions and model state come from the generated API client." />
            </div>
          </section>
        </aside>
      </div>
    </>
  );
}

function SummaryCell({ label, value }: { label: string; value: string }) {
  return <div className="summary-cell"><div className="summary-value" data-testid={`summary-${label.toLowerCase().replace(/[^a-z]+/g, '-')}`}>{value}</div><div className="summary-label">{label}</div></div>;
}

function DataNote({ icon, title, body }: { icon: ReactNode; title: string; body: string }) {
  return <div style={{ display: 'flex', gap: 10, alignItems: 'flex-start' }}><span style={{ color: 'hsl(var(--primary))' }}>{icon}</span><div><div style={{ fontSize: 11, fontWeight: 700 }}>{title}</div><div className="help-text" style={{ marginTop: 3 }}>{body}</div></div></div>;
}

export default App;