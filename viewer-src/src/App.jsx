import React, { useState, useEffect, useCallback, useRef } from 'react';
import SlicePanel from './components/SlicePanel';
import InfoPanel from './components/InfoPanel';
import CTImageViewer from './components/CTImageViewer';
import VolumeRenderer3D from './components/VolumeRenderer3D';
import { loadAllMRISlices, loadSliceFromBlob, sliceToBlob } from './lib/mriLoader';
import { MRI_PATIENT } from './lib/mockData';

// 참고: patient_mri/p1의 슬라이스는 원래 T2 PNG지만, 진짜 T1 원본이 준비되기
// 전까지 "T1 원본" 자리표시자로 쓰고 있다. 예측 체크포인트도 마찬가지로
// 자리표시자(T2→T1용으로 학습된 것)이며, 실제 T1→T2 체크포인트로 교체 예정이다.

export default function App() {
  const [slices, setSlices]             = useState([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [cursorInfo, setCursorInfo]     = useState(null);
  const [loadProgress, setLoadProgress] = useState(0);
  const [loading, setLoading]           = useState(true);
  const [error, setError]               = useState(null);

  // 밝기/대비 — 두 패널이 공유
  const [wc, setWc] = useState(128);
  const [ww, setWw] = useState(256);
  const handleWcWwChange = useCallback((nwc, nww) => { setWc(nwc); setWw(nww); }, []);

  // T2 예측
  const [predSlices, setPredSlices]     = useState([]);
  // 'idle' | 'model_loading' | 'predicting' | 'done' | 'error'
  const [predStatus, setPredStatus]     = useState('idle');
  const [predProgress, setPredProgress] = useState(0);
  const [predFailCount, setPredFailCount] = useState(0);
  const predCancelRef = useRef(false);

  // 모델 상태 폴링
  const [modelReady, setModelReady] = useState(false);
  const [modelError, setModelError] = useState(null);
  const modelReadyRef  = useRef(false);
  const modelErrorRef  = useRef(null);
  const modelPollRef   = useRef(null);

  useEffect(() => {
    const poll = async () => {
      try {
        const res = await fetch('/api/status');
        if (res.ok) {
          const data = await res.json();
          if (data.model_ready) {
            setModelReady(true);
            modelReadyRef.current = true;
            clearInterval(modelPollRef.current);
          } else if (data.model_error) {
            setModelError(data.model_error);
            modelErrorRef.current = data.model_error;
            clearInterval(modelPollRef.current);
          }
        }
      } catch { /* 서버 준비 전일 수 있음 */ }
    };
    poll();
    modelPollRef.current = setInterval(poll, 3000);
    return () => clearInterval(modelPollRef.current);
  }, []);

  // 환자 p1 슬라이스를 서버에서 자동 로드
  useEffect(() => {
    let cancelled = false;
    (async () => {
      setLoading(true);
      setError(null);
      try {
        const loaded = await loadAllMRISlices((done, total) => {
          if (!cancelled) setLoadProgress(Math.round((done / total) * 100));
        });
        if (!cancelled) setSlices(loaded);
      } catch (e) {
        if (!cancelled) setError(e.message);
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => { cancelled = true; };
  }, []);

  const sliceCount = slices.length || MRI_PATIENT.sliceCount;

  const navigate = useCallback((delta) => {
    setCurrentIndex(i => Math.max(0, Math.min(sliceCount - 1, i + delta)));
  }, [sliceCount]);

  const handleKeyDown = useCallback((e) => {
    if (e.key === 'ArrowUp'   || e.key === 'ArrowLeft')  navigate(-1);
    if (e.key === 'ArrowDown' || e.key === 'ArrowRight') navigate(1);
  }, [navigate]);

  useEffect(() => {
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [handleKeyDown]);

  const handleWheel = useCallback((e) => {
    e.preventDefault();
    navigate(e.deltaY > 0 ? 1 : -1);
  }, [navigate]);

  // 예측 버튼 → 전체 슬라이스 T1→T2 추론
  const runPrediction = useCallback(async () => {
    if (predStatus === 'model_loading' || predStatus === 'predicting') return;
    if (slices.length === 0) return;
    predCancelRef.current = false;

    if (!modelReadyRef.current && !modelErrorRef.current) {
      setPredStatus('model_loading');
      await new Promise((resolve) => {
        const check = setInterval(() => {
          if (predCancelRef.current || modelReadyRef.current || modelErrorRef.current) {
            clearInterval(check);
            resolve();
          }
        }, 1000);
      });
      if (predCancelRef.current) { setPredStatus('idle'); return; }
    }
    if (modelErrorRef.current) { setPredStatus('error'); return; }

    setPredStatus('predicting');
    setPredProgress(0);
    setPredFailCount(0);

    const total = slices.length;
    const results = new Array(total).fill(null);
    setPredSlices([...results]);

    let failCount = 0;
    for (let i = 0; i < total; i++) {
      if (predCancelRef.current) break;
      try {
        const pngBlob = await sliceToBlob(slices[i]);
        const res = await fetch('/api/infer-t1', {
          method: 'POST',
          headers: { 'Content-Type': 'image/png' },
          body: pngBlob,
        });
        if (!res.ok) { const j = await res.json().catch(() => ({})); throw new Error(j.error || `HTTP ${res.status}`); }
        const slice = await loadSliceFromBlob(await res.blob());
        results[i] = slice;
        setPredSlices([...results]);
        setPredProgress(i + 1);
      } catch (e) {
        failCount++;
        setPredFailCount(failCount);
        console.error(`예측 슬라이스 ${i}:`, e.message);
      }
    }
    if (!predCancelRef.current) setPredStatus(failCount === total ? 'error' : 'done');
  }, [slices, predStatus]);

  const predPending = (predStatus === 'model_loading' || predStatus === 'predicting') && !predSlices[currentIndex];
  const predFailedSlice = (predStatus === 'done' || predStatus === 'error') && !predSlices[currentIndex];
  const predictDisabled = loading || slices.length === 0 || predStatus === 'model_loading' || predStatus === 'predicting';

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh', background: '#0a0a1a', color: '#e0e0e0', overflow: 'hidden' }}>

      {/* Header */}
      <div style={{ height: 44, background: '#0d0d22', borderBottom: '1px solid #2a2a4a', display: 'flex', alignItems: 'center', padding: '0 16px', gap: 16, flexShrink: 0 }}>
        <span style={{ color: '#ce93d8', fontWeight: 'bold', fontSize: 15, letterSpacing: 1 }}>MRI T1→T2 뷰어</span>
        <span style={{ color: '#555', fontSize: 11 }}>|</span>
        <span style={{ color: '#aaa', fontSize: 12 }}>{MRI_PATIENT.id}</span>

        <button
          onClick={runPrediction}
          disabled={predictDisabled}
          style={{
            padding: '5px 16px', fontSize: 12, fontWeight: 'bold',
            background: predictDisabled ? '#1a1a2a' : '#3a1a4a',
            color: predictDisabled ? '#555' : '#ce93d8',
            border: `1px solid ${predictDisabled ? '#2a2a3a' : '#7a3a9a'}`,
            borderRadius: 4, cursor: predictDisabled ? 'default' : 'pointer',
            marginLeft: 8,
          }}
        >
          T2 예측 실행
        </button>

        {loading && <span style={{ color: '#ffcc80', fontSize: 11 }}>슬라이스 로딩 중... {loadProgress}%</span>}
        {error   && <span style={{ color: '#f87171', fontSize: 11 }}>⚠ {error}</span>}
        {modelError && (
          <span style={{ color: '#f87171', fontSize: 11 }}>모델 오류: {modelError}</span>
        )}
        {predStatus === 'model_loading' && (
          <span style={{ color: '#ffcc80', fontSize: 11, display: 'flex', alignItems: 'center', gap: 6 }}>
            <span style={{ display: 'inline-block', animation: 'spin 1s linear infinite' }}>⏳</span>
            모델 로딩 중...
          </span>
        )}
        {predStatus === 'predicting' && (
          <span style={{ color: '#ce93d8', fontSize: 11, display: 'flex', alignItems: 'center', gap: 6 }}>
            T2 예측 중... {predProgress}/{slices.length}
            {predFailCount > 0 && <span style={{ color: '#f87171' }}>({predFailCount}개 실패)</span>}
            <span style={{ display: 'inline-block', width: 80, height: 4, background: '#2a2a4a', borderRadius: 2 }}>
              <span style={{ display: 'block', width: `${(predProgress / slices.length) * 100}%`, height: '100%', background: '#ce93d8', borderRadius: 2, transition: 'width 0.3s' }} />
            </span>
          </span>
        )}
        {predStatus === 'done' && (
          predFailCount === 0
            ? <span style={{ color: '#81c784', fontSize: 11 }}>T2 예측 완료 ✓ ({slices.length}장)</span>
            : <span style={{ color: '#ffcc80', fontSize: 11 }}>T2 예측 부분 완료 — {slices.length - predFailCount}장 성공 / {predFailCount}장 실패</span>
        )}
        {predStatus === 'error' && (
          <span style={{ color: '#f87171', fontSize: 11 }}>T2 예측 실패 — 모델 상태를 확인하세요</span>
        )}

        <div style={{ marginLeft: 'auto' }}>
          {cursorInfo && (
            <span style={{ fontSize: 11, fontFamily: 'monospace', color: '#aaa' }}>
              X: {cursorInfo.x} &nbsp;Y: {cursorInfo.y} &nbsp;val: {cursorInfo.hu}
            </span>
          )}
        </div>
      </div>

      {/* Main layout */}
      <div style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>

        {/* 슬라이스 썸네일 */}
        <SlicePanel slices={slices} currentIndex={currentIndex} onSelect={setCurrentIndex} />

        <InfoPanel
          patient={{ ...MRI_PATIENT, sliceCount: slices.length || MRI_PATIENT.sliceCount }}
          sliceIndex={currentIndex}
          sliceData={slices[currentIndex] ?? null}
        />

        {/* 중앙: T1 원본 / T2 예측 나란히 비교 */}
        <div
          style={{ flex: 1, display: 'flex', borderRight: '1px solid #2a2a4a', overflow: 'hidden' }}
          onWheel={handleWheel}
        >
          <div style={{ flex: 1, borderRight: '1px solid #2a2a4a' }}>
            <CTImageViewer
              sliceData={slices[currentIndex] ?? null}
              sliceIndex={currentIndex}
              label="T1 원본"
              accentColor="#81c784"
              wc={wc} ww={ww} onWcWwChange={handleWcWwChange}
              showToolbar
              onCursorHU={setCursorInfo}
            />
          </div>
          <div style={{ flex: 1 }}>
            <CTImageViewer
              sliceData={predSlices[currentIndex] ?? null}
              sliceIndex={currentIndex}
              label="T2 예측"
              accentColor="#ce93d8"
              wc={wc} ww={ww} onWcWwChange={handleWcWwChange}
              showToolbar={false}
              placeholder={predStatus === 'idle' && !predSlices[currentIndex]}
              placeholderText="위의 'T2 예측 실행' 버튼을 눌러 생성"
              pending={predPending}
              pendingText="T2 생성 중..."
              failed={predFailedSlice}
              failedText="이 슬라이스는 예측 실패"
            />
          </div>
        </div>

        <div style={{ width: 280, flexShrink: 0, overflow: 'hidden' }}>
          <VolumeRenderer3D slices={slices} sliceIndex={currentIndex} />
        </div>
      </div>

      {/* Footer */}
      <div style={{ height: 24, background: '#050510', borderTop: '1px solid #1a1a3a', display: 'flex', alignItems: 'center', padding: '0 12px', gap: 16, fontSize: 10, color: '#555', flexShrink: 0 }}>
        <span>슬라이스: {currentIndex + 1}/{slices.length || '?'}</span>
        <span>|</span>
        <span>← → 키보드 또는 마우스 휠로 이동</span>
        <span>|</span>
        <span>우클릭 드래그: 밝기/대비 조정 (양쪽 패널 공유)</span>
      </div>
    </div>
  );
}
