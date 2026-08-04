import React, { useState, useEffect, useCallback, useRef } from 'react';
import SlicePanel from './components/SlicePanel';
import InfoPanel from './components/InfoPanel';
import CTImageViewer from './components/CTImageViewer';
import FolderUpload from './components/FolderUpload';
import { loadFilesAsSlices, loadSliceFromBlob, sliceToBlob } from './lib/mriLoader';
import { classifyByModality, filesFromDataTransferItems, folderNameFromFiles } from './lib/patientFolder';
import { MRI_PATIENT } from './lib/mockData';
import { apiUrl } from './lib/api';

// 참고: 예측 체크포인트는 현재 자리표시자(원래 T2→T1용으로 학습된 것)이며,
// 실제 T1→T2 체크포인트로 교체 예정이다.

export default function App() {
  const [t1Slices, setT1Slices]         = useState([]); // 원본 T1
  const [t2Slices, setT2Slices]         = useState([]); // 원본 T2 (ground truth)
  const [currentIndex, setCurrentIndex] = useState(0);
  const [cursorInfo, setCursorInfo]     = useState(null);
  const [loadProgress, setLoadProgress] = useState(0);
  const [loading, setLoading]           = useState(false);
  const [error, setError]               = useState(null);
  const [patientName, setPatientName]   = useState('');

  // 밝기/대비 — 패널 전체가 공유
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
  const [modelError, setModelError] = useState(null);
  const modelReadyRef  = useRef(false);
  const modelErrorRef  = useRef(null);
  const modelPollRef   = useRef(null);

  useEffect(() => {
    const poll = async () => {
      try {
        const res = await fetch(apiUrl('/api/status'));
        if (res.ok) {
          const data = await res.json();
          if (data.model_ready) {
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

  // 업로드된 File[] → T1/T2로 분류 후 로드
  const loadPatientFiles = useCallback(async (files) => {
    const { t1, t2 } = classifyByModality(files);
    if (t1.length === 0 && t2.length === 0) {
      setError('T1/T2로 인식되는 PNG·JPG 파일을 찾지 못했어요 (파일명에 t1 또는 t2가 포함되어야 함)');
      return;
    }

    setLoading(true);
    setError(null);
    setCurrentIndex(0);
    setPredSlices([]);
    setPredStatus('idle');
    setPredProgress(0);
    setPredFailCount(0);
    setPatientName(folderNameFromFiles(files) || '업로드된 환자');

    const total = t1.length + t2.length || 1;
    let done = 0;
    const tick = () => { done++; setLoadProgress(Math.round((done / total) * 100)); };

    try {
      const [t1Loaded, t2Loaded] = await Promise.all([
        loadFilesAsSlices(t1, tick),
        loadFilesAsSlices(t2, tick),
      ]);
      setT1Slices(t1Loaded);
      setT2Slices(t2Loaded);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, []);

  const sliceCount = Math.max(t1Slices.length, t2Slices.length);

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
    if (sliceCount === 0) return;
    e.preventDefault();
    navigate(e.deltaY > 0 ? 1 : -1);
  }, [navigate, sliceCount]);

  // 중앙 영역에 폴더를 직접 드래그&드롭
  const [dragOver, setDragOver] = useState(false);
  const onCenterDragOver = useCallback((e) => { e.preventDefault(); setDragOver(true); }, []);
  const onCenterDragLeave = useCallback(() => setDragOver(false), []);
  const onCenterDrop = useCallback(async (e) => {
    e.preventDefault();
    setDragOver(false);
    const files = await filesFromDataTransferItems(e.dataTransfer.items);
    if (files.length > 0) loadPatientFiles(files);
  }, [loadPatientFiles]);

  // 예측 버튼 → T1 전체 슬라이스에 대해 T1→T2 추론
  const runPrediction = useCallback(async () => {
    if (predStatus === 'model_loading' || predStatus === 'predicting') return;
    if (t1Slices.length === 0) return;
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

    const total = t1Slices.length;
    const results = new Array(total).fill(null);
    setPredSlices([...results]);

    // 지금 보고 있는 슬라이스부터 먼저 추론해서 바로 결과가 보이게 하고,
    // 나머지는 순서대로 이어서 처리한다.
    const order = Array.from({ length: total }, (_, k) => (currentIndex + k) % total);

    let doneCount = 0;
    let failCount = 0;
    for (const i of order) {
      if (predCancelRef.current) break;
      try {
        const pngBlob = await sliceToBlob(t1Slices[i]);
        const res = await fetch(apiUrl('/api/infer-t1'), {
          method: 'POST',
          headers: { 'Content-Type': 'image/png' },
          body: pngBlob,
        });
        if (!res.ok) { const j = await res.json().catch(() => ({})); throw new Error(j.error || `HTTP ${res.status}`); }
        const slice = await loadSliceFromBlob(await res.blob());
        results[i] = slice;
        setPredSlices([...results]);
      } catch (e) {
        failCount++;
        setPredFailCount(failCount);
        console.error(`예측 슬라이스 ${i}:`, e.message);
      }
      doneCount++;
      setPredProgress(doneCount);
    }
    if (!predCancelRef.current) setPredStatus(failCount === total ? 'error' : 'done');
  }, [t1Slices, predStatus, currentIndex]);

  const predPending = (predStatus === 'model_loading' || predStatus === 'predicting') && !predSlices[currentIndex];
  const predFailedSlice = (predStatus === 'done' || predStatus === 'error') && !predSlices[currentIndex];
  const predictDisabled = loading || t1Slices.length === 0 || predStatus === 'model_loading' || predStatus === 'predicting';
  const hasAnyData = t1Slices.length > 0 || t2Slices.length > 0;

  // 썸네일/정보/3D는 T1을 기준으로 하되, T1이 없으면 T2로 대체
  const refSlices = t1Slices.length > 0 ? t1Slices : t2Slices;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh', background: '#0a0a1a', color: '#e0e0e0', overflow: 'hidden' }}>

      {/* Header */}
      <div style={{ height: 44, background: '#0d0d22', borderBottom: '1px solid #2a2a4a', display: 'flex', alignItems: 'center', padding: '0 16px', gap: 16, flexShrink: 0 }}>
        <span style={{ color: '#ce93d8', fontWeight: 'bold', fontSize: 15, letterSpacing: 1 }}>MRI T1→T2 뷰어</span>
        <span style={{ color: '#555', fontSize: 11 }}>|</span>
        <span style={{ color: '#aaa', fontSize: 12 }}>{patientName || '환자 미선택'}</span>

        <FolderUpload onFilesSelected={loadPatientFiles} disabled={loading} />

        <button
          onClick={runPrediction}
          disabled={predictDisabled}
          style={{
            padding: '5px 16px', fontSize: 12, fontWeight: 'bold',
            background: predictDisabled ? '#1a1a2a' : '#3a1a4a',
            color: predictDisabled ? '#555' : '#ce93d8',
            border: `1px solid ${predictDisabled ? '#2a2a3a' : '#7a3a9a'}`,
            borderRadius: 4, cursor: predictDisabled ? 'default' : 'pointer',
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
            T2 예측 중... {predProgress}/{t1Slices.length} ({Math.round((predProgress / t1Slices.length) * 100)}%)
            {predFailCount > 0 && <span style={{ color: '#f87171' }}>({predFailCount}개 실패)</span>}
          </span>
        )}
        {predStatus === 'done' && (
          predFailCount === 0
            ? <span style={{ color: '#81c784', fontSize: 11 }}>T2 예측 완료 ✓ ({t1Slices.length}장)</span>
            : <span style={{ color: '#ffcc80', fontSize: 11 }}>T2 예측 부분 완료 — {t1Slices.length - predFailCount}장 성공 / {predFailCount}장 실패</span>
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

      {/* 예측 진행률 — 눈에 잘 띄는 전체 폭 바 */}
      {(predStatus === 'predicting' || predStatus === 'model_loading') && (
        <div style={{ height: 4, background: '#0d0d22', flexShrink: 0 }}>
          <div style={{
            height: '100%',
            width: predStatus === 'predicting' ? `${(predProgress / t1Slices.length) * 100}%` : '100%',
            background: '#ce93d8',
            opacity: predStatus === 'model_loading' ? 0.5 : 1,
            animation: predStatus === 'model_loading' ? 'pulse 1.2s ease-in-out infinite' : 'none',
            transition: 'width 0.2s linear',
          }} />
          <style>{`@keyframes pulse { 0%, 100% { opacity: 0.25; } 50% { opacity: 0.7; } }`}</style>
        </div>
      )}

      {/* Main layout */}
      <div style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>

        {/* 슬라이스 썸네일 */}
        <SlicePanel slices={refSlices} currentIndex={currentIndex} onSelect={setCurrentIndex} />

        <InfoPanel
          patient={{ ...MRI_PATIENT, id: patientName || MRI_PATIENT.id, sliceCount: sliceCount || MRI_PATIENT.sliceCount }}
          sliceIndex={currentIndex}
          sliceData={refSlices[currentIndex] ?? null}
          slices={refSlices}
        />

        {/* 중앙: T1 원본 / T2 원본 / T2 예측 3단 비교 */}
        <div
          style={{ flex: 1, display: 'flex', borderRight: '1px solid #2a2a4a', overflow: 'hidden', position: 'relative' }}
          onWheel={handleWheel}
          onDragOver={onCenterDragOver}
          onDragLeave={onCenterDragLeave}
          onDrop={onCenterDrop}
        >
          {!hasAnyData && !loading ? (
            <div style={{
              flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 12,
              border: dragOver ? '2px dashed #ce93d8' : '2px dashed transparent',
              background: dragOver ? 'rgba(206,147,216,0.06)' : 'transparent',
            }}>
              <div style={{ fontSize: 40, opacity: 0.5 }}>📂</div>
              <div style={{ color: '#888', fontSize: 13 }}>환자 폴더를 여기로 드래그하거나</div>
              <FolderUpload onFilesSelected={loadPatientFiles} disabled={loading} />
              <div style={{ color: '#444', fontSize: 10, marginTop: 8, textAlign: 'center' }}>
                폴더 안 파일명에 t1 / t2가 포함되어 있어야 원본으로 인식됩니다<br />
                (예: Brats18_..._t1_0000.png, Brats18_..._t2_0000.png)
              </div>
            </div>
          ) : (
            <>
              <div style={{ flex: 1, borderRight: '1px solid #2a2a4a' }}>
                <CTImageViewer
                  sliceData={t1Slices[currentIndex] ?? null}
                  sliceIndex={currentIndex}
                  label="T1 원본"
                  accentColor="#81c784"
                  wc={wc} ww={ww} onWcWwChange={handleWcWwChange}
                  showToolbar
                  placeholder={t1Slices.length === 0}
                  placeholderText="업로드된 폴더에서 T1 파일을 찾지 못했습니다"
                  onCursorHU={setCursorInfo}
                />
              </div>
              <div style={{ flex: 1, borderRight: '1px solid #2a2a4a' }}>
                <CTImageViewer
                  sliceData={t2Slices[currentIndex] ?? null}
                  sliceIndex={currentIndex}
                  label="T2 원본"
                  accentColor="#64b5f6"
                  wc={wc} ww={ww} onWcWwChange={handleWcWwChange}
                  showToolbar={false}
                  placeholder={t2Slices.length === 0}
                  placeholderText="업로드된 폴더에서 T2 파일을 찾지 못했습니다"
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
                  pendingText={`T2 생성 중... (${predProgress}/${t1Slices.length})`}
                  failed={predFailedSlice}
                  failedText="이 슬라이스는 예측 실패"
                />
              </div>
            </>
          )}
        </div>
      </div>

      {/* Footer */}
      <div style={{ height: 24, background: '#050510', borderTop: '1px solid #1a1a3a', display: 'flex', alignItems: 'center', padding: '0 12px', gap: 16, fontSize: 10, color: '#555', flexShrink: 0 }}>
        <span>슬라이스: {sliceCount > 0 ? `${currentIndex + 1}/${sliceCount}` : '-'}</span>
        <span>|</span>
        <span>← → 키보드 또는 마우스 휠로 이동</span>
        <span>|</span>
        <span>우클릭 드래그: 밝기/대비 조정 (세 패널 공유)</span>
      </div>
    </div>
  );
}
