import React, { useEffect, useRef, useCallback } from 'react';
import { renderSliceToCanvas } from '../lib/ctRenderer';

const PRESETS = [
  { label: '전체', wc: 128, ww: 256 },
  { label: '뇌',   wc: 100, ww: 180 },
  { label: '백질', wc: 170, ww: 80  },
  { label: '고대비', wc: 128, ww: 120 },
];

// 한 슬라이스 패널(원본 T1 또는 예측 T2)을 그리는 범용 뷰어.
// wc/ww는 부모에서 내려받는 controlled 값이라 두 패널이 밝기/대비를 공유한다.
export default function CTImageViewer({
  sliceData, sliceIndex, label, accentColor = '#81c784',
  wc, ww, onWcWwChange,
  pending, pendingText, placeholder, placeholderText, failed, failedText,
  showToolbar = true,
  onCursorHU,
}) {
  const canvasRef  = useRef(null);
  const dragging   = useRef(false);
  const dragStart  = useRef(null);
  const wcStart    = useRef(wc);
  const wwStart    = useRef(ww);

  useEffect(() => {
    if (!sliceData || !canvasRef.current) return;
    renderSliceToCanvas(canvasRef.current, sliceData.data, sliceData.width, sliceData.height, wc, ww);
  }, [sliceData, wc, ww]);

  const onMouseDown = useCallback((e) => {
    if (e.button !== 2) return;
    dragging.current = true;
    dragStart.current = { x: e.clientX, y: e.clientY };
    wcStart.current = wc;
    wwStart.current = ww;
  }, [wc, ww]);

  const onMouseMove = useCallback((e) => {
    if (sliceData && canvasRef.current) {
      const rect = canvasRef.current.getBoundingClientRect();
      const scaleX = sliceData.width / rect.width;
      const scaleY = sliceData.height / rect.height;
      const px = Math.floor((e.clientX - rect.left) * scaleX);
      const py = Math.floor((e.clientY - rect.top) * scaleY);
      if (px >= 0 && px < sliceData.width && py >= 0 && py < sliceData.height) {
        onCursorHU?.({ x: px, y: py, hu: sliceData.data[py * sliceData.width + px] });
      }
    }
    if (!dragging.current || !dragStart.current) return;
    const dx = e.clientX - dragStart.current.x;
    const dy = e.clientY - dragStart.current.y;
    onWcWwChange?.(
      Math.max(0, Math.min(255, Math.round(wcStart.current + dx))),
      Math.max(1, Math.min(255, Math.round(wwStart.current - dy))),
    );
  }, [sliceData, onCursorHU, onWcWwChange]);

  const onMouseUp = useCallback(() => { dragging.current = false; }, []);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', background: '#000', position: 'relative' }}>
      {/* Toolbar */}
      <div style={{ display: 'flex', gap: 4, padding: '6px 8px', background: '#0d0d1a', borderBottom: '1px solid #2a2a4a', alignItems: 'center' }}>
        <span style={{ fontSize: 12, fontWeight: 'bold', color: accentColor, marginRight: 8 }}>{label}</span>
        {showToolbar && PRESETS.map(p => (
          <button
            key={p.label}
            onClick={() => onWcWwChange?.(p.wc, p.ww)}
            style={{ fontSize: 10, padding: '2px 8px', background: '#1a1a3a', color: '#aaa', border: '1px solid #3a3a6a', borderRadius: 3, cursor: 'pointer' }}
          >
            {p.label}
          </button>
        ))}
        <div style={{ marginLeft: 'auto', fontSize: 10, color: '#666' }}>
          우클릭 드래그: 밝기/대비
        </div>
      </div>

      {/* Canvas */}
      <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', position: 'relative' }}>
        {!sliceData ? (
          <div style={{ color: '#333', fontSize: 13 }}>로딩 중...</div>
        ) : (
          <canvas
            ref={canvasRef}
            style={{ maxWidth: '100%', maxHeight: '100%', objectFit: 'contain', cursor: 'crosshair', display: 'block' }}
            onMouseDown={onMouseDown}
            onMouseMove={onMouseMove}
            onMouseUp={onMouseUp}
            onMouseLeave={onMouseUp}
            onContextMenu={e => e.preventDefault()}
          />
        )}

        {placeholder && (
          <div style={overlayStyle}>
            <span style={{ fontSize: 30, opacity: 0.5 }}>⇢</span>
            <span style={{ color: '#888', fontSize: 13, textAlign: 'center', maxWidth: 200 }}>
              {placeholderText || '예측 버튼을 눌러 생성'}
            </span>
          </div>
        )}

        {pending && (
          <div style={overlayStyle}>
            <div style={spinnerStyle} />
            <span style={{ color: accentColor, fontSize: 14, fontWeight: 600, letterSpacing: 1 }}>
              {pendingText || '추론 중...'}
            </span>
            <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
          </div>
        )}

        {failed && (
          <div style={{
            position: 'absolute', bottom: 36, left: '50%', transform: 'translateX(-50%)',
            background: 'rgba(30,10,10,0.9)', border: '1px solid #f87171',
            borderRadius: 4, padding: '4px 12px', fontSize: 11, color: '#f87171',
          }}>
            {failedText || '실패'}
          </div>
        )}

        <div style={{ position: 'absolute', top: 8, left: 8, color: '#fff', fontSize: 11, fontFamily: 'monospace', textShadow: '1px 1px 2px #000' }}>
          IM: {sliceIndex + 1}
        </div>
        <div style={{ position: 'absolute', bottom: 8, left: 8, color: '#aaa', fontSize: 11, fontFamily: 'monospace' }}>
          WC: {wc} / WW: {ww}
        </div>
      </div>
    </div>
  );
}

const overlayStyle = {
  position: 'absolute', inset: 0,
  background: 'rgba(5,5,20,0.70)',
  display: 'flex', flexDirection: 'column',
  alignItems: 'center', justifyContent: 'center', gap: 16,
};

const spinnerStyle = {
  width: 52, height: 52,
  border: '4px solid #ce93d8',
  borderTopColor: 'transparent',
  borderRadius: '50%',
  animation: 'spin 0.8s linear infinite',
};
