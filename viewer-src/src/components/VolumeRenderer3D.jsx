import React, { useEffect, useRef, useState } from 'react';

const RENDER_MODES = [
  { key: 'surface', label: '표면' },
  { key: 'mip',     label: 'MIP'  },
  { key: 'xray',    label: 'X-Ray'},
];

const CANVAS_SIZE = 240;

export default function VolumeRenderer3D({ slices, sliceIndex = 0 }) {
  const canvasRef  = useRef(null);
  const [angleX, setAngleX]         = useState(15);
  const [angleY, setAngleY]         = useState(-25);
  const [renderMode, setRenderMode] = useState('surface');
  const [isDragging, setIsDragging] = useState(false);

  const dragging    = useRef(false);
  const lastMouse   = useRef(null);
  const animFrame   = useRef(null);
  const stateRef    = useRef({ angleX: 15, angleY: -25, renderMode: 'surface', dragging: false, sliceIndex: 0 });
  const volumeRef   = useRef(null);
  const depthBufRef = useRef(new Float32Array(CANVAS_SIZE * CANVAS_SIZE).fill(Infinity));

  useEffect(() => {
    if (!slices || slices.length === 0) return;
    volumeRef.current = buildVolume(slices);
  }, [slices]);

  useEffect(() => {
    stateRef.current = { angleX, angleY, renderMode, dragging: isDragging, sliceIndex };
    scheduleRender();
  }, [angleX, angleY, renderMode, isDragging, slices, sliceIndex]);

  function scheduleRender() {
    if (animFrame.current) cancelAnimationFrame(animFrame.current);
    animFrame.current = requestAnimationFrame(() => {
      if (!canvasRef.current || !volumeRef.current) return;
      const { angleX: ax, angleY: ay, renderMode: rm, dragging: dr, sliceIndex: si } = stateRef.current;

      // surface 모드만 뎁스 버퍼로 정확한 가림 처리
      let depthBuf = null;
      if (rm === 'surface') {
        depthBufRef.current.fill(Infinity);
        depthBuf = depthBufRef.current;
      }

      if (rm === 'surface') renderIsosurface(canvasRef.current, volumeRef.current, ax, ay, dr, depthBuf);
      else if (rm === 'mip') renderMIP(canvasRef.current, volumeRef.current, ax, ay, dr);
      else                   renderXRay(canvasRef.current, volumeRef.current, ax, ay, dr);

      drawSlicePlane(canvasRef.current, volumeRef.current, ax, ay, si, depthBuf);
    });
  }

  const onMouseDown = (e) => {
    dragging.current = true;
    setIsDragging(true);
    lastMouse.current = { x: e.clientX, y: e.clientY };
  };
  const onMouseMove = (e) => {
    if (!dragging.current) return;
    const dx = e.clientX - lastMouse.current.x;
    const dy = e.clientY - lastMouse.current.y;
    lastMouse.current = { x: e.clientX, y: e.clientY };
    setAngleY(a => a + dx * 0.9);
    setAngleX(a => Math.max(-80, Math.min(80, a + dy * 0.9)));
  };
  const onMouseUp = () => {
    dragging.current = false;
    setIsDragging(false);
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', background: '#050510' }}>
      {/* 모드 버튼 */}
      <div style={{ padding: '5px 8px', background: '#0d0d1a', borderBottom: '1px solid #2a2a4a', display: 'flex', alignItems: 'center', gap: 5 }}>
        <span style={{ fontSize: 10, color: '#666', marginRight: 2 }}>3D</span>
        {RENDER_MODES.map(m => (
          <button
            key={m.key}
            onClick={() => setRenderMode(m.key)}
            style={{
              fontSize: 10, padding: '2px 7px', borderRadius: 3, cursor: 'pointer',
              background: renderMode === m.key ? '#2a1a4a' : 'transparent',
              color:      renderMode === m.key ? '#ce93d8' : '#555',
              border:     renderMode === m.key ? '1px solid #5a3a8a' : '1px solid #2a2a4a',
            }}
          >
            {m.label}
          </button>
        ))}
        <span style={{ marginLeft: 'auto', fontSize: 9, color: '#333' }}>드래그 회전</span>
      </div>

      {/* 캔버스 */}
      <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', position: 'relative' }}>
        {(!slices || slices.length === 0) ? (
          <div style={{ color: '#444', fontSize: 12 }}>로딩 중...</div>
        ) : (
          <canvas
            ref={canvasRef}
            width={CANVAS_SIZE}
            height={CANVAS_SIZE}
            style={{ cursor: isDragging ? 'grabbing' : 'grab', borderRadius: 4, border: '1px solid #1a1a2a', maxWidth: '100%' }}
            onMouseDown={onMouseDown}
            onMouseMove={onMouseMove}
            onMouseUp={onMouseUp}
            onMouseLeave={onMouseUp}
          />
        )}
        <div style={{ position: 'absolute', bottom: 4, right: 6, fontSize: 9, color: '#333', fontFamily: 'monospace' }}>
          {Math.round(angleX)}° / {Math.round(angleY)}°
        </div>
      </div>
    </div>
  );
}

// ── 볼륨 빌드 ─────────────────────────────────────────────────────────────────
function buildVolume(slices) {
  const D_src = slices.length;
  const srcW  = slices[0].width;
  const srcH  = slices[0].height;

  const stepXY = 3;
  const W = Math.floor(srcW / stepXY);
  const H = Math.floor(srcH / stepXY);
  const stepZ = stepXY;
  const D = Math.floor(D_src / stepZ);

  const vol = new Uint8Array(W * H * D);
  for (let z = 0; z < D; z++) {
    const srcZ = Math.min(D_src - 1, z * stepZ);
    const src  = slices[srcZ].data;
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        vol[z * W * H + y * W + x] = Math.max(0, Math.min(255,
          src[(y * stepXY) * srcW + x * stepXY]
        ));
      }
    }
  }
  return { data: vol, W, H, D, stepZ };
}

// ── 공통: 회전 ────────────────────────────────────────────────────────────────
function rotate(sx, sy, t, ax, ay) {
  const cosY = Math.cos(ay), sinY = Math.sin(ay);
  const cosX = Math.cos(ax), sinX = Math.sin(ax);
  const rx  =  sx * cosY + t  * sinY;
  const rz1 = -sx * sinY + t  * cosY;
  const ry  =  sy * cosX - rz1 * sinX;
  const rz  =  sy * sinX + rz1 * cosX;
  return { rx, ry, rz };
}

function sampleVol(data, W, H, D, vx, vy, vz) {
  if (vx < 0 || vx >= W || vy < 0 || vy >= H || vz < 0 || vz >= D) return 0;
  return data[vz * W * H + vy * W + vx];
}

// ── 1. 등위면 (Isosurface + Phong) + 뎁스 버퍼 ──────────────────────────────
function renderIsosurface(canvas, vol, angleX, angleY, fast, depthBuf) {
  const ctx = canvas.getContext('2d');
  const cw  = canvas.width, ch = canvas.height;
  const img = ctx.createImageData(cw, ch);
  const buf = img.data;

  const { data, W, H, D } = vol;
  const THRESHOLD = 45;
  const STEP = fast ? 1.5 : 0.8;

  const ax = angleX * Math.PI / 180, ay = angleY * Math.PI / 180;
  const cx2 = W/2, cy2 = H/2, cz2 = D/2;
  const scale   = Math.min(cw / W, ch / H) * 0.85;
  const rayLen  = Math.max(W, H, D) * 1.6;

  const lx = 0.4, ly = -0.7, lz = 0.6;
  const lLen = Math.sqrt(lx*lx + ly*ly + lz*lz);
  const lnx = lx/lLen, lny = ly/lLen, lnz = lz/lLen;

  for (let py = 0; py < ch; py++) {
    for (let px = 0; px < cw; px++) {
      const sx = (px - cw/2) / scale;
      const sy = (py - ch/2) / scale;
      let hit = false;

      for (let t = -rayLen/2; t < rayLen/2; t += STEP) {
        const { rx, ry, rz } = rotate(sx, sy, t, ax, ay);
        const vx = rx + cx2, vy = ry + cy2, vz = rz + cz2;
        const v  = sampleVol(data, W, H, D, Math.round(vx), Math.round(vy), Math.round(vz));

        if (v >= THRESHOLD) {
          const gx = sampleVol(data,W,H,D,Math.round(vx+1),Math.round(vy),Math.round(vz))
                   - sampleVol(data,W,H,D,Math.round(vx-1),Math.round(vy),Math.round(vz));
          const gy = sampleVol(data,W,H,D,Math.round(vx),Math.round(vy+1),Math.round(vz))
                   - sampleVol(data,W,H,D,Math.round(vx),Math.round(vy-1),Math.round(vz));
          const gz = sampleVol(data,W,H,D,Math.round(vx),Math.round(vy),Math.round(vz+1))
                   - sampleVol(data,W,H,D,Math.round(vx),Math.round(vy),Math.round(vz-1));
          const gLen = Math.sqrt(gx*gx + gy*gy + gz*gz) || 1;
          const nx = gx/gLen, ny = gy/gLen, nz = gz/gLen;

          const diffuse  = Math.max(0, -(nx*lnx + ny*lny + nz*lnz));
          const specPow  = Math.pow(Math.max(0, diffuse), 8) * 0.3;
          const intensity = Math.min(1, 0.25 + diffuse * 0.75 + specPow);

          const i4 = (py * cw + px) * 4;
          buf[i4]   = Math.round(180 * intensity);
          buf[i4+1] = Math.round(148 * intensity);
          buf[i4+2] = Math.round(140 * intensity);
          buf[i4+3] = 255;

          if (depthBuf) depthBuf[py * cw + px] = t;
          hit = true;
          break;
        }
      }

      if (!hit) {
        const i4 = (py * cw + px) * 4;
        buf[i4] = 4; buf[i4+1] = 4; buf[i4+2] = 10; buf[i4+3] = 255;
      }
    }
  }

  ctx.putImageData(img, 0, 0);
  ctx.save(); ctx.font = '9px monospace'; ctx.fillStyle = '#ce93d8';
  ctx.fillText('Surface', 6, 14); ctx.restore();
}

// ── 2. MIP ────────────────────────────────────────────────────────────────────
function renderMIP(canvas, vol, angleX, angleY, fast) {
  const ctx = canvas.getContext('2d');
  const cw  = canvas.width, ch = canvas.height;
  const img = ctx.createImageData(cw, ch);
  const buf = img.data;

  const { data, W, H, D } = vol;
  const STEP = fast ? 1.5 : 0.9;
  const ax = angleX * Math.PI / 180, ay = angleY * Math.PI / 180;
  const cx2 = W/2, cy2 = H/2, cz2 = D/2;
  const scale  = Math.min(cw/W, ch/H) * 0.85;
  const rayLen = Math.max(W, H, D) * 1.6;

  for (let py = 0; py < ch; py++) {
    for (let px = 0; px < cw; px++) {
      const sx = (px - cw/2) / scale, sy = (py - ch/2) / scale;
      let maxV = 0;
      for (let t = -rayLen/2; t < rayLen/2; t += STEP) {
        const { rx, ry, rz } = rotate(sx, sy, t, ax, ay);
        const v = sampleVol(data, W, H, D, Math.round(rx+cx2), Math.round(ry+cy2), Math.round(rz+cz2));
        if (v > maxV) maxV = v;
      }
      const i4 = (py * cw + px) * 4;
      buf[i4] = maxV; buf[i4+1] = Math.round(maxV*0.85); buf[i4+2] = Math.round(maxV*0.8); buf[i4+3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
  ctx.save(); ctx.font = '9px monospace'; ctx.fillStyle = '#ce93d8'; ctx.fillText('MIP', 6, 14); ctx.restore();
}

// ── 3. X-Ray ─────────────────────────────────────────────────────────────────
function renderXRay(canvas, vol, angleX, angleY, fast) {
  const ctx = canvas.getContext('2d');
  const cw  = canvas.width, ch = canvas.height;
  const img = ctx.createImageData(cw, ch);
  const buf = img.data;

  const { data, W, H, D } = vol;
  const STEP = fast ? 2.0 : 1.0;
  const ax = angleX * Math.PI / 180, ay = angleY * Math.PI / 180;
  const cx2 = W/2, cy2 = H/2, cz2 = D/2;
  const scale  = Math.min(cw/W, ch/H) * 0.85;
  const rayLen = Math.max(W, H, D) * 1.6;

  for (let py = 0; py < ch; py++) {
    for (let px = 0; px < cw; px++) {
      const sx = (px - cw/2) / scale, sy = (py - ch/2) / scale;
      let acc = 0, cnt = 0;
      for (let t = -rayLen/2; t < rayLen/2; t += STEP) {
        const { rx, ry, rz } = rotate(sx, sy, t, ax, ay);
        const v = sampleVol(data, W, H, D, Math.round(rx+cx2), Math.round(ry+cy2), Math.round(rz+cz2));
        if (v > 20) { acc += v; cnt++; }
      }
      const i4 = (py * cw + px) * 4;
      if (cnt === 0) {
        buf[i4] = 4; buf[i4+1] = 4; buf[i4+2] = 10; buf[i4+3] = 255;
      } else {
        const g = Math.min(255, (acc / cnt) * 1.2);
        buf[i4] = Math.round(g*0.7); buf[i4+1] = Math.round(g*0.9); buf[i4+2] = Math.round(g); buf[i4+3] = 255;
      }
    }
  }
  ctx.putImageData(img, 0, 0);
  ctx.save(); ctx.font = '9px monospace'; ctx.fillStyle = '#ce93d8'; ctx.fillText('X-Ray', 6, 14); ctx.restore();
}

// ── 볼록 사각형 내부 판정 ─────────────────────────────────────────────────────
function isInsideQuad(px, py, corners) {
  let allPos = true, allNeg = true;
  for (let i = 0; i < 4; i++) {
    const ax = corners[(i + 1) % 4][0] - corners[i][0];
    const ay = corners[(i + 1) % 4][1] - corners[i][1];
    const cross = ax * (py - corners[i][1]) - ay * (px - corners[i][0]);
    if (cross < 0) allPos = false;
    if (cross > 0) allNeg = false;
  }
  return allPos || allNeg;
}

// ── 슬라이스 위치 평면 오버레이 ───────────────────────────────────────────────
function drawSlicePlane(canvas, vol, angleX, angleY, sliceIndex, depthBuf) {
  const { W, H, D, stepZ = 3 } = vol;
  const cw = canvas.width, ch = canvas.height;
  const scale = Math.min(cw / W, ch / H) * 0.85;

  const ax = angleX * Math.PI / 180, ay = angleY * Math.PI / 180;
  const cosX = Math.cos(ax), sinX = Math.sin(ax);
  const cosY = Math.cos(ay), sinY = Math.sin(ay);

  const vz = Math.min(D - 1, Math.max(0, Math.round(sliceIndex / stepZ)));
  const wz = vz - D / 2;

  const proj = (wx, wy, wzv) => {
    const sx = wx * cosY + wy * sinX * sinY - wzv * cosX * sinY;
    const sy = wy * cosX + wzv * sinX;
    return [sx * scale + cw / 2, sy * scale + ch / 2];
  };

  const hw = W / 2, hh = H / 2;
  const corners = [
    proj(-hw, -hh, wz),
    proj( hw, -hh, wz),
    proj( hw,  hh, wz),
    proj(-hw,  hh, wz),
  ];

  const ctx = canvas.getContext('2d');

  // ── 반투명 면 채움 (뎁스 버퍼로 가림 처리) ──────────────────────────────────
  if (depthBuf) {
    // t_slice(px,py): rz=wz 가 되는 t 값 (카메라에서의 깊이)
    // rz = sy*sinX + (-sx*sinY + t*cosY)*cosX = wz
    // → t = (wz - sy*sinX + sx*sinY*cosX) / (cosY*cosX)
    const denom = cosY * cosX;

    const imgData = ctx.getImageData(0, 0, cw, ch);
    const buf = imgData.data;

    const minPx = Math.max(0, Math.floor(Math.min(corners[0][0], corners[1][0], corners[2][0], corners[3][0])));
    const maxPx = Math.min(cw - 1, Math.ceil (Math.max(corners[0][0], corners[1][0], corners[2][0], corners[3][0])));
    const minPy = Math.max(0, Math.floor(Math.min(corners[0][1], corners[1][1], corners[2][1], corners[3][1])));
    const maxPy = Math.min(ch - 1, Math.ceil (Math.max(corners[0][1], corners[1][1], corners[2][1], corners[3][1])));

    const FILL_A = 0.22;  // 면 불투명도
    for (let py = minPy; py <= maxPy; py++) {
      for (let px = minPx; px <= maxPx; px++) {
        if (!isInsideQuad(px, py, corners)) continue;

        // 이 픽셀에서 슬라이스 평면의 카메라 깊이
        const sx = (px - cw / 2) / scale;
        const sy = (py - ch / 2) / scale;
        const t_slice = denom !== 0
          ? (wz - sy * sinX + sx * sinY * cosX) / denom
          : Infinity;

        // 뇌 표면보다 뒤에 있으면 건너뜀 (뇌가 앞을 가림)
        if (t_slice > depthBuf[py * cw + px]) continue;

        const i4 = (py * cw + px) * 4;
        buf[i4]     = Math.round(buf[i4]     * (1 - FILL_A) + 0   * FILL_A);
        buf[i4 + 1] = Math.round(buf[i4 + 1] * (1 - FILL_A) + 200 * FILL_A);
        buf[i4 + 2] = Math.round(buf[i4 + 2] * (1 - FILL_A) + 255 * FILL_A);
      }
    }
    ctx.putImageData(imgData, 0, 0);
  } else {
    // MIP / X-Ray 모드: 뎁스 없이 전체 반투명 채움
    ctx.save();
    ctx.beginPath();
    ctx.moveTo(...corners[0]);
    for (let i = 1; i < 4; i++) ctx.lineTo(...corners[i]);
    ctx.closePath();
    ctx.fillStyle = 'rgba(0, 200, 255, 0.15)';
    ctx.fill();
    ctx.restore();
  }

  // ── 외곽선 ──────────────────────────────────────────────────────────────────
  ctx.save();
  ctx.beginPath();
  ctx.moveTo(...corners[0]);
  for (let i = 1; i < 4; i++) ctx.lineTo(...corners[i]);
  ctx.closePath();
  ctx.strokeStyle = 'rgba(0, 230, 255, 0.6)';
  ctx.lineWidth = 1;
  ctx.setLineDash([5, 3]);
  ctx.stroke();

  // ── 중심 십자 ────────────────────────────────────────────────────────────────
  const cxPt = proj(0, 0, wz);
  ctx.setLineDash([]);
  ctx.strokeStyle = 'rgba(0, 230, 255, 0.9)';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(cxPt[0] - 5, cxPt[1]); ctx.lineTo(cxPt[0] + 5, cxPt[1]);
  ctx.moveTo(cxPt[0], cxPt[1] - 5); ctx.lineTo(cxPt[0], cxPt[1] + 5);
  ctx.stroke();

  // ── 레이블 ───────────────────────────────────────────────────────────────────
  ctx.fillStyle = 'rgba(0, 230, 255, 0.85)';
  ctx.font = '9px monospace';
  ctx.fillText(`z:${sliceIndex + 1}`, cxPt[0] + 7, cxPt[1] - 3);

  ctx.restore();
}
